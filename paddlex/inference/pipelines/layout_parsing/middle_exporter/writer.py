from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np

from .builder import layout_parsing_result_to_middle_page
from .geometry import pdf_bottom_left_box_to_img
from ..merge_table import can_merge_tables, perform_table_merge
from ..utils import get_seg_flag


def _persist_image_assets(
    page_dict: Dict[str, Any], page_result: Any, output_dir: Union[str, Path]
) -> None:
    if "parsing_res_list" not in page_result:
        return
    parsing_res_list = page_result["parsing_res_list"]
    page = page_dict["pdf_info"][0]
    for para in page.get("para_blocks", []):
        if para.get("type") not in ("image", "seal"):
            continue
        block_idx = para.get("index", None)
        if block_idx is None:
            continue
        if not isinstance(block_idx, int) or block_idx < 0 or block_idx >= len(parsing_res_list):
            continue
        block = parsing_res_list[block_idx]
        if not getattr(block, "image", None) or not isinstance(block.image, dict):
            continue
        img_rel_path = str(block.image.get("path", "") or "")
        img_obj = block.image.get("img", None)
        if not img_rel_path or img_obj is None:
            continue
        save_path = Path(output_dir) / img_rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            img_obj.save(str(save_path))
        except Exception:
            continue


def write_middle_index_json(
    output_dir: Union[str, Path], total_pages: int, *, pages_subdir: str = "middle_pages"
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = {
        "output_directory": pages_subdir,
        "total_pages": int(total_pages),
        "pages": [
            {"page_idx": i, "path": f"{pages_subdir}/page_{i:04d}.json", "file_name": f"page_{i:04d}.json"}
            for i in range(int(total_pages))
        ],
    }
    p = out / "middle_index.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
    return p


def save_middle_page_json(
    page_dict: Dict[str, Any], output_dir: Union[str, Path], *, pages_subdir: str = "middle_pages"
) -> Path:
    out = Path(output_dir) / pages_subdir
    out.mkdir(parents=True, exist_ok=True)
    page_idx = int(page_dict["pdf_info"][0]["page_idx"])
    path = out / f"page_{page_idx:04d}.json"
    path.write_text(json.dumps(page_dict, ensure_ascii=False, indent=4), encoding="utf-8")
    return path


def save_middle_page_visualization(
    page_dict: Dict[str, Any], page_result: Any, output_dir: Union[str, Path], *, vis_subdir: str = "middle_vis"
) -> Optional[Path]:
    if "doc_preprocessor_res" not in page_result:
        return None
    img = page_result["doc_preprocessor_res"].get("output_img", None)
    if img is None:
        return None
    vis = np.array(img).copy()
    if vis.ndim != 3 or vis.shape[2] != 3:
        return None
    page = page_dict["pdf_info"][0]
    pdf_w, pdf_h = float(page["page_size"][0]), float(page["page_size"][1])
    img_h, img_w = vis.shape[0], vis.shape[1]
    for para in page.get("para_blocks", []):
        for line in para.get("lines", []):
            lb = pdf_bottom_left_box_to_img(line["bbox"], img_w, img_h, pdf_w, pdf_h)
            cv2.rectangle(vis, (lb[0], lb[1]), (lb[2], lb[3]), (0, 180, 0), 2)
            for span in line.get("spans", []):
                sb = pdf_bottom_left_box_to_img(span["bbox"], img_w, img_h, pdf_w, pdf_h)
                cv2.rectangle(vis, (sb[0], sb[1]), (sb[2], sb[3]), (255, 0, 0), 1)
    out = Path(output_dir) / vis_subdir
    out.mkdir(parents=True, exist_ok=True)
    page_idx = int(page["page_idx"])
    path = out / f"page_{page_idx:04d}_line_span_vis.png"
    cv2.imwrite(str(path), vis)
    return path


def _is_chinese_char(ch: str) -> bool:
    return bool(ch) and re.match(r"[\u4e00-\u9fff]", ch) is not None


def _needs_boundary_space(left_text: str, right_text: str) -> bool:
    last_char = left_text[-1] if left_text else ""
    first_char = right_text[0] if right_text else ""
    if not last_char or not first_char:
        return False
    return not (_is_chinese_char(last_char) or _is_chinese_char(first_char))


def _mark_lines_cross_page(lines: Any) -> None:
    if not isinstance(lines, list):
        return
    for line in lines:
        if not isinstance(line, dict):
            continue
        spans = line.get("spans")
        if isinstance(spans, list):
            for span in spans:
                if isinstance(span, dict):
                    span["cross_page"] = 1
        else:
            line["cross_page"] = 1


def _first_para_of_type(page_dict: Dict[str, Any], para_type: str) -> Optional[Dict[str, Any]]:
    paras = page_dict.get("pdf_info", [{}])[0].get("para_blocks", [])
    for para in paras:
        if isinstance(para, dict) and para.get("type") == para_type:
            return para
    return None


def _last_para_of_type(page_dict: Dict[str, Any], para_type: str) -> Optional[Dict[str, Any]]:
    paras = page_dict.get("pdf_info", [{}])[0].get("para_blocks", [])
    for para in reversed(paras):
        if isinstance(para, dict) and para.get("type") == para_type:
            return para
    return None


def _first_text_block_and_seg_flag(page_result: Any) -> tuple[Optional[Any], bool]:
    parsing_res_list = page_result.get("parsing_res_list", [])
    prev_block = None
    for block in parsing_res_list:
        seg_start_flag, _seg_end_flag = get_seg_flag(block, prev_block)
        prev_block = block
        if str(getattr(block, "label", "") or "") == "text":
            return block, bool(seg_start_flag)
    return None, True


def _last_text_block(page_result: Any) -> Optional[Any]:
    parsing_res_list = page_result.get("parsing_res_list", [])
    for block in reversed(parsing_res_list):
        if str(getattr(block, "label", "") or "") == "text":
            return block
    return None


def _get_para_last_span(para: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    lines = para.get("lines")
    if not isinstance(lines, list):
        return None
    for line in reversed(lines):
        if not isinstance(line, dict):
            continue
        spans = line.get("spans")
        if not isinstance(spans, list):
            continue
        for span in reversed(spans):
            if isinstance(span, dict):
                return span
    return None


def _get_para_first_span(para: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    lines = para.get("lines")
    if not isinstance(lines, list):
        return None
    for line in lines:
        if not isinstance(line, dict):
            continue
        spans = line.get("spans")
        if not isinstance(spans, list):
            continue
        for span in spans:
            if isinstance(span, dict):
                return span
    return None


def _find_table_body_block(table_para: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    blocks = table_para.get("blocks")
    if not isinstance(blocks, list):
        return None
    for blk in blocks:
        if isinstance(blk, dict) and blk.get("type") == "table_body":
            return blk
    return None


def _merge_middle_text_across_pages(page_dicts: List[Dict[str, Any]], results: Sequence[Any]) -> None:
    for i in range(1, min(len(page_dicts), len(results))):
        prev_text_para = _last_para_of_type(page_dicts[i - 1], "text")
        curr_text_para = _first_para_of_type(page_dicts[i], "text")
        if prev_text_para is None or curr_text_para is None:
            continue

        prev_text_block = _last_text_block(results[i - 1])
        curr_text_block, curr_seg_start_flag = _first_text_block_and_seg_flag(results[i])
        if prev_text_block is None or curr_text_block is None:
            continue
        if curr_seg_start_flag:
            continue

        prev_last_span = _get_para_last_span(prev_text_para)
        curr_first_span = _get_para_first_span(curr_text_para)
        prev_last_text = str(prev_last_span.get("content", "") or "") if isinstance(prev_last_span, dict) else ""
        curr_first_text = str(curr_first_span.get("content", "") or "") if isinstance(curr_first_span, dict) else ""
        if _needs_boundary_space(prev_last_text, curr_first_text) and isinstance(prev_last_span, dict):
            prev_last_span["content"] = prev_last_text + " "

        prev_lines = prev_text_para.get("lines")
        curr_lines = curr_text_para.get("lines")
        if not isinstance(prev_lines, list):
            prev_lines = []
            prev_text_para["lines"] = prev_lines
        if isinstance(curr_lines, list):
            _mark_lines_cross_page(curr_lines)
            prev_lines.extend(curr_lines)

        curr_text_para["lines"] = []
        curr_text_para["lines_deleted"] = 1


def _merge_middle_table_across_pages(page_dicts: List[Dict[str, Any]], results: Sequence[Any]) -> None:
    for i in range(1, min(len(page_dicts), len(results))):
        prev_table_para = _last_para_of_type(page_dicts[i - 1], "table")
        curr_table_para = _first_para_of_type(page_dicts[i], "table")
        if prev_table_para is None or curr_table_para is None:
            continue

        page_prev_blocks = list(results[i - 1].get("parsing_res_list", []))
        page_curr_blocks = list(results[i].get("parsing_res_list", []))
        prev_table_block = next((b for b in reversed(page_prev_blocks) if getattr(b, "label", "") == "table"), None)
        curr_table_block = next((b for b in page_curr_blocks if getattr(b, "label", "") == "table"), None)
        if prev_table_block is None or curr_table_block is None:
            continue

        can_merge, soup_prev, soup_curr = can_merge_tables(
            page_prev_blocks, prev_table_block, page_curr_blocks, curr_table_block
        )
        if not can_merge:
            continue

        prev_table_body = _find_table_body_block(prev_table_para)
        curr_table_body = _find_table_body_block(curr_table_para)
        if prev_table_body is None or curr_table_body is None:
            continue

        merged_html = perform_table_merge(soup_prev, soup_curr)
        curr_lines = curr_table_body.get("lines")
        if isinstance(curr_lines, list):
            _mark_lines_cross_page(curr_lines)
            # Replace previous-page half-table lines to avoid duplicated html content.
            prev_table_body["lines"] = curr_lines
            for line in curr_lines:
                if not isinstance(line, dict):
                    continue
                spans = line.get("spans")
                if not isinstance(spans, list):
                    continue
                for span in spans:
                    if isinstance(span, dict) and span.get("type") == "table":
                        span["html"] = merged_html
                        break
                else:
                    continue
                break

        curr_table_body["lines"] = []
        curr_table_para["lines"] = []
        curr_table_para["lines_deleted"] = 1


def _apply_cross_page_merge_to_middle(page_dicts: List[Dict[str, Any]], results: Sequence[Any]) -> None:
    _merge_middle_text_across_pages(page_dicts, results)
    _merge_middle_table_across_pages(page_dicts, results)


def export_middle_bundle(
    results: Sequence[Any],
    output_dir: Union[str, Path],
    *,
    min_ocr_coverage: float = 0.7,
    pages_subdir: str = "middle_pages",
    char_source: str = "pdf_text",
    save_vis: bool = False,
    vis_subdir: str = "middle_vis",
) -> None:
    if not results:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = int(results[0]["page_count"]) if "page_count" in results[0] and results[0]["page_count"] is not None else len(results)
    write_middle_index_json(out, total, pages_subdir=pages_subdir)
    page_dicts: List[Dict[str, Any]] = []
    for item in results:
        page_dict = layout_parsing_result_to_middle_page(
            item, min_ocr_coverage=min_ocr_coverage, char_source=char_source
        )
        page_dicts.append(page_dict)
    _apply_cross_page_merge_to_middle(page_dicts, results)
    for item, page_dict in zip(results, page_dicts):
        _persist_image_assets(page_dict, item, out)
        save_middle_page_json(page_dict, out, pages_subdir=pages_subdir)
        if save_vis:
            save_middle_page_visualization(page_dict, item, out, vis_subdir=vis_subdir)

