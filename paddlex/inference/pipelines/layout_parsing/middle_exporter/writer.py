from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import cv2
import numpy as np

from .builder import layout_parsing_result_to_middle_page
from .geometry import pdf_bottom_left_box_to_img


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
    for item in results:
        page_dict = layout_parsing_result_to_middle_page(
            item, min_ocr_coverage=min_ocr_coverage, char_source=char_source
        )
        _persist_image_assets(page_dict, item, out)
        save_middle_page_json(page_dict, out, pages_subdir=pages_subdir)
        if save_vis:
            save_middle_page_visualization(page_dict, item, out, vis_subdir=vis_subdir)

