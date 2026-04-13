from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..layout_objects import LayoutBlock
from ..setting import BLOCK_LABEL_MAP
from .geometry import bbox_union, img_axis_aligned_box_to_pdf_top_left, read_pdf_page_size_pts
from .grouping import assign_ocr_indices_exclusive_by_coverage, group_spans_to_lines
from .pdfium_chars import approx_char_bboxes, char_positions_from_pdfium_chars, read_pdfium_page_chars


def layout_label_to_middle_para_type(label: str) -> str:
    low = label.lower()
    if low in BLOCK_LABEL_MAP["doc_title_labels"]:
        return "title"
    if low in BLOCK_LABEL_MAP["paragraph_title_labels"]:
        return "title"
    if low in BLOCK_LABEL_MAP["vision_title_labels"]:
        return "title"
    if low == "table":
        return "table"
    if low in BLOCK_LABEL_MAP["image_labels"] or low in ("chart", "flowchart", "figure"):
        return "image"
    if low == "formula":
        return "interline_equation"
    return "text"


def merge_line_spans_by_runs(line_spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(line_spans) <= 1:
        return line_spans
    spans = sorted(line_spans, key=lambda s: float(s["span_bbox_img"][0]))
    widths = [max(1.0, float(s["span_bbox_img"][2]) - float(s["span_bbox_img"][0])) for s in spans]
    median_w = float(np.median(np.asarray(widths, dtype=np.float64))) if widths else 8.0
    merge_gap = max(2.0, 0.8 * median_w)
    merged: List[Dict[str, Any]] = []
    cur = dict(spans[0])
    cur["char_positions"] = list(cur.get("char_positions", []))
    for nxt in spans[1:]:
        cur_box, nxt_box = cur["span_bbox_img"], nxt["span_bbox_img"]
        gap = float(nxt_box[0]) - float(cur_box[2])
        same_tag = cur.get("type") == nxt.get("type") and cur.get("char_source") == nxt.get("char_source")
        if same_tag and gap <= merge_gap:
            cur["content"] = f"{cur.get('content', '')}{nxt.get('content', '')}"
            cur["span_bbox_img"] = bbox_union([cur_box, nxt_box])
            cur["span_bbox_pdf"] = [round(v, 2) for v in bbox_union([cur["span_bbox_pdf"], nxt["span_bbox_pdf"]])]
            cur["score"] = float(max(float(cur.get("score", 0.0)), float(nxt.get("score", 0.0))))
            cur["char_positions"] = list(cur.get("char_positions", [])) + list(nxt.get("char_positions", []))
            for pos, ch in enumerate(cur["char_positions"]):
                ch["position_in_span"] = pos
        else:
            merged.append(cur)
            cur = dict(nxt)
            cur["char_positions"] = list(cur.get("char_positions", []))
    merged.append(cur)
    return merged


def layout_parsing_result_to_middle_page(
    result: Any, *, min_ocr_coverage: float = 0.7, char_source: str = "pdf_text"
) -> Dict[str, Any]:
    input_path = result["input_path"]
    page_idx = int(result["page_index"])
    img_w, img_h = int(result["width"]), int(result["height"])
    pdf_size = read_pdf_page_size_pts(str(input_path), page_idx)
    pdf_w, pdf_h = (float(img_w), float(img_h)) if pdf_size is None else pdf_size
    pdfium_chars = read_pdfium_page_chars(str(input_path), page_idx, img_w, img_h, pdf_w, pdf_h)

    parsing_res_list: List[LayoutBlock] = result["parsing_res_list"]
    overall_ocr = result["overall_ocr_res"]
    rec_boxes, rec_texts, rec_scores = overall_ocr["rec_boxes"], overall_ocr["rec_texts"], overall_ocr["rec_scores"]

    text_blocks: List[Tuple[int, np.ndarray]] = []
    for bi, block in enumerate(parsing_res_list):
        if layout_label_to_middle_para_type(block.label) in ("text", "title"):
            text_blocks.append((bi, np.asarray(block.bbox, dtype=np.float64)))
    assign = assign_ocr_indices_exclusive_by_coverage(rec_boxes, text_blocks, min_ocr_coverage)

    para_blocks: List[Dict[str, Any]] = []
    global_char_idx = 0
    for bi, block in enumerate(parsing_res_list):
        mtype = layout_label_to_middle_para_type(block.label)
        bbox_img = np.asarray(block.bbox, dtype=np.float64)
        pdf_bbox = img_axis_aligned_box_to_pdf_top_left(bbox_img, img_w, img_h, pdf_w, pdf_h)

        if mtype == "image":
            img_path = ""
            if getattr(block, "image", None) and isinstance(block.image, dict):
                img_path = str(block.image.get("path", "") or "")
            body_bbox = [round(v, 2) for v in pdf_bbox]
            para_blocks.append(
                {
                    "type": "image",
                    "bbox": body_bbox,
                    "blocks": [{"type": "image_body", "bbox": body_bbox, "group_id": 0, "lines": [{"bbox": body_bbox, "spans": [{"bbox": body_bbox, "score": 1.0, "type": "image", "image_path": img_path}]}], "index": bi}],
                    "index": bi,
                }
            )
            continue

        if mtype == "table":
            body_bbox = [round(v, 2) for v in pdf_bbox]
            para_blocks.append(
                {
                    "type": "table",
                    "bbox": body_bbox,
                    "blocks": [{"type": "table_body", "bbox": body_bbox, "group_id": 0, "lines": [{"bbox": body_bbox, "spans": [{"bbox": body_bbox, "score": 1.0, "type": "table", "content": block.content}]}], "index": bi}],
                    "index": bi,
                }
            )
            continue

        if mtype == "interline_equation":
            span_pdf = [float(x) for x in pdf_bbox]
            txt = block.content or ""
            char_positions, global_char_idx = approx_char_bboxes(txt, span_pdf, global_char_idx)
            para_blocks.append(
                {
                    "type": "interline_equation",
                    "bbox": [round(x, 2) for x in pdf_bbox],
                    "lines": [{"bbox": [round(x, 2) for x in span_pdf], "spans": [{"bbox": [round(x, 2) for x in span_pdf], "score": 1.0, "content": txt, "type": "interline_equation", "char_positions": char_positions, "char_source": "formula"}], "index": 0}],
                    "index": bi,
                }
            )
            continue

        span_candidates: List[Dict[str, Any]] = []
        for ocr_i in assign.get(bi, []):
            txt = str(rec_texts[ocr_i])
            if not txt:
                continue
            sc = float(rec_scores[ocr_i]) if ocr_i < len(rec_scores) else 1.0
            ocr_img = rec_boxes[ocr_i]
            span_pdf = img_axis_aligned_box_to_pdf_top_left(ocr_img, img_w, img_h, pdf_w, pdf_h)
            char_positions = char_positions_from_pdfium_chars(pdfium_chars, ocr_img)
            used_char_source = char_source
            if not char_positions:
                char_positions, global_char_idx = approx_char_bboxes(txt, span_pdf, global_char_idx)
                used_char_source = "approx_fallback"
            span_candidates.append(
                {
                    "span_bbox_img": [float(v) for v in ocr_img],
                    "span_bbox_pdf": [round(x, 2) for x in span_pdf],
                    "score": sc,
                    "content": txt,
                    "type": "text",
                    "char_positions": char_positions,
                    "char_source": used_char_source,
                }
            )
        lines_out: List[Dict[str, Any]] = []
        for line_i, line_spans in enumerate(group_spans_to_lines(span_candidates)):
            line_spans = merge_line_spans_by_runs(line_spans)
            line_img_box = bbox_union([s["span_bbox_img"] for s in line_spans])
            line_pdf = img_axis_aligned_box_to_pdf_top_left(line_img_box, img_w, img_h, pdf_w, pdf_h)
            lines_out.append(
                {
                    "bbox": [round(x, 2) for x in line_pdf],
                    "spans": [{"bbox": s["span_bbox_pdf"], "score": s["score"], "content": s["content"], "type": s["type"], "char_positions": s["char_positions"], "char_source": s["char_source"]} for s in line_spans],
                    "index": line_i,
                }
            )
        if not lines_out and (block.content or "").strip():
            span_pdf = pdf_bbox
            txt = block.content
            char_positions, global_char_idx = approx_char_bboxes(txt, span_pdf, global_char_idx)
            lines_out.append(
                {
                    "bbox": [round(x, 2) for x in span_pdf],
                    "spans": [{"bbox": [round(x, 2) for x in span_pdf], "score": 1.0, "content": txt, "type": "text", "char_positions": char_positions, "char_source": "block_fallback"}],
                    "index": 0,
                }
            )
        para_blocks.append({"type": mtype, "bbox": [round(x, 2) for x in pdf_bbox], "lines": lines_out, "index": bi})

    para_blocks = [p for p in para_blocks if p.get("type") not in ("text", "title") or len(p.get("lines", [])) > 0]
    return {"pdf_info": [{"page_idx": page_idx, "page_size": [int(round(pdf_w)), int(round(pdf_h))], "para_blocks": para_blocks}]}

