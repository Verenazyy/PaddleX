"""Backward-compatible middle export API.

Core implementation has been moved into `layout_parsing/middle_exporter/`.
Keep this module as a stable import surface for existing callers.
"""

from .middle_exporter import (
    MiddleExportConfig,
    assign_ocr_indices_exclusive_by_coverage,
    build_layout_block_to_ocr_exclusive,
    export_middle_bundle,
    layout_parsing_result_to_middle_page,
    run_pp_structure_v3_to_middle,
    save_middle_page_json,
    save_middle_page_visualization,
    write_middle_index_json,
)

__all__ = [
    "MiddleExportConfig",
    "assign_ocr_indices_exclusive_by_coverage",
    "build_layout_block_to_ocr_exclusive",
    "layout_parsing_result_to_middle_page",
    "write_middle_index_json",
    "save_middle_page_json",
    "save_middle_page_visualization",
    "export_middle_bundle",
    "run_pp_structure_v3_to_middle",
]

# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build `middle_index.json` / `middle_pages/page_XXXX.json` style exports from layout parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import cv2

from .layout_objects import LayoutBlock
from .setting import BLOCK_LABEL_MAP
from .utils import calculate_bbox_area, get_bbox_intersection


def read_pdf_page_size_pts(input_path: str, page_index: int) -> Optional[Tuple[float, float]]:
    """Return (width, height) in PDF points for one page, or None if not a PDF / on error."""
    if not input_path or not str(input_path).lower().endswith(".pdf"):
        return None
    try:
        import pypdfium2 as pdfium  # type: ignore

        pdf = pdfium.PdfDocument(input_path)
        page = pdf.get_page(int(page_index))
        w, h = float(page.get_width()), float(page.get_height())
        if w > 0 and h > 0:
            return w, h
    except Exception:
        return None
    return None


def img_axis_aligned_box_to_pdf_top_left(
    box: Sequence[float],
    img_w: int,
    img_h: int,
    pdf_w: float,
    pdf_h: float,
) -> List[float]:
    """Map image pixel box (x_min,y_min,x_max,y_max, top-left origin, y down) to PDF user space with y down."""
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    px1 = x1 / img_w * pdf_w
    px2 = x2 / img_w * pdf_w
    py1 = y1 / img_h * pdf_h
    py2 = y2 / img_h * pdf_h
    return [px1, py1, px2, py2]


def pdf_top_left_box_to_img(
    box: Sequence[float],
    img_w: int,
    img_h: int,
    pdf_w: float,
    pdf_h: float,
) -> List[int]:
    """Map PDF top-left box to image pixel box."""
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    ix1 = int(round((x1 / pdf_w) * img_w))
    ix2 = int(round((x2 / pdf_w) * img_w))
    iy1 = int(round((y1 / pdf_h) * img_h))
    iy2 = int(round((y2 / pdf_h) * img_h))
    ix1 = max(0, min(img_w - 1, ix1))
    ix2 = max(0, min(img_w - 1, ix2))
    iy1 = max(0, min(img_h - 1, iy1))
    iy2 = max(0, min(img_h - 1, iy2))
    if ix2 < ix1:
        ix1, ix2 = ix2, ix1
    if iy2 < iy1:
        iy1, iy2 = iy2, iy1
    return [ix1, iy1, ix2, iy2]


def ocr_coverage_in_block(ocr_box: np.ndarray, block_box: np.ndarray) -> float:
    """Fraction of OCR box area covered by intersection with block (0~1)."""
    ocr_a = float(calculate_bbox_area(ocr_box))
    if ocr_a <= 1e-6:
        return 0.0
    inter = get_bbox_intersection(ocr_box, block_box, return_format="bbox")
    if inter is None:
        return 0.0
    return float(calculate_bbox_area(inter)) / ocr_a


def assign_ocr_indices_exclusive_by_coverage(
    rec_boxes: np.ndarray,
    blocks: Sequence[Tuple[int, np.ndarray]],
    min_coverage: float,
) -> Dict[int, List[int]]:
    """Each OCR box goes to at most one layout block — the one with largest coverage if >= min_coverage."""
    out: Dict[int, List[int]] = {block_idx: [] for block_idx, _ in blocks}
    for ocr_i in range(len(rec_boxes)):
        ocr_b = rec_boxes[ocr_i]
        best_cov = -1.0
        best_block: Optional[int] = None
        for block_idx, bbox in blocks:
            cov = ocr_coverage_in_block(ocr_b, bbox)
            if cov >= min_coverage and cov > best_cov:
                best_cov = cov
                best_block = block_idx
        if best_block is not None:
            out[best_block].append(ocr_i)
    for k in out:
        out[k].sort(
            key=lambda idx: (
                (float(rec_boxes[idx][1]) + float(rec_boxes[idx][3])) * 0.5,
                float(rec_boxes[idx][0]),
            )
        )
    return out


def build_layout_block_to_ocr_exclusive(
    layout_det_res: Any,
    overall_ocr_res: Any,
    min_coverage: float,
) -> Dict[int, List[int]]:
    """Like ``assign_ocr_indices_exclusive_by_coverage`` but from raw layout detection boxes."""
    blocks: List[Tuple[int, np.ndarray]] = []
    for box_idx, box_info in enumerate(layout_det_res["boxes"]):
        label = str(box_info["label"]).lower()
        if label in ("formula", "table", "seal"):
            continue
        blocks.append(
            (box_idx, np.asarray(box_info["coordinate"], dtype=np.float64)),
        )
    return assign_ocr_indices_exclusive_by_coverage(
        overall_ocr_res["rec_boxes"], blocks, min_coverage
    )


def _layout_label_to_middle_para_type(label: str) -> str:
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


def _approx_char_bboxes(
    text: str, pdf_box: Sequence[float], char_idx_start: int
) -> Tuple[List[Dict[str, Any]], int]:
    x1, y1, x2, y2 = map(float, pdf_box)
    chars = list(text)
    n = max(len(chars), 1)
    w = (x2 - x1) / n
    out: List[Dict[str, Any]] = []
    cur = char_idx_start
    for pos, ch in enumerate(chars):
        cx1 = x1 + pos * w
        cx2 = x1 + (pos + 1) * w
        out.append(
            {
                "char": ch,
                "bbox": [cx1, y1, cx2, y2],
                "char_idx": cur,
                "position_in_span": pos,
            }
        )
        cur += 1
    return out, cur


def _read_pdfium_page_chars(
    input_path: str,
    page_idx: int,
    img_w: int,
    img_h: int,
    pdf_w: float,
    pdf_h: float,
) -> List[Dict[str, Any]]:
    """Read per-char boxes from PDFium and map to both PDF(top-left) and image coordinates."""
    if not input_path or not str(input_path).lower().endswith(".pdf"):
        return []
    try:
        import pypdfium2 as pdfium  # type: ignore

        pdf = pdfium.PdfDocument(input_path)
        page = pdf.get_page(int(page_idx))
        textpage = page.get_textpage()
    except Exception:
        return []

    if not hasattr(textpage, "get_charbox"):
        return []
    if hasattr(textpage, "count_chars"):
        char_num = int(textpage.count_chars())
    elif hasattr(textpage, "get_char_count"):
        char_num = int(textpage.get_char_count())
    else:
        return []

    out: List[Dict[str, Any]] = []
    for ci in range(char_num):
        try:
            left, bottom, right, top = textpage.get_charbox(ci)
        except Exception:
            continue
        if right < left or top < bottom:
            continue

        ch = ""
        if hasattr(textpage, "get_text_range"):
            try:
                ch = textpage.get_text_range(ci, 1) or ""
            except Exception:
                ch = ""
        if not ch:
            continue

        x1_pdf = float(left)
        x2_pdf = float(right)
        y1_pdf = float(pdf_h - float(top))
        y2_pdf = float(pdf_h - float(bottom))
        if x2_pdf < x1_pdf:
            x1_pdf, x2_pdf = x2_pdf, x1_pdf
        if y2_pdf < y1_pdf:
            y1_pdf, y2_pdf = y2_pdf, y1_pdf

        x1_img = (x1_pdf / pdf_w) * img_w
        x2_img = (x2_pdf / pdf_w) * img_w
        y1_img = (y1_pdf / pdf_h) * img_h
        y2_img = (y2_pdf / pdf_h) * img_h

        out.append(
            {
                "char": ch,
                "char_idx": ci,
                "pdf_bbox": [x1_pdf, y1_pdf, x2_pdf, y2_pdf],
                "img_bbox": [x1_img, y1_img, x2_img, y2_img],
            }
        )
    return out


def _char_positions_from_pdfium_chars(
    chars: List[Dict[str, Any]],
    span_img_box: Sequence[float],
) -> List[Dict[str, Any]]:
    """Select PDFium chars whose boxes are largely covered by span bbox."""
    span_box = np.asarray(span_img_box, dtype=np.float64)
    selected: List[Tuple[float, float, Dict[str, Any]]] = []
    for item in chars:
        ch_box = np.asarray(item["img_bbox"], dtype=np.float64)
        ch_area = float(calculate_bbox_area(ch_box))
        if ch_area <= 1e-6:
            continue
        inter = get_bbox_intersection(ch_box, span_box, return_format="bbox")
        if inter is None:
            continue
        cov = float(calculate_bbox_area(inter)) / ch_area
        if cov < 0.5:
            continue
        y_center = (ch_box[1] + ch_box[3]) * 0.5
        x_left = ch_box[0]
        selected.append((y_center, x_left, item))

    selected.sort(key=lambda x: (x[0], x[1]))
    out: List[Dict[str, Any]] = []
    for pos, (_, _, item) in enumerate(selected):
        out.append(
            {
                "char": item["char"],
                "bbox": [round(v, 4) for v in item["pdf_bbox"]],
                "char_idx": int(item["char_idx"]),
                "position_in_span": pos,
            }
        )
    return out


def _bbox_union(boxes: List[Sequence[float]]) -> List[float]:
    """Return enclosing bbox for a list of boxes."""
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    arr = np.asarray(boxes, dtype=np.float64)
    return [
        float(np.min(arr[:, 0])),
        float(np.min(arr[:, 1])),
        float(np.max(arr[:, 2])),
        float(np.max(arr[:, 3])),
    ]


def _group_spans_to_lines_in_column(spans: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group spans into lines inside one x-column."""
    spans_sorted = sorted(
        spans,
        key=lambda s: (
            (float(s["span_bbox_img"][1]) + float(s["span_bbox_img"][3])) * 0.5,
            float(s["span_bbox_img"][0]),
        ),
    )
    lines: List[List[Dict[str, Any]]] = []
    line_bboxes: List[List[float]] = []
    for span in spans_sorted:
        sb = np.asarray(span["span_bbox_img"], dtype=np.float64)
        sh = max(1.0, float(sb[3] - sb[1]))
        scy = (float(sb[1]) + float(sb[3])) * 0.5
        matched_idx = None
        best_dist = float("inf")
        for li, lb in enumerate(line_bboxes):
            lcy = (float(lb[1]) + float(lb[3])) * 0.5
            lh = max(1.0, float(lb[3] - lb[1]))
            inter_h = max(0.0, min(float(sb[3]), float(lb[3])) - max(float(sb[1]), float(lb[1])))
            overlap_small = inter_h / min(sh, lh)
            y_dist = abs(scy - lcy)
            if overlap_small >= 0.45 or y_dist <= max(2.0, 0.35 * min(sh, lh)):
                if y_dist < best_dist:
                    best_dist = y_dist
                    matched_idx = li
        if matched_idx is None:
            lines.append([span])
            line_bboxes.append([float(sb[0]), float(sb[1]), float(sb[2]), float(sb[3])])
        else:
            lines[matched_idx].append(span)
            lb = line_bboxes[matched_idx]
            line_bboxes[matched_idx] = [
                min(lb[0], float(sb[0])),
                min(lb[1], float(sb[1])),
                max(lb[2], float(sb[2])),
                max(lb[3], float(sb[3])),
            ]
    lines = sorted(
        lines,
        key=lambda line: (
            min((float(s["span_bbox_img"][1]) + float(s["span_bbox_img"][3])) * 0.5 for s in line),
            min(float(s["span_bbox_img"][0]) for s in line),
        ),
    )
    for line in lines:
        line.sort(key=lambda s: float(s["span_bbox_img"][0]))
    return lines


def _split_spans_into_columns(spans: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Split spans into x-columns first to reduce cross-column line merge."""
    if not spans:
        return []
    spans_sorted = sorted(spans, key=lambda s: float(s["span_bbox_img"][0]))
    widths = [
        max(1.0, float(s["span_bbox_img"][2]) - float(s["span_bbox_img"][0]))
        for s in spans_sorted
    ]
    median_w = float(np.median(np.asarray(widths, dtype=np.float64))) if widths else 8.0
    col_gap_threshold = max(20.0, 4.0 * median_w)

    columns: List[List[Dict[str, Any]]] = []
    col_ranges: List[List[float]] = []
    for sp in spans_sorted:
        x1, _, x2, _ = [float(v) for v in sp["span_bbox_img"]]
        assigned = False
        for ci, rg in enumerate(col_ranges):
            inter = max(0.0, min(x2, rg[1]) - max(x1, rg[0]))
            min_w = max(1.0, min(x2 - x1, rg[1] - rg[0]))
            close = (x1 <= rg[1] + col_gap_threshold) and (x2 >= rg[0] - col_gap_threshold)
            if inter / min_w >= 0.1 or close:
                columns[ci].append(sp)
                col_ranges[ci] = [min(rg[0], x1), max(rg[1], x2)]
                assigned = True
                break
        if not assigned:
            columns.append([sp])
            col_ranges.append([x1, x2])

    cols_sorted = sorted(
        zip(columns, col_ranges),
        key=lambda pair: pair[1][0],
    )
    return [c for c, _ in cols_sorted]


def _group_spans_to_lines(spans: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Two-stage line grouping: split columns by x, then cluster lines by y in each column."""
    if not spans:
        return []
    columns = _split_spans_into_columns(spans)
    lines_all: List[List[Dict[str, Any]]] = []
    for col in columns:
        lines_all.extend(_group_spans_to_lines_in_column(col))
    return lines_all


def _merge_line_spans_by_runs(line_spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge adjacent spans in one line to avoid per-character spans.
    Since PDFium style fields are not exposed here, we merge by geometry continuity.
    """
    if not line_spans:
        return []
    if len(line_spans) == 1:
        return line_spans

    spans = sorted(line_spans, key=lambda s: float(s["span_bbox_img"][0]))
    widths = [max(1.0, float(s["span_bbox_img"][2]) - float(s["span_bbox_img"][0])) for s in spans]
    median_w = float(np.median(np.asarray(widths, dtype=np.float64))) if widths else 8.0
    merge_gap = max(2.0, 0.8 * median_w)

    merged: List[Dict[str, Any]] = []
    cur = dict(spans[0])
    cur["char_positions"] = list(cur.get("char_positions", []))

    for nxt in spans[1:]:
        cur_box = cur["span_bbox_img"]
        nxt_box = nxt["span_bbox_img"]
        gap = float(nxt_box[0]) - float(cur_box[2])
        same_tag = (
            cur.get("type") == nxt.get("type")
            and cur.get("char_source") == nxt.get("char_source")
        )
        if same_tag and gap <= merge_gap:
            cur["content"] = f"{cur.get('content', '')}{nxt.get('content', '')}"
            cur["span_bbox_img"] = _bbox_union([cur_box, nxt_box])
            cur["span_bbox_pdf"] = [
                round(v, 2)
                for v in _bbox_union([cur.get("span_bbox_pdf", cur_box), nxt.get("span_bbox_pdf", nxt_box)])
            ]
            cur["score"] = float(max(float(cur.get("score", 0.0)), float(nxt.get("score", 0.0))))
            cur["char_positions"] = list(cur.get("char_positions", [])) + list(
                nxt.get("char_positions", [])
            )
            for pos, ch in enumerate(cur["char_positions"]):
                ch["position_in_span"] = pos
        else:
            merged.append(cur)
            cur = dict(nxt)
            cur["char_positions"] = list(cur.get("char_positions", []))
    merged.append(cur)
    return merged


def layout_parsing_result_to_middle_page(
    result: Any,
    *,
    min_ocr_coverage: float = 0.7,
    char_source: str = "pdf_text",
) -> Dict[str, Any]:
    """
    Convert one page ``LayoutParsingResultV2`` (mapping) into one ``middle_pages`` JSON object.

    Text/title paragraphs: PDFium (or OCR) boxes are matched to PP-Structure blocks with
    ``inter_area / ocr_box_area >= min_ocr_coverage`` (exclusive: each OCR box to one block).

    Args:
        result: ``LayoutParsingResultV2`` instance or dict with keys
            ``input_path``, ``page_index``, ``width``, ``height``, ``parsing_res_list``, ``overall_ocr_res``.
        min_ocr_coverage: Minimum fraction of OCR rectangle area that must lie inside the block.
        char_source: Value stored in each text span (e.g. ``pdf_text`` when using text layer).

    Returns:
        Dict matching ``{"pdf_info": [{"page_idx", "page_size", "para_blocks": [...]}]}``.
    """
    input_path = result["input_path"]
    page_idx = int(result["page_index"])
    img_w = int(result["width"])
    img_h = int(result["height"])
    pdf_size = read_pdf_page_size_pts(str(input_path), page_idx)
    if pdf_size is None:
        pdf_w, pdf_h = float(img_w), float(img_h)
    else:
        pdf_w, pdf_h = pdf_size
    pdfium_chars = _read_pdfium_page_chars(
        str(input_path), page_idx, img_w, img_h, pdf_w, pdf_h
    )

    parsing_res_list: List[LayoutBlock] = result["parsing_res_list"]
    overall_ocr = result["overall_ocr_res"]
    rec_boxes = overall_ocr["rec_boxes"]
    rec_texts = overall_ocr["rec_texts"]
    rec_scores = overall_ocr["rec_scores"]

    text_blocks: List[Tuple[int, np.ndarray]] = []
    for bi, block in enumerate(parsing_res_list):
        mtype = _layout_label_to_middle_para_type(block.label)
        if mtype in ("text", "title"):
            text_blocks.append((bi, np.asarray(block.bbox, dtype=np.float64)))

    assign = assign_ocr_indices_exclusive_by_coverage(
        rec_boxes, text_blocks, min_ocr_coverage
    )

    para_blocks: List[Dict[str, Any]] = []
    global_char_idx = 0

    for bi, block in enumerate(parsing_res_list):
        label = block.label
        mtype = _layout_label_to_middle_para_type(label)
        bbox_img = np.asarray(block.bbox, dtype=np.float64)
        pdf_bbox = img_axis_aligned_box_to_pdf_top_left(bbox_img, img_w, img_h, pdf_w, pdf_h)

        if mtype == "table":
            body_bbox = [round(v, 2) for v in pdf_bbox]
            para_blocks.append(
                {
                    "type": "table",
                    "bbox": body_bbox,
                    "blocks": [
                        {
                            "type": "table_body",
                            "bbox": body_bbox,
                            "group_id": 0,
                            "lines": [
                                {
                                    "bbox": body_bbox,
                                    "spans": [
                                        {
                                            "bbox": body_bbox,
                                            "score": 1.0,
                                            "type": "table",
                                            "content": block.content,
                                        }
                                    ],
                                }
                            ],
                            "index": bi,
                        }
                    ],
                    "index": bi,
                }
            )
            continue

        if mtype == "interline_equation":
            span_pdf = [float(x) for x in pdf_bbox]
            txt = block.content or ""
            char_positions, global_char_idx = _approx_char_bboxes(
                txt, span_pdf, global_char_idx
            )
            para_blocks.append(
                {
                    "type": "interline_equation",
                    "bbox": [round(x, 2) for x in pdf_bbox],
                    "lines": [
                        {
                            "bbox": [round(x, 2) for x in span_pdf],
                            "spans": [
                                {
                                    "bbox": [round(x, 2) for x in span_pdf],
                                    "score": 1.0,
                                    "content": txt,
                                    "type": "interline_equation",
                                    "char_positions": char_positions,
                                    "char_source": "formula",
                                }
                            ],
                            "index": 0,
                        }
                    ],
                    "index": bi,
                }
            )
            continue

        if mtype == "image":
            img_path = ""
            if getattr(block, "image", None) and isinstance(block.image, dict):
                img_path = str(block.image.get("path", "") or "")
            body_bbox = [round(v, 2) for v in pdf_bbox]
            para_blocks.append(
                {
                    "type": "image",
                    "bbox": body_bbox,
                    "blocks": [
                        {
                            "type": "image_body",
                            "bbox": body_bbox,
                            "group_id": 0,
                            "lines": [
                                {
                                    "bbox": body_bbox,
                                    "spans": [
                                        {
                                            "bbox": body_bbox,
                                            "score": 1.0,
                                            "type": "image",
                                            "image_path": img_path,
                                        }
                                    ],
                                }
                            ],
                            "index": bi,
                        }
                    ],
                    "index": bi,
                }
            )
            continue

        ocr_idxes = assign.get(bi, [])
        lines_out: List[Dict[str, Any]] = []
        span_candidates: List[Dict[str, Any]] = []
        for ocr_i in ocr_idxes:
            txt = str(rec_texts[ocr_i])
            if not txt:
                continue
            sc = float(rec_scores[ocr_i]) if ocr_i < len(rec_scores) else 1.0
            ocr_img = rec_boxes[ocr_i]
            span_pdf = img_axis_aligned_box_to_pdf_top_left(
                ocr_img, img_w, img_h, pdf_w, pdf_h
            )
            char_positions = _char_positions_from_pdfium_chars(pdfium_chars, ocr_img)
            used_char_source = char_source
            if not char_positions:
                char_positions, global_char_idx = _approx_char_bboxes(
                    txt, span_pdf, global_char_idx
                )
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
        grouped = _group_spans_to_lines(span_candidates)
        for line_i, line_spans in enumerate(grouped):
            line_spans = _merge_line_spans_by_runs(line_spans)
            line_img_box = _bbox_union([s["span_bbox_img"] for s in line_spans])
            line_pdf = img_axis_aligned_box_to_pdf_top_left(
                line_img_box, img_w, img_h, pdf_w, pdf_h
            )
            lines_out.append(
                {
                    "bbox": [round(x, 2) for x in line_pdf],
                    "spans": [
                        {
                            "bbox": s["span_bbox_pdf"],
                            "score": s["score"],
                            "content": s["content"],
                            "type": s["type"],
                            "char_positions": s["char_positions"],
                            "char_source": s["char_source"],
                        }
                        for s in line_spans
                    ],
                    "index": line_i,
                }
            )

        if not lines_out and (block.content or "").strip():
            span_pdf = pdf_bbox
            txt = block.content
            char_positions, global_char_idx = _approx_char_bboxes(
                txt, span_pdf, global_char_idx
            )
            lines_out.append(
                {
                    "bbox": [round(x, 2) for x in span_pdf],
                    "spans": [
                        {
                            "bbox": [round(x, 2) for x in span_pdf],
                            "score": 1.0,
                            "content": txt,
                            "type": "text",
                            "char_positions": char_positions,
                            "char_source": "block_fallback",
                        }
                    ],
                    "index": 0,
                }
            )

        para_blocks.append(
            {
                "type": mtype,
                "bbox": [round(x, 2) for x in pdf_bbox],
                "lines": lines_out,
                "index": bi,
            }
        )
    # Remove empty text/title para blocks (no lines), keep vision/table/formula blocks.
    para_blocks = [
        p
        for p in para_blocks
        if p.get("type") not in ("text", "title") or len(p.get("lines", [])) > 0
    ]

    page_entry = {
        "page_idx": page_idx,
        "page_size": [int(round(pdf_w)), int(round(pdf_h))],
        "para_blocks": para_blocks,
    }
    return {"pdf_info": [page_entry]}


def write_middle_index_json(
    output_dir: Union[str, Path],
    total_pages: int,
    *,
    pages_subdir: str = "middle_pages",
) -> Path:
    """Write ``middle_index.json`` listing ``page_XXXX.json`` paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = {
        "output_directory": pages_subdir,
        "total_pages": int(total_pages),
        "pages": [
            {
                "page_idx": i,
                "path": f"{pages_subdir}/page_{i:04d}.json",
                "file_name": f"page_{i:04d}.json",
            }
            for i in range(int(total_pages))
        ],
    }
    p = out / "middle_index.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
    return p


def save_middle_page_json(
    page_dict: Dict[str, Any],
    output_dir: Union[str, Path],
    *,
    pages_subdir: str = "middle_pages",
) -> Path:
    """Write one page JSON under ``output_dir / pages_subdir``."""
    out = Path(output_dir) / pages_subdir
    out.mkdir(parents=True, exist_ok=True)
    page_idx = int(page_dict["pdf_info"][0]["page_idx"])
    path = out / f"page_{page_idx:04d}.json"
    path.write_text(json.dumps(page_dict, ensure_ascii=False, indent=4), encoding="utf-8")
    return path


def save_middle_page_visualization(
    page_dict: Dict[str, Any],
    page_result: Any,
    output_dir: Union[str, Path],
    *,
    vis_subdir: str = "middle_vis",
) -> Optional[Path]:
    """
    Save line/span visualization for one middle page.
    Green = line bbox, Blue = span bbox.
    """
    if "doc_preprocessor_res" not in page_result:
        return None
    img = page_result["doc_preprocessor_res"].get("output_img", None)
    if img is None:
        return None
    vis = np.array(img).copy()
    if vis.ndim != 3 or vis.shape[2] != 3:
        return None

    page = page_dict["pdf_info"][0]
    page_size = page["page_size"]
    pdf_w, pdf_h = float(page_size[0]), float(page_size[1])
    img_h, img_w = vis.shape[0], vis.shape[1]

    for para in page.get("para_blocks", []):
        for line in para.get("lines", []):
            lb = pdf_top_left_box_to_img(line["bbox"], img_w, img_h, pdf_w, pdf_h)
            cv2.rectangle(vis, (lb[0], lb[1]), (lb[2], lb[3]), (0, 180, 0), 2)
            for span in line.get("spans", []):
                sb = pdf_top_left_box_to_img(
                    span["bbox"], img_w, img_h, pdf_w, pdf_h
                )
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
    """
    Export a full document: ``middle_index.json`` + ``middle_pages/page_XXXX.json`` for each page result.

    ``results`` should be ordered by ``page_index`` (0 .. N-1). ``page_count`` on the first item
    sets ``total_pages`` when present; otherwise ``len(results)``.
    """
    if not results:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    page_dir = out / pages_subdir
    page_dir.mkdir(parents=True, exist_ok=True)

    first = results[0]
    total = first["page_count"] if "page_count" in first else None
    if total is None:
        total = len(results)
    total = int(total)

    write_middle_index_json(out, total, pages_subdir=pages_subdir)

    for item in results:
        page_dict = layout_parsing_result_to_middle_page(
            item, min_ocr_coverage=min_ocr_coverage, char_source=char_source
        )
        save_middle_page_json(page_dict, out, pages_subdir=pages_subdir)
        if save_vis:
            save_middle_page_visualization(
                page_dict,
                item,
                out,
                vis_subdir=vis_subdir,
            )
