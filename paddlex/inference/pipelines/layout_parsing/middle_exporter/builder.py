from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import unicodedata

from ..layout_objects import LayoutBlock
from ..setting import BLOCK_LABEL_MAP
from .geometry import bbox_union, img_axis_aligned_box_to_pdf_top_left, read_pdf_page_size_pts
from .grouping import assign_ocr_indices_exclusive_by_coverage, group_spans_to_lines
from .pdfium_chars import (
    approx_char_bboxes,
    char_positions_from_pdfium_chars,
    read_pdfium_page_chars,
    sanitize_middle_text,
)


def _convert_bbox_top_left_to_bottom_left(bbox: Sequence[float], page_h: float) -> List[float]:
    x1, y1, x2, y2 = map(float, bbox)
    by1 = float(page_h) - y2
    by2 = float(page_h) - y1
    if by2 < by1:
        by1, by2 = by2, by1
    return [x1, by1, x2, by2]


def _flip_export_bboxes_to_bottom_left(obj: Any, page_h: float) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "bbox" and isinstance(value, list) and len(value) == 4:
                obj[key] = [round(v, 4) for v in _convert_bbox_top_left_to_bottom_left(value, page_h)]
            else:
                _flip_export_bboxes_to_bottom_left(value, page_h)
        return
    if isinstance(obj, list):
        for item in obj:
            _flip_export_bboxes_to_bottom_left(item, page_h)


def _is_discarded_block_label(label: str) -> bool:
    low = (label or "").lower()
    return (
        (low in BLOCK_LABEL_MAP["header_labels"])
        or (low in BLOCK_LABEL_MAP["footer_labels"])
        or (low in ("number", "formula_number"))
    )


def _bbox_overlap_x(a: np.ndarray, b: np.ndarray) -> float:
    inter = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    denom = max(1.0, min(float(a[2] - a[0]), float(b[2] - b[0])))
    return inter / denom


def _bbox_intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = _bbox_intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    a_area = max(1.0, (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])))
    b_area = max(1.0, (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])))
    return float(inter) / float(a_area + b_area - inter)


def _bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(float, bbox)
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _bbox_edge_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Minimum L2 distance between two axis-aligned rectangles (0 if overlap).
    Coordinates are in the same space.
    """
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    dx = 0.0
    if ax2 < bx1:
        dx = bx1 - ax2
    elif bx2 < ax1:
        dx = ax1 - bx2
    dy = 0.0
    if ay2 < by1:
        dy = by1 - ay2
    elif by2 < ay1:
        dy = ay1 - by2
    return float((dx * dx + dy * dy) ** 0.5)


def _effective_index_diff(child_bi: int, parent_bi: int, between_bis: Sequence[int]) -> int:
    """
    "有效差值": abs(child-parent) - (# of same-kind children between).
    between_bis should only include indices of the same child category within (min,max).
    """
    raw = abs(int(child_bi) - int(parent_bi))
    return int(max(0, raw - len(list(between_bis))))


def tie_up_category_by_index(
    *,
    children: List[Dict[str, Any]],
    parents: List[Dict[str, Any]],
    child_kind: str,
    edge_tie_thr: float = 2.0,
) -> Dict[int, int]:
    """
    Match child blocks to parent blocks following MinerU-like priority:
    1) effective index diff
    2) bbox edge distance
    3) bbox center distance

    Special tie-break for table caption/footnote:
    - caption: prefer larger parent index (later table) when index diff equal and edge dist similar
    - footnote: prefer smaller parent index (earlier table) under the same condition

    Return mapping: child_bi -> parent_bi (both are parsing_res_list indices).
    """
    if not children or not parents:
        return {}
    parent_by_bi = {int(p["bi"]): p for p in parents if "bi" in p}
    out: Dict[int, int] = {}
    # Precompute same-kind children indices for effective diff deduction.
    child_bis = sorted({int(c["bi"]) for c in children if "bi" in c})
    for c in children:
        c_bi = int(c["bi"])
        c_bbox = c["bbox_img"]
        best = None
        best_tuple = None
        for p_bi, p in parent_by_bi.items():
            p_bbox = p["bbox_img"]
            lo, hi = (c_bi, p_bi) if c_bi < p_bi else (p_bi, c_bi)
            between = [bi for bi in child_bis if lo < bi < hi]
            eff = _effective_index_diff(c_bi, p_bi, between)
            edge = _bbox_edge_distance(c_bbox, p_bbox)
            cx, cy = _bbox_center(c_bbox)
            px, py = _bbox_center(p_bbox)
            center = float(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5)
            tup = (eff, edge, center)
            if best_tuple is None or tup < best_tuple:
                best_tuple = tup
                best = p_bi
            elif best_tuple is not None and tup[0] == best_tuple[0] and abs(tup[1] - best_tuple[1]) <= edge_tie_thr:
                # table_caption/table_footnote special tie-break
                if child_kind == "table_caption":
                    if int(p_bi) > int(best):
                        best = p_bi
                        best_tuple = tup
                elif child_kind == "table_footnote":
                    if int(p_bi) < int(best):
                        best = p_bi
                        best_tuple = tup
        if best is not None:
            out[c_bi] = int(best)
    return out


def fix_two_layer_blocks(
    *,
    body_bi: int,
    caption_bis: List[int],
    footnote_bis: List[int],
) -> Tuple[List[int], List[int]]:
    """
    MinerU-like two-layer continuity filtering:
    - caption: start from closest to body, walk backwards (descending bi), keep continuous chain;
      once a real gap occurs, drop earlier (farther) captions.
    - footnote: start from closest to body, walk forwards (ascending bi), keep continuous chain;
      once a real gap occurs, drop later (farther) footnotes.
    """
    caps = sorted({int(x) for x in caption_bis})
    foots = sorted({int(x) for x in footnote_bis})
    kept_caps: List[int] = []
    if caps:
        # closest caption is the one with max bi < body
        i = len(caps) - 1
        prev = caps[i]
        kept_caps.append(prev)
        i -= 1
        while i >= 0:
            cur = caps[i]
            if cur == prev - 1:
                kept_caps.append(cur)
                prev = cur
                i -= 1
                continue
            break
    kept_caps = sorted(kept_caps)

    kept_foots: List[int] = []
    if foots:
        # closest footnote is the one with min bi > body
        i = 0
        prev = foots[i]
        kept_foots.append(prev)
        i += 1
        while i < len(foots):
            cur = foots[i]
            if cur == prev + 1:
                kept_foots.append(cur)
                prev = cur
                i += 1
                continue
            break
    return kept_caps, kept_foots


def _table_res_item_bbox_img(item: Dict[str, Any]) -> List[float] | None:
    """
    Table recognition results may optionally carry a table-level bbox.
    We accept a few common key variants and normalize to [x1,y1,x2,y2] in image coords.
    """
    for key in ("bbox", "table_bbox", "table_box", "table_bboxes"):
        raw = item.get(key)
        if raw is None:
            continue
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            try:
                return [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]
            except Exception:
                return None
    return None


def _build_table_cells(
    table_res_item: Dict[str, Any],
    table_html: str,
    pdfium_chars: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    pdf_w: float,
    pdf_h: float,
    global_char_idx: int,
    effective_char_source: str,
) -> Tuple[List[Dict[str, Any]], int]:
    cells_out: List[Dict[str, Any]] = []
    cell_boxes_raw = table_res_item.get("cell_box_list", [])
    if cell_boxes_raw is None:
        cell_boxes = []
    elif isinstance(cell_boxes_raw, np.ndarray):
        cell_boxes = cell_boxes_raw.tolist()
    else:
        cell_boxes = list(cell_boxes_raw)
    ocr_pred = table_res_item.get("table_ocr_pred") or {}
    rec_boxes_raw = ocr_pred.get("rec_boxes", [])
    if rec_boxes_raw is None:
        rec_boxes = []
    elif isinstance(rec_boxes_raw, np.ndarray):
        rec_boxes = rec_boxes_raw.tolist()
    else:
        rec_boxes = list(rec_boxes_raw)
    rec_texts_raw = ocr_pred.get("rec_texts", [])
    rec_texts = [] if rec_texts_raw is None else list(rec_texts_raw)
    html_cell_texts = _extract_cell_texts_from_html(table_html)
    for cell in cell_boxes:
        cell_box = [float(v) for v in cell]
        cell_np = np.asarray(cell_box, dtype=np.float64)
        matched_indices: List[int] = []
        for i, rb in enumerate(rec_boxes):
            rb_box = [float(v) for v in rb]
            rb_area = max(1.0, (rb_box[2] - rb_box[0]) * (rb_box[3] - rb_box[1]))
            if _bbox_intersection_area(rb_box, cell_box) / rb_area >= 0.3:
                matched_indices.append(i)
        chars: List[Dict[str, Any]] = []
        for i in matched_indices:
            txt = sanitize_middle_text(str(rec_texts[i])) if i < len(rec_texts) else ""
            rb = np.asarray(rec_boxes[i], dtype=np.float64)
            span_pdf = img_axis_aligned_box_to_pdf_top_left(rb, img_w, img_h, pdf_w, pdf_h)
            cpos, _span_source, global_char_idx = _pick_char_positions_for_span(
                text=txt,
                ocr_img_box=rb,
                span_pdf_box=span_pdf,
                pdfium_chars=pdfium_chars,
                preferred_source=effective_char_source,
                global_char_idx=global_char_idx,
            )
            chars.extend(cpos)
        for pos, ch in enumerate(chars):
            ch["position_in_span"] = pos
        cell_idx = len(cells_out)
        html_text = html_cell_texts[cell_idx] if cell_idx < len(html_cell_texts) else ""
        content = sanitize_middle_text(html_text)
        cells_out.append({"content": content, "char_positions": chars})
    return cells_out, global_char_idx


def _extract_cell_texts_from_html(table_html: str) -> List[str]:
    if not table_html:
        return []
    cells = re.findall(
        r"<(?:td|th)\b[^>]*>(.*?)</(?:td|th)>", table_html, flags=re.IGNORECASE | re.DOTALL
    )
    out: List[str] = []
    for cell in cells:
        no_tags = re.sub(r"<[^>]+>", "", cell)
        txt = html.unescape(no_tags).replace("\xa0", " ").strip()
        out.append(txt)
    return out


def _build_table_virtual_lines(
    table_res_item: Dict[str, Any],
    img_w: int,
    img_h: int,
    pdf_w: float,
    pdf_h: float,
    fallback_bbox_pdf: Sequence[float],
) -> List[Dict[str, Any]]:
    cell_boxes_raw = table_res_item.get("cell_box_list", [])
    if cell_boxes_raw is None:
        cell_boxes = []
    elif isinstance(cell_boxes_raw, np.ndarray):
        cell_boxes = cell_boxes_raw.tolist()
    else:
        cell_boxes = list(cell_boxes_raw)
    if not cell_boxes:
        return [{"bbox": [round(v, 2) for v in fallback_bbox_pdf], "spans": [], "index": 0}]
    boxes = [np.asarray([float(v) for v in b], dtype=np.float64) for b in cell_boxes]
    heights = [max(1.0, float(b[3] - b[1])) for b in boxes]
    median_h = float(np.median(np.asarray(heights, dtype=np.float64))) if heights else 12.0
    y_merge_thr = max(3.0, 0.5 * median_h)
    boxes_sorted = sorted(boxes, key=lambda b: ((float(b[1]) + float(b[3])) * 0.5, float(b[0])))
    row_groups: List[List[np.ndarray]] = []
    row_centers: List[float] = []
    for b in boxes_sorted:
        cy = (float(b[1]) + float(b[3])) * 0.5
        matched = None
        best_dist = float("inf")
        for i, rcy in enumerate(row_centers):
            d = abs(cy - rcy)
            if d <= y_merge_thr and d < best_dist:
                matched = i
                best_dist = d
        if matched is None:
            row_groups.append([b])
            row_centers.append(cy)
        else:
            row_groups[matched].append(b)
            vals = [((float(x[1]) + float(x[3])) * 0.5) for x in row_groups[matched]]
            row_centers[matched] = float(np.mean(np.asarray(vals, dtype=np.float64)))
    virtual_lines: List[Dict[str, Any]] = []
    for i, group in enumerate(row_groups):
        row_img = bbox_union([g.tolist() for g in group])
        row_pdf = img_axis_aligned_box_to_pdf_top_left(row_img, img_w, img_h, pdf_w, pdf_h)
        virtual_lines.append({"bbox": [round(v, 2) for v in row_pdf], "spans": [], "index": i})
    virtual_lines.sort(key=lambda r: (float(r["bbox"][1]), float(r["bbox"][0])))
    for i, row in enumerate(virtual_lines):
        row["index"] = i
    return virtual_lines


def _is_pure_punctuation(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return all(unicodedata.category(ch).startswith("P") for ch in t)


def layout_label_to_middle_para_type(label: str) -> str:
    low = label.lower()
    if low in BLOCK_LABEL_MAP["doc_title_labels"]:
        return "title"
    if low in BLOCK_LABEL_MAP["paragraph_title_labels"]:
        return "title"
    if low in BLOCK_LABEL_MAP["vision_title_labels"]:
        return "title"
    if low == "content":
        return "content"
    if low in ("number", "formula_number"):
        return low
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
    heights = [max(1.0, float(s["span_bbox_img"][3]) - float(s["span_bbox_img"][1])) for s in spans]
    median_w = float(np.median(np.asarray(widths, dtype=np.float64))) if widths else 8.0
    median_h = float(np.median(np.asarray(heights, dtype=np.float64))) if heights else 16.0
    # Use a coarser merge policy to reduce over-fragmented text spans.
    merge_gap = max(4.0, 1.8 * median_w, 0.35 * median_h)
    merged: List[Dict[str, Any]] = []
    cur = dict(spans[0])
    cur["char_positions"] = list(cur.get("char_positions", []))
    for nxt in spans[1:]:
        cur_box, nxt_box = cur["span_bbox_img"], nxt["span_bbox_img"]
        gap = float(nxt_box[0]) - float(cur_box[2])
        same_type = cur.get("type") == nxt.get("type")
        same_source = cur.get("char_source") == nxt.get("char_source")
        punct_attach = _is_pure_punctuation(str(cur.get("content", ""))) or _is_pure_punctuation(
            str(nxt.get("content", ""))
        )
        if same_type and gap <= merge_gap and (same_source or punct_attach):
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


def _build_span_candidate(
    *,
    span_bbox_img: Sequence[float],
    span_bbox_pdf: Sequence[float],
    score: float,
    content: str,
    span_type: str,
    char_positions: List[Dict[str, Any]],
    char_source: str,
) -> Dict[str, Any]:
    # Inline equations are represented by LaTeX content; do not attach char_positions
    # (PDF char boxes are often incomplete/misaligned for formula glyphs).
    if str(span_type) == "inline_equation":
        char_positions = []
    return {
        "span_bbox_img": [float(v) for v in span_bbox_img],
        "span_bbox_pdf": [round(float(x), 2) for x in span_bbox_pdf],
        "score": float(score),
        "content": str(content),
        "type": str(span_type),
        "char_positions": char_positions,
        "char_source": str(char_source),
    }


def _to_middle_span(span: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "bbox": span["span_bbox_pdf"],
        "score": span["score"],
        "content": span["content"],
        "type": span["type"],
        "char_source": span["char_source"],
    }
    if span.get("type") != "inline_equation":
        out["char_positions"] = span["char_positions"]
    return out


def _contains_math_alphanumeric_symbols(text: str) -> bool:
    # Mathematical Alphanumeric Symbols block (italic math letters/digits etc.)
    for ch in str(text or ""):
        cp = ord(ch)
        if 0x1D400 <= cp <= 0x1D7FF:
            return True
    return False


def _is_unreliable_char(ch: str) -> bool:
    if not ch:
        return True
    if ch == "\ufffd":
        return True
    cat = unicodedata.category(ch)
    if cat == "Cs":
        return True
    if cat.startswith("C") and ch not in ("\t", "\n", "\r"):
        return True
    return False


def _pick_char_positions_for_span(
    *,
    text: str,
    ocr_img_box: Sequence[float],
    span_pdf_box: Sequence[float],
    pdfium_chars: List[Dict[str, Any]],
    preferred_source: str,
    global_char_idx: int,
) -> Tuple[List[Dict[str, Any]], str, int]:
    """
    MinerU-like region routing:
    - Prefer pdf_text only when extracted chars are sufficiently reliable for this span.
    - Fallback to OCR-derived approximate char boxes for this span only.
    """
    clean_text = sanitize_middle_text(text)
    if not clean_text:
        return [], preferred_source, int(global_char_idx)
    if preferred_source != "pdf_text":
        approx, nxt_idx = approx_char_bboxes(clean_text, span_pdf_box, int(global_char_idx))
        return approx, "ocr_text", int(nxt_idx)

    pdf_chars = char_positions_from_pdfium_chars(pdfium_chars, ocr_img_box)
    if not pdf_chars:
        approx, nxt_idx = approx_char_bboxes(clean_text, span_pdf_box, int(global_char_idx))
        return approx, "ocr_text", int(nxt_idx)

    bad = 0
    for it in pdf_chars:
        if _is_unreliable_char(str(it.get("char", ""))):
            bad += 1
    bad_ratio = float(bad) / max(1, len(pdf_chars))
    coverage_ratio = float(len(pdf_chars)) / max(1, len(clean_text))

    # Conservative span-level quality gate to avoid switching good spans.
    if bad_ratio >= 0.05 or coverage_ratio < 0.35:
        approx, nxt_idx = approx_char_bboxes(clean_text, span_pdf_box, int(global_char_idx))
        return approx, "ocr_text", int(nxt_idx)
    return pdf_chars, "pdf_text", int(global_char_idx)


def _resolve_span_content(
    default_text: str, char_positions: List[Dict[str, Any]], char_source: str
) -> str:
    """
    Prefer PDF-char reconstructed text when available.
    This keeps punctuation that OCR text may miss.
    Falls back to OCR text when PDF-char reconstruction is shorter (chars dropped due to
    missing ToUnicode mapping in the PDF font).
    """
    if str(char_source) != "pdf_text" or not char_positions:
        return default_text
    rebuilt = sanitize_middle_text(
        "".join(str(ch.get("char", "") or "") for ch in char_positions)
    )
    if not rebuilt:
        return default_text
    # If PDF-char text is shorter than OCR text, the font encoding likely dropped some
    # characters (e.g. punctuation with missing ToUnicode mapping). Prefer OCR in that case.
    if len(rebuilt) < len(default_text):
        return default_text
    return rebuilt


def _build_text_caption_block(
    *,
    caption_type: str,
    bbox: Sequence[float],
    content: str,
    char_positions: List[Dict[str, Any]],
    line_index: int,
    char_source: str,
) -> Dict[str, Any]:
    cap_bbox = [round(float(v), 2) for v in bbox]
    return {
        "type": caption_type,
        "bbox": cap_bbox,
        "group_id": 0,
        "lines": [
            {
                "bbox": cap_bbox,
                "spans": [
                    {
                        "type": "text",
                        "bbox": cap_bbox,
                        "score": 1.0,
                        "content": content,
                        "char_source": str(char_source),
                        "char_positions": char_positions,
                    }
                ],
                "index": int(line_index),
            }
        ],
        "index": int(line_index),
    }


def _build_split_virtual_lines(
    body_bbox: Sequence[float], *, split_count: int, precision: int = 4
) -> List[Dict[str, Any]]:
    x1, y1, x2, y2 = [float(v) for v in body_bbox]
    h = max(1e-6, y2 - y1)
    step = h / max(1, int(split_count))
    out: List[Dict[str, Any]] = []
    for i in range(int(split_count)):
        lo = y1 + step * i
        hi = y1 + step * (i + 1)
        out.append(
            {
                "bbox": [round(x1, precision), round(lo, precision), round(x2, precision), round(hi, precision)],
                "spans": [],
                "index": -1,
            }
        )
    return out


def _collect_inline_formula_blocks(
    parsing_res_list: List[LayoutBlock], img_w: int, img_h: int, pdf_w: float, pdf_h: float
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for bi, block in enumerate(parsing_res_list):
        if str(getattr(block, "label", "") or "").lower() != "inline_formula":
            continue
        bbox_img = np.asarray(block.bbox, dtype=np.float64)
        pdf_bbox = img_axis_aligned_box_to_pdf_top_left(bbox_img, img_w, img_h, pdf_w, pdf_h)
        blocks.append(
            {
                "bbox_img": bbox_img,
                "bbox_pdf": [round(x, 2) for x in pdf_bbox],
                "content": str(getattr(block, "content", "") or ""),
                "image": getattr(block, "image", None),
                "index": bi,
            }
        )
    return blocks


def _inject_inline_formula_spans(
    *,
    span_candidates: List[Dict[str, Any]],
    mtype: str,
    bbox_img: np.ndarray,
    inline_formula_blocks: List[Dict[str, Any]],
    consumed_inline_formula_indices: set,
    pdfium_chars: List[Dict[str, Any]],
) -> None:
    if mtype not in ("text", "title") or not inline_formula_blocks:
        return
    block_box_img = bbox_img
    for inf in inline_formula_blocks:
        if inf["index"] in consumed_inline_formula_indices:
            continue
        inf_box_img = inf["bbox_img"]
        inter = _bbox_intersection_area(block_box_img.tolist(), inf_box_img.tolist())
        inf_area = max(1.0, float((inf_box_img[2] - inf_box_img[0]) * (inf_box_img[3] - inf_box_img[1])))
        if inter / inf_area < 0.35:
            continue
        content = str(inf.get("content", "") or "")
        span_pdf = [float(v) for v in inf["bbox_pdf"]]
        # Do not compute/attach char_positions for inline equations.
        char_positions: List[Dict[str, Any]] = []
        span_candidates.append(
            _build_span_candidate(
                span_bbox_img=inf_box_img.tolist(),
                span_bbox_pdf=span_pdf,
                score=1.0,
                content=content,
                span_type="inline_equation",
                char_positions=char_positions,
                char_source="inline_formula_block",
            )
        )
        consumed_inline_formula_indices.add(inf["index"])


def _fuse_inline_formula_into_spans(
    *,
    span_candidates: List[Dict[str, Any]],
    mtype: str,
    bbox_img: np.ndarray,
    inline_formula_blocks: List[Dict[str, Any]],
    consumed_inline_formula_indices: set,
    pdfium_chars: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    _inject_inline_formula_spans(
        span_candidates=span_candidates,
        mtype=mtype,
        bbox_img=bbox_img,
        inline_formula_blocks=inline_formula_blocks,
        consumed_inline_formula_indices=consumed_inline_formula_indices,
        pdfium_chars=pdfium_chars,
    )
    # NOTE: De-dup / trimming logic intentionally disabled to avoid
    # dropping characters that exist in content but are missing in char_positions.
    return span_candidates


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
    table_res_list: List[Dict[str, Any]] = result.get("table_res_list", []) or []
    overall_ocr = result["overall_ocr_res"]
    rec_boxes, rec_texts, rec_scores = overall_ocr["rec_boxes"], overall_ocr["rec_texts"], overall_ocr["rec_scores"]
    rec_labels_raw = overall_ocr.get("rec_labels", [])
    rec_labels = [] if rec_labels_raw is None else list(rec_labels_raw)
    text_source_raw = str(overall_ocr.get("text_source", "") or "").strip().lower()
    if text_source_raw == "ocr_text":
        effective_char_source = "ocr_text"
    elif text_source_raw == "pdf_text":
        effective_char_source = "pdf_text"
    else:
        effective_char_source = str(char_source)

    # MinerU-style inline formula handling:
    # collect once, then fuse into text/title spans via a dedicated helper.
    inline_formula_blocks = _collect_inline_formula_blocks(parsing_res_list, img_w, img_h, pdf_w, pdf_h)
    consumed_inline_formula_indices = set()

    text_blocks: List[Tuple[int, np.ndarray]] = []
    for bi, block in enumerate(parsing_res_list):
        # Exclude inline_formula blocks from text assignment; they will be merged as spans.
        if str(getattr(block, "label", "") or "").lower() == "inline_formula":
            continue
        if layout_label_to_middle_para_type(block.label) in (
            "text",
            "title",
            "content",
            "number",
            "formula_number",
        ):
            text_blocks.append((bi, np.asarray(block.bbox, dtype=np.float64)))
    assign = assign_ocr_indices_exclusive_by_coverage(rec_boxes, text_blocks, min_ocr_coverage)

    para_blocks: List[Dict[str, Any]] = []
    discarded_blocks: List[Dict[str, Any]] = []
    global_char_idx = 0
    # Caption de-dup:
    # if a parsing block is bound as a table caption (table_caption) it should not be exported
    # again as a standalone top-level `type: "title"`.
    processed_caption_indices: set[int] = set()
    consumed_caption_indices = set()
    table_seen = 0
    used_table_res_indices: set[int] = set()
    global_line_index = 0

    # Pre-pass: mine which parsing blocks will be used as table captions.
    # This prevents earlier title blocks (before the table block is processed) from being
    # exported and later duplicated inside `table.blocks`.
    _pass1_consumed_caption_indices: set[int] = set()

    def _pass1_is_table_caption_candidate(idx: int) -> bool:
        if idx < 0 or idx >= len(parsing_res_list):
            return False
        if idx in _pass1_consumed_caption_indices:
            return False
        blk = parsing_res_list[idx]
        if _is_discarded_block_label(str(getattr(blk, "label", "") or "")):
            return False
        if str(getattr(blk, "label", "") or "").lower() == "inline_formula":
            return False
        raw_label = str(getattr(blk, "label", "") or "").strip().lower()
        return raw_label in ("table_caption", "figure_title", "figuretitle")

    for bi, block in enumerate(parsing_res_list):
        if layout_label_to_middle_para_type(block.label) != "table":
            continue
        bbox_img = np.asarray(block.bbox, dtype=np.float64)
        caption_candidate_bis: List[int] = []
        up = bi - 1
        while up >= 0 and _pass1_is_table_caption_candidate(up):
            up_blk = parsing_res_list[up]
            up_bbox = np.asarray(up_blk.bbox, dtype=np.float64)
            v_gap = float(bbox_img[1]) - float(up_bbox[3])
            overlap = _bbox_overlap_x(up_bbox, bbox_img)
            up_txt = str(getattr(up_blk, "content", "") or "").strip()
            if up_txt and -6.0 <= v_gap <= 48.0 and overlap >= 0.45:
                caption_candidate_bis.append(int(up))
                up -= 1
                continue
            break

        kept_caps, _kept_foots = fix_two_layer_blocks(
            body_bi=int(bi),
            caption_bis=caption_candidate_bis,
            footnote_bis=[],
        )
        for c_bi in kept_caps:
            c_bi_int = int(c_bi)
            _pass1_consumed_caption_indices.add(c_bi_int)
            processed_caption_indices.add(c_bi_int)

    def _next_line_index() -> int:
        nonlocal global_line_index
        idx = int(global_line_index)
        global_line_index += 1
        return idx

    def _export_text_like_block_as_child(
        *,
        bi: int,
        block: LayoutBlock,
        forced_type: str,
    ) -> Dict[str, Any]:
        """
        Export a parsing_res_list text/title block into a child block payload (caption/footnote),
        reusing the same span/line construction logic so inline_equation spans are preserved.
        """
        nonlocal global_char_idx
        bbox_img_local = np.asarray(block.bbox, dtype=np.float64)
        pdf_bbox_local = img_axis_aligned_box_to_pdf_top_left(bbox_img_local, img_w, img_h, pdf_w, pdf_h)

        span_candidates_local: List[Dict[str, Any]] = []
        for ocr_i in assign.get(int(bi), []):
            txt = sanitize_middle_text(str(rec_texts[ocr_i]))
            if not txt:
                continue
            sc = float(rec_scores[ocr_i]) if ocr_i < len(rec_scores) else 1.0
            ocr_img = rec_boxes[ocr_i]
            span_pdf = img_axis_aligned_box_to_pdf_top_left(ocr_img, img_w, img_h, pdf_w, pdf_h)
            char_positions, span_char_source, global_char_idx = _pick_char_positions_for_span(
                text=txt,
                ocr_img_box=ocr_img,
                span_pdf_box=span_pdf,
                pdfium_chars=pdfium_chars,
                preferred_source=effective_char_source,
                global_char_idx=global_char_idx,
            )
            span_content = _resolve_span_content(txt, char_positions, span_char_source)
            rec_label = str(rec_labels[ocr_i]).lower() if ocr_i < len(rec_labels) else "text"
            span_type = "inline_equation" if rec_label in ("formula", "inline_formula", "inline_equation") else "text"
            span_candidates_local.append(
                _build_span_candidate(
                    span_bbox_img=ocr_img,
                    span_bbox_pdf=span_pdf,
                    score=sc,
                    content=span_content,
                    span_type=span_type,
                    char_positions=char_positions,
                    char_source=span_char_source,
                )
            )

        span_candidates_local = _fuse_inline_formula_into_spans(
            span_candidates=span_candidates_local,
            mtype="text",
            bbox_img=bbox_img_local,
            inline_formula_blocks=inline_formula_blocks,
            consumed_inline_formula_indices=consumed_inline_formula_indices,
            pdfium_chars=pdfium_chars,
        )
        lines_out_local: List[Dict[str, Any]] = []
        for _, spans_list in enumerate(group_spans_to_lines(span_candidates_local)):
            spans_list = merge_line_spans_by_runs(spans_list)
            line_img_box = bbox_union([s["span_bbox_img"] for s in spans_list])
            line_pdf = img_axis_aligned_box_to_pdf_top_left(line_img_box, img_w, img_h, pdf_w, pdf_h)
            lines_out_local.append(
                {
                    "bbox": [round(x, 2) for x in line_pdf],
                    "spans": [_to_middle_span(s) for s in spans_list],
                    "index": _next_line_index(),
                }
            )
        if not lines_out_local and (block.content or "").strip():
            span_pdf = pdf_bbox_local
            txt = sanitize_middle_text(str(block.content))
            fb_line_idx = _next_line_index()
            lines_out_local.append(
                {
                    "bbox": [round(x, 2) for x in span_pdf],
                    "spans": [
                        {
                            "bbox": [round(x, 2) for x in span_pdf],
                            "score": 1.0,
                            "content": txt,
                            "type": "text",
                            "char_positions": [],
                            "char_source": "block_fallback",
                        }
                    ],
                    "index": fb_line_idx,
                }
            )

        child_block = {
            "type": forced_type,
            "bbox": [round(x, 2) for x in pdf_bbox_local],
            "group_id": 0,
            "lines": lines_out_local,
        }
        child_block["index"] = _compute_para_block_index(child_block, float(bi))
        return child_block

    def _select_table_res_item(table_bbox_img: np.ndarray) -> Dict[str, Any]:
        """
        Pick the best-matching table_res_list item for the current layout table block.
        Prefer IoU matching when table_res items expose a table-level bbox; otherwise
        fall back to sequential consumption (table_seen).
        """
        nonlocal table_seen
        # IoU match when possible.
        scored: List[Tuple[float, int]] = []
        for i, it in enumerate(table_res_list):
            if i in used_table_res_indices:
                continue
            bb = _table_res_item_bbox_img(it)
            if bb is None:
                continue
            iou = _bbox_iou(bb, table_bbox_img.tolist())
            if iou > 0.0:
                scored.append((float(iou), int(i)))
        if scored:
            scored.sort(key=lambda t: t[0], reverse=True)
            best_iou, best_i = scored[0]
            # A small threshold avoids accidental matches across far-away tables.
            if best_iou >= 0.2:
                used_table_res_indices.add(best_i)
                return table_res_list[best_i]

        # Sequential fallback (keeps previous behavior).
        if table_seen < len(table_res_list):
            idx = int(table_seen)
            table_seen += 1
            used_table_res_indices.add(idx)
            return table_res_list[idx]
        table_seen += 1
        return {}

    def _collect_block_line_indices(node: Any, out: List[float]) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("lines"), list):
                for ln in node["lines"]:
                    if isinstance(ln, dict) and isinstance(ln.get("index"), (int, float)):
                        out.append(float(ln["index"]))
            if isinstance(node.get("virtual_lines"), list):
                for vln in node["virtual_lines"]:
                    if isinstance(vln, dict) and isinstance(vln.get("index"), (int, float)):
                        out.append(float(vln["index"]))
            if isinstance(node.get("blocks"), list):
                for blk in node["blocks"]:
                    _collect_block_line_indices(blk, out)
            return
        if isinstance(node, list):
            for item in node:
                _collect_block_line_indices(item, out)

    def _compute_para_block_index(block_dict: Dict[str, Any], fallback: float) -> float:
        idxs: List[float] = []
        _collect_block_line_indices(block_dict, idxs)
        if not idxs:
            return float(fallback)
        return float(sum(idxs) / len(idxs))

    def _finalize_para_block(block_dict: Dict[str, Any], fallback: float) -> Dict[str, Any]:
        block_dict["index"] = _compute_para_block_index(block_dict, fallback)
        return block_dict

    for bi, block in enumerate(parsing_res_list):
        if bi in processed_caption_indices:
            continue
        # Skip exporting inline_formula as standalone blocks; they are merged into text spans below.
        if str(getattr(block, "label", "") or "").lower() == "inline_formula":
            continue
        if _is_discarded_block_label(block.label):
            bbox_img = np.asarray(block.bbox, dtype=np.float64)
            pdf_bbox = img_axis_aligned_box_to_pdf_top_left(bbox_img, img_w, img_h, pdf_w, pdf_h)
            discarded_blocks.append(
                {
                    "label": str(block.label).lower(),
                    "bbox": [round(x, 2) for x in pdf_bbox],
                    "content": str(getattr(block, "content", "") or ""),
                    "index": bi,
                }
            )
            continue
        mtype = layout_label_to_middle_para_type(block.label)
        bbox_img = np.asarray(block.bbox, dtype=np.float64)
        pdf_bbox = img_axis_aligned_box_to_pdf_top_left(bbox_img, img_w, img_h, pdf_w, pdf_h)

        if mtype == "image":
            img_path = ""
            if getattr(block, "image", None) and isinstance(block.image, dict):
                img_path = str(block.image.get("path", "") or "")
            body_bbox = [round(v, 2) for v in pdf_bbox]
            body_bbox_float = [float(v) for v in pdf_bbox]
            body_score = float(getattr(block, "score", 1.0) or 1.0)

            # Export MinerU-style image caption as a sub-block, not a standalone para_block.
            # Common cases:
            # - caption above image (prev block)
            # - caption below image (next block)
            caption_payload: Dict[str, Any] | None = None
            cand_indices = [ci for ci in (bi - 1, bi + 1) if 0 <= ci < len(parsing_res_list)]
            best = None
            best_key = None
            for ci in cand_indices:
                if ci in consumed_caption_indices:
                    continue
                cblk = parsing_res_list[ci]
                clabel = str(getattr(cblk, "label", "") or "").lower()
                if clabel not in ("figure_title", "chart_title"):
                    continue
                ctype = layout_label_to_middle_para_type(cblk.label)
                if ctype not in ("text", "title"):
                    continue
                cbox = np.asarray(cblk.bbox, dtype=np.float64)
                # vertical relationship
                gap_above = float(bbox_img[1]) - float(cbox[3])  # caption above image
                gap_below = float(cbox[1]) - float(bbox_img[3])  # caption below image
                if not (-6.0 <= gap_above <= 64.0 or -6.0 <= gap_below <= 64.0):
                    continue
                overlap = _bbox_overlap_x(cbox, bbox_img)
                if overlap < 0.35:
                    continue
                # prefer closer gap; if tie, prefer below-caption (more common)
                gap = gap_above if gap_above >= -6.0 else gap_below
                prefer_below = 0 if gap_below >= -6.0 else 1
                key = (abs(gap), prefer_below)
                if best is None or key < best_key:
                    best = (ci, cblk, cbox)
                    best_key = key
            if best is not None:
                ci, cblk, cbox = best
                cap_txt = str(getattr(cblk, "content", "") or "").strip()
                if cap_txt:
                    cap_txt = sanitize_middle_text(cap_txt)
                if cap_txt:
                    cap_pdf = img_axis_aligned_box_to_pdf_top_left(cbox, img_w, img_h, pdf_w, pdf_h)
                    caption_bbox = [round(v, 2) for v in cap_pdf]
                    caption_chars, cap_char_source, global_char_idx = _pick_char_positions_for_span(
                        text=cap_txt,
                        ocr_img_box=cbox,
                        span_pdf_box=caption_bbox,
                        pdfium_chars=pdfium_chars,
                        preferred_source=effective_char_source,
                        global_char_idx=global_char_idx,
                    )
                    # Delay index assignment so image body stays ahead of caption in reading order.
                    caption_payload = {
                        "ci": ci,
                        "bbox": caption_bbox,
                        "content": cap_txt,
                        "char_positions": caption_chars,
                        "char_source": cap_char_source,
                    }

            # Keep para_blocks[].bbox as image body bbox (to match existing middle json shape).
            overall_bbox = body_bbox

            # Add virtual_lines for image body: split into 3 rows.
            virtual_lines = _build_split_virtual_lines(body_bbox_float, split_count=3, precision=4)
            # Assign line indices by PDF top->bottom position (smaller y first).
            pending_line_items: List[Dict[str, Any]] = []
            for i, vln in enumerate(virtual_lines):
                vb = vln["bbox"]
                v_center_y = (float(vb[1]) + float(vb[3])) * 0.5
                pending_line_items.append({"kind": "virtual", "slot": i, "y": v_center_y})
            if caption_payload is not None:
                cb = caption_payload["bbox"]
                pending_line_items.append(
                    {
                        "kind": "caption",
                        "slot": 0,
                        "y": (float(cb[1]) + float(cb[3])) * 0.5,
                    }
                )
            pending_line_items.sort(key=lambda it: float(it["y"]))
            cap_line_idx = -1
            for item in pending_line_items:
                idx = _next_line_index()
                if item["kind"] == "virtual":
                    virtual_lines[int(item["slot"])]["index"] = idx
                else:
                    cap_line_idx = idx
            # Reuse the center virtual line index for image_body; do not consume an extra global index.
            body_center_idx = int(virtual_lines[1]["index"]) if len(virtual_lines) >= 2 else _next_line_index()
            caption_blocks: List[Dict[str, Any]] = []
            if caption_payload is not None:
                # Disable approx fallback: keep empty char positions if PDF chars unavailable.
                if cap_line_idx < 0:
                    cap_line_idx = _next_line_index()
                caption_blocks.append(
                    _build_text_caption_block(
                        caption_type="image_caption",
                        bbox=caption_payload["bbox"],
                        content=caption_payload["content"],
                        char_positions=caption_payload["char_positions"],
                        line_index=cap_line_idx,
                        char_source=caption_payload.get("char_source", effective_char_source),
                    )
                )
                consumed_caption_indices.add(int(caption_payload["ci"]))
                processed_caption_indices.add(int(caption_payload["ci"]))

            image_para_block = {
                "type": "image",
                "bbox": overall_bbox,
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
                                        "score": body_score,
                                        "type": "image",
                                        "image_path": img_path,
                                    }
                                ],
                            }
                        ],
                        "virtual_lines": virtual_lines,
                        "index": body_center_idx,
                    }
                ]
                + caption_blocks,
            }
            para_blocks.append(_finalize_para_block(image_para_block, float(bi)))
            continue

        if mtype == "table":
            body_bbox = [round(v, 2) for v in pdf_bbox]
            table_img_path = ""
            if getattr(block, "image", None) and isinstance(block.image, dict):
                table_img_path = str(block.image.get("path", "") or "")
            # --- pick the best-matching table_res item (avoid cross-filling) ---
            table_item: Dict[str, Any] = {}
            best_i = -1
            best_iou = 0.0
            for i, it in enumerate(table_res_list):
                if i in used_table_res_indices:
                    continue
                raw_cells = it.get("cell_box_list", [])
                if raw_cells is None:
                    continue
                if isinstance(raw_cells, np.ndarray):
                    cell_boxes = raw_cells.tolist()
                else:
                    cell_boxes = list(raw_cells)
                if not cell_boxes:
                    continue
                try:
                    cell_boxes_np = [np.asarray([float(v) for v in b], dtype=np.float64) for b in cell_boxes]
                except Exception:
                    continue
                if not cell_boxes_np:
                    continue
                table_box_img = bbox_union([b.tolist() for b in cell_boxes_np])
                iou = _bbox_iou(table_box_img, bbox_img.tolist())
                if iou > best_iou:
                    best_iou = float(iou)
                    best_i = int(i)
            if best_i >= 0 and best_iou >= 0.15:
                used_table_res_indices.add(best_i)
                table_item = table_res_list[best_i]
            else:
                table_item = _select_table_res_item(bbox_img)

            table_html = str(table_item.get("pred_html", "") or block.content or "")

            # --- Mine caption/footnote candidates from the adjacency chain (MinerU-like) ---
            caption_candidate_bis: List[int] = []
            # Temporary strategy: table footnote matching is disabled.
            footnote_candidate_bis: List[int] = []

            def _is_table_caption_candidate(idx: int) -> bool:
                if idx < 0 or idx >= len(parsing_res_list):
                    return False
                if idx in consumed_caption_indices:
                    return False
                blk = parsing_res_list[idx]
                if _is_discarded_block_label(str(getattr(blk, "label", "") or "")):
                    return False
                if str(getattr(blk, "label", "") or "").lower() == "inline_formula":
                    return False
                raw_label = str(getattr(blk, "label", "") or "").strip().lower()
                # Restrict table caption source to explicit caption-like labels only.
                # Keep table_caption if exists upstream; add figure_title/figuretitle mapping.
                return raw_label in ("table_caption", "figure_title", "figuretitle")

            # scan upwards for caption chain
            up = bi - 1
            while up >= 0 and _is_table_caption_candidate(up):
                up_blk = parsing_res_list[up]
                up_bbox = np.asarray(up_blk.bbox, dtype=np.float64)
                v_gap = float(bbox_img[1]) - float(up_bbox[3])
                overlap = _bbox_overlap_x(up_bbox, bbox_img)
                up_txt = str(getattr(up_blk, "content", "") or "").strip()
                if up_txt and -6.0 <= v_gap <= 48.0 and overlap >= 0.45:
                    caption_candidate_bis.append(int(up))
                    up -= 1
                    continue
                break

            # Footnote path disabled for table export.
            footnote_candidate_bis = []

            # continuity filtering (two-layer fix)
            kept_caps, kept_foots = fix_two_layer_blocks(
                body_bi=int(bi),
                caption_bis=caption_candidate_bis,
                footnote_bis=footnote_candidate_bis,
            )

            caption_blocks: List[Dict[str, Any]] = []
            for c_bi in kept_caps:
                caption_blocks.append(
                    _export_text_like_block_as_child(
                        bi=int(c_bi),
                        block=parsing_res_list[int(c_bi)],
                        forced_type="table_caption",
                    )
                )
                consumed_caption_indices.add(int(c_bi))
                processed_caption_indices.add(int(c_bi))

            table_cells, global_char_idx = _build_table_cells(
                table_item,
                table_html,
                pdfium_chars,
                img_w,
                img_h,
                pdf_w,
                pdf_h,
                global_char_idx,
                effective_char_source,
            )
            virtual_lines = _build_table_virtual_lines(
                table_item, img_w, img_h, pdf_w, pdf_h, body_bbox
            )
            for vl in virtual_lines:
                vl["index"] = _next_line_index()
            table_para_block = {
                "type": "table",
                "bbox": body_bbox,
                "blocks": [],
            }
            table_body_block = {
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
                                "html": table_html,
                                "image_path": table_img_path,
                            }
                        ],
                    }
                ],
                "virtual_lines": virtual_lines,
                "cells": table_cells,
                "index": float(sum(float(vl["index"]) for vl in virtual_lines) / len(virtual_lines))
                if virtual_lines
                else float(bi),
            }
            table_para_block["blocks"] = caption_blocks + [table_body_block]
            table_para_block["blocks"].sort(key=lambda b: float(b.get("index", 0.0)))
            para_blocks.append(_finalize_para_block(table_para_block, float(bi)))
            continue

        if mtype == "interline_equation":
            span_pdf = [float(x) for x in pdf_bbox]
            txt = block.content or ""
            # MinerU-style equation export: keep equation text and image crop path.
            eq_img_path = ""
            if getattr(block, "image", None) and isinstance(block.image, dict):
                eq_img_path = str(block.image.get("path", "") or "")
            # Build simple virtual_lines by splitting bbox into two rows (better than none).
            x1, y1, x2, y2 = [float(v) for v in span_pdf]
            mid_y = (y1 + y2) * 0.5
            virtual_lines = [
                {"bbox": [round(x1, 2), round(mid_y, 2), round(x2, 2), round(y2, 2)], "spans": [], "index": _next_line_index()},
                {"bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(mid_y, 2)], "spans": [], "index": _next_line_index()},
            ]
            eq_para_block = {
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
                                "image_path": eq_img_path,
                            }
                        ],
                    }
                ],
                "virtual_lines": virtual_lines,
            }
            para_blocks.append(_finalize_para_block(eq_para_block, float(bi)))
            continue

        span_candidates: List[Dict[str, Any]] = []
        for ocr_i in assign.get(bi, []):
            txt = sanitize_middle_text(str(rec_texts[ocr_i]))
            if not txt:
                continue
            sc = float(rec_scores[ocr_i]) if ocr_i < len(rec_scores) else 1.0
            ocr_img = rec_boxes[ocr_i]
            span_pdf = img_axis_aligned_box_to_pdf_top_left(ocr_img, img_w, img_h, pdf_w, pdf_h)
            char_positions, used_char_source, global_char_idx = _pick_char_positions_for_span(
                text=txt,
                ocr_img_box=ocr_img,
                span_pdf_box=span_pdf,
                pdfium_chars=pdfium_chars,
                preferred_source=effective_char_source,
                global_char_idx=global_char_idx,
            )
            span_content = _resolve_span_content(txt, char_positions, used_char_source)
            rec_label = str(rec_labels[ocr_i]).lower() if ocr_i < len(rec_labels) else "text"
            span_type = "inline_equation" if rec_label in ("formula", "inline_formula", "inline_equation") else "text"
            span_candidates.append(
                _build_span_candidate(
                    span_bbox_img=ocr_img,
                    span_bbox_pdf=span_pdf,
                    score=sc,
                    content=span_content,
                    span_type=span_type,
                    char_positions=char_positions,
                    char_source=used_char_source,
                )
            )

        # Fuse layout inline_formula spans and run text/equation de-dup in one place.
        span_candidates = _fuse_inline_formula_into_spans(
            span_candidates=span_candidates,
            mtype=mtype,
            bbox_img=bbox_img,
            inline_formula_blocks=inline_formula_blocks,
            consumed_inline_formula_indices=consumed_inline_formula_indices,
            pdfium_chars=pdfium_chars,
        )
        lines_out: List[Dict[str, Any]] = []
        for line_i, line_spans in enumerate(group_spans_to_lines(span_candidates)):
            line_spans = merge_line_spans_by_runs(line_spans)
            line_img_box = bbox_union([s["span_bbox_img"] for s in line_spans])
            line_pdf = img_axis_aligned_box_to_pdf_top_left(line_img_box, img_w, img_h, pdf_w, pdf_h)
            lines_out.append(
                {
                    "bbox": [round(x, 2) for x in line_pdf],
                    "spans": [_to_middle_span(s) for s in line_spans],
                    "index": _next_line_index(),
                }
            )
        if not lines_out and (block.content or "").strip():
            span_pdf = pdf_bbox
            txt = block.content
            # Disable approx fallback: keep empty char positions if PDF chars unavailable.
            char_positions = []
            fb_line_idx = _next_line_index()
            lines_out.append(
                {
                    "bbox": [round(x, 2) for x in span_pdf],
                    "spans": [{"bbox": [round(x, 2) for x in span_pdf], "score": 1.0, "content": txt, "type": "text", "char_positions": char_positions, "char_source": "block_fallback"}],
                    "index": fb_line_idx,
                }
            )
        text_para_block = {"type": mtype, "bbox": [round(x, 2) for x in pdf_bbox], "lines": lines_out}
        para_blocks.append(_finalize_para_block(text_para_block, float(bi)))

    para_blocks = [p for p in para_blocks if p.get("type") not in ("text", "title") or len(p.get("lines", [])) > 0]
    page_dict = {
        "pdf_info": [
            {
                "page_idx": page_idx,
                "page_size": [int(round(pdf_w)), int(round(pdf_h))],
                "para_blocks": para_blocks,
                "discarded_blocks": discarded_blocks,
            }
        ]
    }
    _flip_export_bboxes_to_bottom_left(page_dict, float(pdf_h))
    return page_dict

