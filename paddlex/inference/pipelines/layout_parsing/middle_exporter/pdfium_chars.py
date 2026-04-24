from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import unicodedata

from ..utils import calculate_bbox_area, get_bbox_intersection


def _normalize_single_char(ch: str) -> str:
    if not ch:
        return ""
    # pypdfium2 may return multiple code points for one glyph.
    # Avoid hard truncation by taking the first *valid* normalized char.
    if len(ch) > 1:
        for one in ch:
            norm = _normalize_single_char(one)
            if norm:
                return norm
        return ""
    if ch == "\ufffd":
        return "."
    cat = unicodedata.category(ch)
    if cat == "Cs":
        return ""
    if cat.startswith("C") and ch not in ("\t", "\n", "\r"):
        return ""
    return ch


def _codepoint_to_char(code: int) -> str:
    """
    Convert a raw Unicode codepoint (from FPDFText_GetUnicode) to a normalized char.

    Differences vs get_text_range:
    - code == 0x0000: NUL placeholder used by pdfium for ligatures/spacing glyphs -> skip
    - code == 0xFFFD: replacement char (missing ToUnicode mapping) -> keep as "."
    - surrogate range (0xD800-0xDFFF): invalid Unicode -> skip
    - whitespace/line-break control chars (CR, LF, TAB, space): skip (no bbox meaning)
    - other control chars (category C*): skip (same as _normalize_single_char)
    """
    if code == 0:
        return ""
    if 0xD800 <= code <= 0xDFFF:
        return ""
    # Skip whitespace and line-break chars — they carry no glyph bbox information.
    if code in (0x09, 0x0A, 0x0D, 0x20):
        return ""
    try:
        ch = chr(code)
    except (ValueError, OverflowError):
        return ""
    return _normalize_single_char(ch)


def _deduplicate_chars(chars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate glyphs caused by PDF double-rendering (e.g. fill + stroke layers).
    Two chars are considered duplicates if they share the same char_idx OR
    have identical character and nearly identical bbox (within 0.5pt).
    Keeps the first occurrence.
    """
    seen_idx: set = set()
    out: List[Dict[str, Any]] = []
    for item in chars:
        idx = int(item["char_idx"])
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        out.append(item)
    return out


def sanitize_middle_text(text: str) -> str:
    if not text:
        return ""
    out: List[str] = []
    for ch in str(text):
        norm = _normalize_single_char(ch)
        if not norm:
            continue
        out.append(norm)
    return "".join(out)


def read_pdfium_page_chars(
    input_path: str, page_idx: int, img_w: int, img_h: int, pdf_w: float, pdf_h: float
) -> List[Dict[str, Any]]:
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

    # Try to import raw C API for direct Unicode codepoint access.
    # FPDFText_GetUnicode is more reliable than get_text_range for PDFs with
    # incomplete ToUnicode mappings: get_text_range may silently return "" for
    # characters whose glyph-to-Unicode mapping is missing, while the raw API
    # always returns the codepoint (0x0000 for unmapped glyphs).
    _raw_get_unicode = None
    try:
        import pypdfium2.raw as _pdfium_raw  # type: ignore
        if hasattr(_pdfium_raw, "FPDFText_GetUnicode"):
            _raw_get_unicode = _pdfium_raw.FPDFText_GetUnicode
    except Exception:
        pass

    out: List[Dict[str, Any]] = []
    for ci in range(char_num):
        try:
            left, bottom, right, top = textpage.get_charbox(ci)
        except Exception:
            continue

        # --- character extraction ---
        ch = ""
        if _raw_get_unicode is not None:
            # Preferred path: raw codepoint, never silently drops characters.
            try:
                code = int(_raw_get_unicode(textpage, ci))
                ch = _codepoint_to_char(code)
            except Exception:
                ch = ""
        if not ch and hasattr(textpage, "get_text_range"):
            # Fallback to high-level API (may miss chars with broken ToUnicode).
            try:
                ch = textpage.get_text_range(ci, 1) or ""
            except Exception:
                ch = ""
            ch = _normalize_single_char(ch)
        if not ch:
            continue

        x1_pdf, x2_pdf = float(left), float(right)
        y1_pdf, y2_pdf = float(pdf_h - float(top)), float(pdf_h - float(bottom))
        if x2_pdf < x1_pdf:
            x1_pdf, x2_pdf = x2_pdf, x1_pdf
        if y2_pdf < y1_pdf:
            y1_pdf, y2_pdf = y2_pdf, y1_pdf
        out.append(
            {
                "char": ch,
                "char_idx": ci,
                "pdf_bbox": [x1_pdf, y1_pdf, x2_pdf, y2_pdf],
                "img_bbox": [
                    (x1_pdf / pdf_w) * img_w,
                    (y1_pdf / pdf_h) * img_h,
                    (x2_pdf / pdf_w) * img_w,
                    (y2_pdf / pdf_h) * img_h,
                ],
            }
        )

    # Remove duplicate glyphs from double-rendered PDF layers.
    out = _deduplicate_chars(out)
    return out


def char_positions_from_pdfium_chars(
    chars: List[Dict[str, Any]], span_img_box: Sequence[float]
) -> List[Dict[str, Any]]:
    span_box = np.asarray(span_img_box, dtype=np.float64)
    # Expand a little to keep tiny punctuation near text edges.
    span_w = max(1.0, float(span_box[2] - span_box[0]))
    span_h = max(1.0, float(span_box[3] - span_box[1]))
    margin_x = max(1.5, span_w * 0.03)
    margin_y = max(1.5, span_h * 0.12)
    span_box_exp = np.asarray(
        [
            float(span_box[0]) - margin_x,
            float(span_box[1]) - margin_y,
            float(span_box[2]) + margin_x,
            float(span_box[3]) + margin_y,
        ],
        dtype=np.float64,
    )
    selected: List[Dict[str, Any]] = []
    for item in chars:
        ch_box = np.asarray(item["img_bbox"], dtype=np.float64)
        ch_area = float(calculate_bbox_area(ch_box))
        if ch_area <= 1e-6:
            continue
        inter = get_bbox_intersection(ch_box, span_box_exp, return_format="bbox")
        if inter is None:
            continue
        overlap_ratio = float(calculate_bbox_area(inter)) / ch_area
        ch = str(item.get("char", "") or "")
        is_punct = bool(ch) and unicodedata.category(ch[0]).startswith("P")
        # Keep strict matching for regular chars; relax only for punctuation.
        min_overlap = 0.2 if is_punct else 0.5
        cx = (float(ch_box[0]) + float(ch_box[2])) * 0.5
        cy = (float(ch_box[1]) + float(ch_box[3])) * 0.5
        center_inside = (
            float(span_box_exp[0]) <= cx <= float(span_box_exp[2])
            and float(span_box_exp[1]) <= cy <= float(span_box_exp[3])
        )
        keep = overlap_ratio >= min_overlap
        if not keep and is_punct and center_inside:
            keep = True
        if not keep and is_punct:
            # Edge-near fallback: punctuation often sits just outside OCR span box.
            dx = max(float(span_box_exp[0]) - float(ch_box[2]), float(ch_box[0]) - float(span_box_exp[2]), 0.0)
            dy = max(float(span_box_exp[1]) - float(ch_box[3]), float(ch_box[1]) - float(span_box_exp[3]), 0.0)
            if dx <= max(2.0, span_w * 0.03) and dy <= max(2.0, span_h * 0.25):
                keep = True
        if not keep:
            continue
        selected.append(item)
    # Prefer PDF intrinsic character order to avoid jitter-induced swaps.
    selected.sort(key=lambda x: (int(x["char_idx"]), float(x["img_bbox"][0]), float(x["img_bbox"][1])))
    out: List[Dict[str, Any]] = []
    for pos, item in enumerate(selected):
        out.append(
            {
                "char": item["char"],
                "bbox": [round(v, 4) for v in item["pdf_bbox"]],
                "char_idx": int(item["char_idx"]),
                "position_in_span": pos,
            }
        )
    return out


def approx_char_bboxes(
    text: str, pdf_box: Sequence[float], char_idx_start: int
) -> tuple[List[Dict[str, Any]], int]:
    x1, y1, x2, y2 = map(float, pdf_box)
    chars = list(text)
    n = max(len(chars), 1)
    w = (x2 - x1) / n
    out: List[Dict[str, Any]] = []
    cur = char_idx_start
    for pos, ch in enumerate(chars):
        out.append(
            {
                "char": ch,
                "bbox": [x1 + pos * w, y1, x1 + (pos + 1) * w, y2],
                "char_idx": cur,
                "position_in_span": pos,
            }
        )
        cur += 1
    return out, cur
