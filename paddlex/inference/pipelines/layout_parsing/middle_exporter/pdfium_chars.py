from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from ..utils import calculate_bbox_area, get_bbox_intersection


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

    out: List[Dict[str, Any]] = []
    for ci in range(char_num):
        try:
            left, bottom, right, top = textpage.get_charbox(ci)
        except Exception:
            continue
        ch = ""
        if hasattr(textpage, "get_text_range"):
            try:
                ch = textpage.get_text_range(ci, 1) or ""
            except Exception:
                ch = ""
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
    return out


def char_positions_from_pdfium_chars(
    chars: List[Dict[str, Any]], span_img_box: Sequence[float]
) -> List[Dict[str, Any]]:
    span_box = np.asarray(span_img_box, dtype=np.float64)
    selected: List[Dict[str, Any]] = []
    for item in chars:
        ch_box = np.asarray(item["img_bbox"], dtype=np.float64)
        ch_area = float(calculate_bbox_area(ch_box))
        if ch_area <= 1e-6:
            continue
        inter = get_bbox_intersection(ch_box, span_box, return_format="bbox")
        if inter is None:
            continue
        if float(calculate_bbox_area(inter)) / ch_area < 0.5:
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

