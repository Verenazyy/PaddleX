from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def read_pdf_page_size_pts(input_path: str, page_index: int) -> Tuple[float, float] | None:
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
    box: Sequence[float], img_w: int, img_h: int, pdf_w: float, pdf_h: float
) -> List[float]:
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    return [x1 / img_w * pdf_w, y1 / img_h * pdf_h, x2 / img_w * pdf_w, y2 / img_h * pdf_h]


def pdf_top_left_box_to_img(
    box: Sequence[float], img_w: int, img_h: int, pdf_w: float, pdf_h: float
) -> List[int]:
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


def pdf_bottom_left_box_to_img(
    box: Sequence[float], img_w: int, img_h: int, pdf_w: float, pdf_h: float
) -> List[int]:
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    ix1 = int(round((x1 / pdf_w) * img_w))
    ix2 = int(round((x2 / pdf_w) * img_w))
    iy1 = int(round(((pdf_h - y2) / pdf_h) * img_h))
    iy2 = int(round(((pdf_h - y1) / pdf_h) * img_h))
    ix1 = max(0, min(img_w - 1, ix1))
    ix2 = max(0, min(img_w - 1, ix2))
    iy1 = max(0, min(img_h - 1, iy1))
    iy2 = max(0, min(img_h - 1, iy2))
    if ix2 < ix1:
        ix1, ix2 = ix2, ix1
    if iy2 < iy1:
        iy1, iy2 = iy2, iy1
    return [ix1, iy1, ix2, iy2]


def bbox_union(boxes: List[Sequence[float]]) -> List[float]:
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    arr = np.asarray(boxes, dtype=np.float64)
    return [
        float(np.min(arr[:, 0])),
        float(np.min(arr[:, 1])),
        float(np.max(arr[:, 2])),
        float(np.max(arr[:, 3])),
    ]

