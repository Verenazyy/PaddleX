import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pypdfium2 as pdfium


def pdf_xy_to_px(
    x_pdf: float, y_pdf: float, pdf_w: float, pdf_h: float, img_w: int, img_h: int
) -> Tuple[int, int]:
    """
    Convert PDF coordinates (origin at bottom-left) to image pixel coordinates
    (origin at top-left).
    """
    x_px = int(round((x_pdf / pdf_w) * img_w))
    y_px = int(round(((pdf_h - y_pdf) / pdf_h) * img_h))
    x_px = max(0, min(img_w - 1, x_px))
    y_px = max(0, min(img_h - 1, y_px))
    return x_px, y_px


def extract_textlayer_rects(
    pdf_path: Path, page_index: int, scale: float = 2.0
) -> Dict[str, Any]:
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page = doc.get_page(page_index)
        try:
            pdf_w = float(page.get_width())
            pdf_h = float(page.get_height())

            # Render page so we can verify mapped boxes visually.
            image = page.render(scale=scale).to_numpy()
            img_h, img_w = image.shape[:2]

            textpage = page.get_textpage()

            text_len = None
            if hasattr(textpage, "count_chars"):
                text_len = int(textpage.count_chars())
            elif hasattr(textpage, "get_char_count"):
                text_len = int(textpage.get_char_count())

            rect_count = 0
            if text_len is not None and hasattr(textpage, "count_rects"):
                rect_count = int(textpage.count_rects(0, text_len))

            rect_items: List[Dict[str, Any]] = []
            for ridx in range(rect_count):
                if not hasattr(textpage, "get_rect"):
                    break
                left, bottom, right, top = textpage.get_rect(ridx)
                if right <= left or top <= bottom:
                    continue

                txt = ""
                if hasattr(textpage, "get_text_bounded"):
                    try:
                        txt = textpage.get_text_bounded(left, bottom, right, top) or ""
                    except Exception:
                        txt = ""
                txt = " ".join(txt.split())
                if not txt:
                    continue

                x1, y2 = pdf_xy_to_px(left, bottom, pdf_w, pdf_h, img_w, img_h)
                x2, y1 = pdf_xy_to_px(right, top, pdf_w, pdf_h, img_w, img_h)
                x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
                y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)

                if x_max <= x_min or y_max <= y_min:
                    continue

                rect_items.append(
                    {
                        "text": txt,
                        "bbox": [x_min, y_min, x_max, y_max],
                    }
                )

            return {
                "page_index": page_index,
                "pdf_size": [pdf_w, pdf_h],
                "img_size": [img_w, img_h],
                "rect_count": len(rect_items),
                "char_count": sum(len(it["text"]) for it in rect_items),
                "rects": rect_items,
                "image": image,
            }
        finally:
            page.close()
    finally:
        doc.close()


def draw_rects(image: np.ndarray, rects: List[Dict[str, Any]]) -> np.ndarray:
    vis = image.copy()
    for it in rects:
        x1, y1, x2, y2 = it["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify PDFium textlayer extraction and coordinate mapping only."
    )
    parser.add_argument("pdf", type=str, help="Input PDF path")
    parser.add_argument("--page-index", type=int, default=0, help="0-based page index")
    parser.add_argument("--scale", type=float, default=2.0, help="Render scale for image")
    parser.add_argument(
        "--out-dir", type=str, default="./output_verify_pdfium_only", help="Output directory"
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = extract_textlayer_rects(pdf_path, args.page_index, args.scale)
    vis = draw_rects(res["image"], res["rects"])

    vis_path = out_dir / f"{pdf_path.stem}_page{args.page_index}_vis.jpg"
    json_path = out_dir / f"{pdf_path.stem}_page{args.page_index}.json"

    cv2.imwrite(str(vis_path), vis)
    with json_path.open("w", encoding="utf-8") as f:
        to_dump = {k: v for k, v in res.items() if k != "image"}
        json.dump(to_dump, f, ensure_ascii=False, indent=2)

    print(f"page_index={res['page_index']}")
    print(f"pdf_size={res['pdf_size']} img_size={res['img_size']}")
    print(f"rect_count={res['rect_count']} char_count={res['char_count']}")
    print(f"saved: {vis_path}")
    print(f"saved: {json_path}")


if __name__ == "__main__":
    main()

