import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _guess_ocr_source(overall_ocr_json: Dict[str, Any]) -> str:
    """
    Heuristic:
    - pdfium textlayer path in pipeline_v2 builds OCRResult with:
      - text_det_params == {}
      - text_rec_score_thresh == 0.0
      - rec_scores all 1.0
      - model_settings.use_doc_preprocessor == False
    """
    try:
        ms = overall_ocr_json.get("model_settings", {}) or {}
        det_params = overall_ocr_json.get("text_det_params", None)
        rec_thresh = overall_ocr_json.get("text_rec_score_thresh", None)
        rec_scores = overall_ocr_json.get("rec_scores", []) or []
        if (
            ms.get("use_doc_preprocessor") is False
            and det_params == {}
            and rec_thresh == 0.0
            and rec_scores
            and all(float(s) == 1.0 for s in rec_scores)
        ):
            return "pdfium_textlayer"
    except Exception:
        pass
    return "ocr_fallback_or_other"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify PP-StructureV3 PDF textlayer-first + OCR fallback behavior."
    )
    parser.add_argument("pdf", type=str, help="Path to a PDF file.")
    parser.add_argument(
        "--out",
        type=str,
        default="output_verify_pp_structure_v3_pdf",
        help="Output directory to save json (optional).",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save per-page LayoutParsingResultV2 json to --out.",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save per-page visualization images to --out.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    # Import lazily to keep CLI responsive
    from paddlex import create_pipeline

    pipeline = create_pipeline(pipeline="PP-StructureV3")

    out_dir = Path(args.out)
    if args.save_json:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Try to keep runtime smaller: disable extra sub-pipelines unless you need them.
    # (Textlayer-first behavior is in overall OCR stage, independent of table/seal/etc.)
    results = pipeline.predict(
        str(pdf_path),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_seal_recognition=False,
        use_table_recognition=False,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_region_detection=False,
    )

    for i, res in enumerate(results):
        j = res.json["res"]
        overall = j["overall_ocr_res"]
        src = _guess_ocr_source(overall)
        rec_texts = overall.get("rec_texts", []) or []
        total_chars = sum(len(str(t)) for t in rec_texts)
        num_boxes = len(rec_texts)
        page_index = j.get("page_index", None)
        page_count = j.get("page_count", None)

        print(
            f"[page {page_index}/{page_count}] source={src} boxes={num_boxes} chars={total_chars}"
        )

        if args.save_json:
            with (out_dir / f"page_{i}_result.json").open("w", encoding="utf-8") as f:
                json.dump(j, f, ensure_ascii=False, indent=2)
        if args.save_vis:
            # Use built-in visualization exporter from LayoutParsingResultV2.
            res.save_to_img(str(out_dir))


if __name__ == "__main__":
    main()

