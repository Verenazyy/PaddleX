import argparse
from pathlib import Path

from paddlex.inference.pipelines.layout_parsing.middle_export import (
    run_pp_structure_v3_to_middle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PP-StructureV3 and export middle_index.json + middle_pages/page_xxxx.json."
    )
    parser.add_argument("pdf", type=str, help="Input PDF path, e.g. B_007.pdf")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="auto",
        help="Output directory containing middle_index.json and middle_pages/",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.7,
        help="Minimum OCR-box coverage ratio in block (intersection_area / ocr_area).",
    )
    parser.add_argument(
        "--pages-subdir",
        type=str,
        default="middle_pages",
        help="Subdirectory name for per-page json files.",
    )
    parser.add_argument(
        "--char-source",
        type=str,
        default="pdf_text",
        help="char_source value stored in text spans when PDFium chars are used.",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save both built-in vis and middle line/span vis.",
    )
    parser.add_argument(
        "--use-doc-orientation-classify",
        action="store_true",
        help="Enable document orientation classify in predict.",
    )
    parser.add_argument(
        "--use-doc-unwarping",
        action="store_true",
        help="Enable document unwarping in predict.",
    )
    parser.add_argument(
        "--use-textline-orientation",
        action="store_true",
        help="Enable textline orientation in OCR predict.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_pp_structure_v3_to_middle(
        str(pdf_path),
        output_dir=str(out_dir),
        min_ocr_coverage=args.min_coverage,
        pages_subdir=args.pages_subdir,
        char_source=args.char_source,
        save_vis=args.save_vis,
        use_doc_orientation_classify=args.use_doc_orientation_classify,
        use_doc_unwarping=args.use_doc_unwarping,
        use_textline_orientation=args.use_textline_orientation,
    )

    print(f"done: {out_dir / 'middle_index.json'}")
    print(f"done: {out_dir / args.pages_subdir}")
    if args.save_vis:
        print(f"done: built-in vis + middle line/span vis under {out_dir}")


if __name__ == "__main__":
    main()
