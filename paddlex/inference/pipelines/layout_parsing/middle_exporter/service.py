from __future__ import annotations

from typing import Any, Optional

from .writer import export_middle_bundle


def run_pp_structure_v3_to_middle(
    pdf_path: str,
    output_dir: str = "auto",
    *,
    min_ocr_coverage: float = 0.7,
    pages_subdir: str = "middle_pages",
    char_source: str = "pdf_text",
    save_vis: bool = False,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_textline_orientation: bool = False,
    pipeline: Optional[Any] = None,
) -> None:
    """
    High-level API: run PP-StructureV3 on a PDF and export middle json files.

    Pass a pre-created ``pipeline`` (from ``create_pipeline("PP-StructureV3")``)
    to avoid reloading weights when processing many PDFs in a loop.
    """
    if pipeline is None:
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="PP-StructureV3")
    results = list(
        pipeline.predict(
            str(pdf_path),
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
        )
    )
    if save_vis:
        for res in results:
            res.save_to_img(str(output_dir))

    export_middle_bundle(
        results,
        output_dir=str(output_dir),
        min_ocr_coverage=min_ocr_coverage,
        pages_subdir=pages_subdir,
        char_source=char_source,
        save_vis=save_vis,
    )

