from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MiddleExportConfig:
    """Configuration for middle json export."""

    min_ocr_coverage: float = 0.7
    pages_subdir: str = "middle_pages"
    char_source: str = "pdf_text"
    save_vis: bool = False
    vis_subdir: str = "middle_vis"

