from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..utils import calculate_bbox_area, get_bbox_intersection


def ocr_coverage_in_block(ocr_box: np.ndarray, block_box: np.ndarray) -> float:
    ocr_a = float(calculate_bbox_area(ocr_box))
    if ocr_a <= 1e-6:
        return 0.0
    inter = get_bbox_intersection(ocr_box, block_box, return_format="bbox")
    if inter is None:
        return 0.0
    return float(calculate_bbox_area(inter)) / ocr_a


def assign_ocr_indices_exclusive_by_coverage(
    rec_boxes: np.ndarray, blocks: Sequence[Tuple[int, np.ndarray]], min_coverage: float
) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {block_idx: [] for block_idx, _ in blocks}
    for ocr_i in range(len(rec_boxes)):
        ocr_b = rec_boxes[ocr_i]
        best_cov = -1.0
        best_block = None
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
    layout_det_res: Any, overall_ocr_res: Any, min_coverage: float
) -> Dict[int, List[int]]:
    blocks: List[Tuple[int, np.ndarray]] = []
    for box_idx, box_info in enumerate(layout_det_res["boxes"]):
        if str(box_info["label"]).lower() in ("formula", "table", "seal"):
            continue
        blocks.append((box_idx, np.asarray(box_info["coordinate"], dtype=np.float64)))
    return assign_ocr_indices_exclusive_by_coverage(
        overall_ocr_res["rec_boxes"], blocks, min_coverage
    )


def split_spans_into_columns(spans: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not spans:
        return []
    spans_sorted = sorted(spans, key=lambda s: float(s["span_bbox_img"][0]))
    widths = [max(1.0, float(s["span_bbox_img"][2]) - float(s["span_bbox_img"][0])) for s in spans_sorted]
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
    return [c for c, _ in sorted(zip(columns, col_ranges), key=lambda pair: pair[1][0])]


def group_spans_to_lines_in_column(spans: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
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


def group_spans_to_lines(spans: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    lines_all: List[List[Dict[str, Any]]] = []
    for col in split_spans_into_columns(spans):
        lines_all.extend(group_spans_to_lines_in_column(col))
    return lines_all

