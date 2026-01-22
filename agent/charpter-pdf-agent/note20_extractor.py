#!/usr/bin/env python3
"""Extract Note 20 with layout-aware section markers."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_INPUT_PDF = Path(__file__).resolve().parent / "charpter-data" / "SubchapterIII_USNote_20.pdf"
DEFAULT_OUTPUT_TXT = (
    Path(__file__).resolve().parent / "charpter-data-txt" / "SubchapterIII_USNote_20.normalized.txt"
)

HEADER_PATTERNS = [
    re.compile(r"^Harmonized Tariff Schedule of the United States", re.IGNORECASE),
    re.compile(r"^Annotated for Statistical Reporting Purposes$", re.IGNORECASE),
    re.compile(r"^U\\.S\\. Notes \\(con\\.\\)$", re.IGNORECASE),
    re.compile(r"^XXII$", re.IGNORECASE),
    re.compile(r"^99\\s*-\\s*III\\s*-\\s*\\d+$", re.IGNORECASE),
]

TOP_PATTERN = re.compile(r"^20\\.?\\s*\\(([a-z]{1,5})\\)$", re.IGNORECASE)
SUB_PATTERN = re.compile(r"^\\(([a-z]{1,5})\\)$", re.IGNORECASE)


@dataclass(frozen=True)
class TextLine:
    text: str
    norm_text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page: int


def _normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _iter_text_lines(layout_obj: object) -> Iterable[object]:
    try:
        from pdfminer.layout import LTTextLine
    except ModuleNotFoundError:
        return []

    if isinstance(layout_obj, LTTextLine):
        yield layout_obj
        return

    children = getattr(layout_obj, "_objs", None)
    if not children:
        return

    for child in children:
        yield from _iter_text_lines(child)


def _bbox_intersects(
    bbox: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
) -> bool:
    x0, y0, x1, y1 = bbox
    cx0, cy0, cx1, cy1 = crop
    return x1 > cx0 and x0 < cx1 and y1 > cy0 and y0 < cy1


def _load_crop_boxes(pdf_path: Path) -> list[tuple[float, float, float, float]]:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: install pypdf.") from exc

    reader = PdfReader(str(pdf_path), strict=False)
    boxes: list[tuple[float, float, float, float]] = []
    for page in reader.pages:
        crop = page.cropbox or page.mediabox
        x0, y0 = crop.lower_left
        x1, y1 = crop.upper_right
        boxes.append((float(x0), float(y0), float(x1), float(y1)))
    return boxes


def _extract_lines(pdf_path: Path, *, use_cropbox: bool) -> list[TextLine]:
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LAParams
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: install pdfminer.six.") from exc

    crop_boxes = _load_crop_boxes(pdf_path) if use_cropbox else []
    laparams = LAParams(
        line_margin=0.2,
        char_margin=2.0,
        word_margin=0.1,
        boxes_flow=None,
    )

    lines: list[TextLine] = []
    for page_index, page_layout in enumerate(
        extract_pages(str(pdf_path), laparams=laparams)
    ):
        crop_box = crop_boxes[page_index] if page_index < len(crop_boxes) else None
        for line in _iter_text_lines(page_layout):
            if crop_box is not None and not _bbox_intersects(line.bbox, crop_box):
                continue
            raw_text = line.get_text().strip()
            if not raw_text:
                continue
            norm_text = _normalize_text(raw_text)
            if not norm_text:
                continue
            x0, y0, x1, y1 = line.bbox
            lines.append(
                TextLine(
                    text=raw_text,
                    norm_text=norm_text,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    page=page_index + 1,
                )
            )

    return lines


def _is_header(text: str) -> bool:
    for pattern in HEADER_PATTERNS:
        if pattern.match(text):
            return True
    return False


def _cluster_indents(values: list[float], tolerance: float) -> list[float]:
    if not values:
        return []
    sorted_vals = sorted(values)
    clusters = [[sorted_vals[0]]]
    for value in sorted_vals[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _page_top_indents(
    lines: list[TextLine],
    *,
    indent_tolerance: float,
) -> tuple[dict[int, float], Optional[float]]:
    per_page: dict[int, list[float]] = {}
    for line in lines:
        if TOP_PATTERN.fullmatch(line.norm_text) or SUB_PATTERN.fullmatch(line.norm_text):
            per_page.setdefault(line.page, []).append(line.x0)

    page_top: dict[int, float] = {}
    for page, values in per_page.items():
        clusters = _cluster_indents(values, indent_tolerance)
        if clusters:
            page_top[page] = min(clusters)

    global_top = min(page_top.values()) if page_top else None
    return page_top, global_top


def _normalize_lines(
    lines: list[TextLine],
    *,
    indent_tolerance: float,
    include_page_markers: bool,
) -> list[str]:
    sorted_lines = sorted(lines, key=lambda line: (line.page, -line.y1, line.x0))
    page_top, global_top = _page_top_indents(sorted_lines, indent_tolerance=indent_tolerance)

    output: list[str] = []
    current_main: Optional[str] = None
    current_page: Optional[int] = None

    for line in sorted_lines:
        if _is_header(line.norm_text):
            continue

        if include_page_markers and current_page != line.page:
            output.append(f"--- Page {line.page} ---")
            current_page = line.page

        top_match = TOP_PATTERN.fullmatch(line.norm_text)
        sub_match = SUB_PATTERN.fullmatch(line.norm_text)

        if top_match:
            current_main = top_match.group(1).lower()
            output.append(f"20. ({current_main})")
            continue

        if sub_match:
            label = sub_match.group(1).lower()
            top_indent = page_top.get(line.page, global_top)
            if top_indent is not None and abs(line.x0 - top_indent) <= indent_tolerance:
                current_main = label
                output.append(f"20. ({label})")
            elif current_main:
                output.append(f"20({current_main})({label})")
            else:
                output.append(f"({label})")
            continue

        output.append(line.norm_text)

    return output


def extract_note20(
    pdf_path: Path,
    output_path: Path,
    *,
    indent_tolerance: float = 6.0,
    include_page_markers: bool = False,
    use_cropbox: bool = True,
) -> None:
    lines = _extract_lines(pdf_path, use_cropbox=use_cropbox)
    normalized = _normalize_lines(
        lines,
        indent_tolerance=indent_tolerance,
        include_page_markers=include_page_markers,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(normalized).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Note 20 content from PDF with normalized section markers."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PDF),
        help="Path to SubchapterIII_USNote_20.pdf.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_TXT),
        help="Output path for normalized Note 20 text.",
    )
    parser.add_argument(
        "--indent-tolerance",
        type=float,
        default=6.0,
        help="Tolerance for matching top-level indent.",
    )
    parser.add_argument(
        "--keep-page-markers",
        action="store_true",
        help="Include page markers in output.",
    )
    parser.add_argument(
        "--no-cropbox",
        action="store_true",
        help="Do not filter text outside the PDF crop box.",
    )
    args = parser.parse_args()

    extract_note20(
        Path(args.input),
        Path(args.output),
        indent_tolerance=args.indent_tolerance,
        include_page_markers=args.keep_page_markers,
        use_cropbox=not args.no_cropbox,
    )


if __name__ == "__main__":
    main()
