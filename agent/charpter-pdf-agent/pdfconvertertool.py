#!/usr/bin/env python3
"""Batch convert chapter PDFs to text using pdfminer.six."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "charpter-data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "charpter-data-txt"


def _iter_pdfs(input_dir: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


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


def _bbox_intersects(bbox: tuple[float, float, float, float], crop: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    cx0, cy0, cx1, cy1 = crop
    return x1 > cx0 and x0 < cx1 and y1 > cy0 and y0 < cy1


def _sorted_page_text(
    page_layout: object,
    crop_box: Optional[tuple[float, float, float, float]],
) -> str:
    lines: list[tuple[tuple[float, float, float, float], str]] = []
    for line in _iter_text_lines(page_layout):
        if crop_box is not None and not _bbox_intersects(line.bbox, crop_box):
            continue
        text = line.get_text()
        if not text.strip():
            continue
        lines.append((line.bbox, text))

    lines.sort(key=lambda item: (-item[0][3], item[0][0]))
    return "".join(text for _, text in lines).strip()


def _load_crop_boxes(pdf_path: Path) -> list[tuple[float, float, float, float]]:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: install pypdf."
        ) from exc

    reader = PdfReader(str(pdf_path), strict=False)
    boxes: list[tuple[float, float, float, float]] = []
    for page in reader.pages:
        crop = page.cropbox or page.mediabox
        x0, y0 = crop.lower_left
        x1, y1 = crop.upper_right
        boxes.append((float(x0), float(y0), float(x1), float(y1)))
    return boxes


def _extract_pdf_text(
    pdf_path: Path,
    *,
    include_page_markers: bool,
    use_cropbox: bool,
) -> str:
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LAParams
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: install pdfminer.six."
        ) from exc

    crop_boxes = _load_crop_boxes(pdf_path) if use_cropbox else []

    laparams = LAParams(
        line_margin=0.2,
        char_margin=2.0,
        word_margin=0.1,
        boxes_flow=None,
    )

    parts: list[str] = []
    for page_index, page_layout in enumerate(
        extract_pages(str(pdf_path), laparams=laparams)
    ):
        crop_box = crop_boxes[page_index] if page_index < len(crop_boxes) else None
        page_text = _sorted_page_text(page_layout, crop_box)
        if include_page_markers:
            parts.append(f"--- Page {page_index + 1} ---\n{page_text}")
        else:
            parts.append(page_text)

    return "\n\n".join(parts).strip() + "\n"


def convert_pdfs(
    input_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    include_page_markers: bool = True,
    use_cropbox: bool = True,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdfs = _iter_pdfs(input_dir)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")

    converted = 0
    skipped = 0
    failed = 0

    for pdf_path in pdfs:
        output_path = output_dir / f"{pdf_path.stem}.txt"
        if output_path.exists() and not overwrite:
            print(f"Skip existing: {output_path}")
            skipped += 1
            continue

        try:
            text = _extract_pdf_text(
                pdf_path,
                include_page_markers=include_page_markers,
                use_cropbox=use_cropbox,
            )
        except Exception as exc:
            failed += 1
            print(f"Failed: {pdf_path.name} ({exc})")
            continue

        output_path.write_text(text, encoding="utf-8")
        converted += 1
        print(f"Converted: {pdf_path.name} -> {output_path.name}")

    print(
        f"Done. converted={converted} skipped={skipped} failed={failed} "
        f"input={input_dir} output={output_dir}"
    )
    return 0 if failed == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDFs in charpter-data to TXT using pdfminer.six."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing input PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for output TXT files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--no-page-markers",
        action="store_true",
        help="Do not include per-page markers in output.",
    )
    parser.add_argument(
        "--no-cropbox",
        action="store_true",
        help="Do not filter text outside the PDF crop box.",
    )
    args = parser.parse_args()

    exit_code = convert_pdfs(
        Path(args.input_dir),
        Path(args.output_dir),
        overwrite=args.overwrite,
        include_page_markers=not args.no_page_markers,
        use_cropbox=not args.no_cropbox,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
