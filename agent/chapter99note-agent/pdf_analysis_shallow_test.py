#!/usr/bin/env python3
"""CLI test harness for shallow PDF note extraction."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import requests

from pdf_analysis_shallow import PdfNoteAnalyzerShallow
from pdf_analysis_shallow import CHAPTER_NUMBER

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_values
except ModuleNotFoundError:  # pragma: no cover - optional when verify is unused
    psycopg2 = None  # type: ignore[assignment]
    RealDictCursor = None  # type: ignore[assignment]
    execute_values = None  # type: ignore[assignment]


def _parse_note_list(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("note_list must be comma-separated numbers")
    notes: list[int] = []
    for part in parts:
        if not part.isdigit():
            raise argparse.ArgumentTypeError(
                f"invalid note number: {part} (expected integers)"
            )
        notes.append(int(part))
    return notes


def _make_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("/tmp/tariff-simulator")
    base_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"note99_{timestamp}_", dir=base_dir))


def _download_pdf(pdf_url: str, output_dir: Path) -> Path:
    output_path = output_dir / "chapter99.pdf"
    with requests.get(pdf_url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return output_path


def _normalize_content(value: str) -> str:
    value = value.replace("\r\n", "\n")
    return "\n".join(line.rstrip() for line in value.split("\n"))


def _normalize_parent_label(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


def _parse_path(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value)
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1]
        if not inner:
            return []
        return [part.strip() for part in inner.split(",")]
    return [text]


def _fetch_db_notes(note_numbers: list[int], database_dsn: str) -> list[dict]:
    if psycopg2 is None or RealDictCursor is None:  # pragma: no cover
        raise RuntimeError("psycopg2 is required to run --verify.")
    note_tokens = [str(num) for num in note_numbers]
    with psycopg2.connect(database_dsn) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT chapter, subchapter, label, path, parent_label, content
                FROM hts_notes
                WHERE chapter = %s AND path[2] = ANY(%s)
                ORDER BY subchapter, label, path
                """ ,
                (CHAPTER_NUMBER, note_tokens),
            )
            return list(cur.fetchall())


def _parse_path_array(value: str) -> list[str]:
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1]
        if not inner:
            return []
        return [part.strip() for part in inner.split(",")]
    return [value]


def _prepare_row_for_db(row: dict) -> dict:
    return {
        "chapter": int(row["chapter"]),
        "subchapter": row["subchapter"],
        "label": row["label"],
        "path": _parse_path_array(row["path"]),
        "parent_label": row["parent_label"],
        "content": row["content"],
        "raw_html": "",
    }


def _upsert_notes(rows: list[dict], database_dsn: str) -> int:
    if psycopg2 is None:  # pragma: no cover
        raise RuntimeError("psycopg2 is required to run --import-data.")
    if execute_values is None:  # pragma: no cover
        raise RuntimeError("psycopg2 execute_values is required to run --import-data.")
    if not rows:
        return 0
    prepared = [_prepare_row_for_db(row) for row in rows]
    with psycopg2.connect(database_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE hts_notes")
            values = [
                (
                    row["chapter"],
                    row["subchapter"],
                    row["label"],
                    row["path"],
                    row["parent_label"],
                    row["content"],
                    row["raw_html"],
                )
                for row in prepared
            ]
            insert_query = """
                INSERT INTO hts_notes (
                    chapter,
                    subchapter,
                    label,
                    path,
                    parent_label,
                    content,
                    raw_html
                )
                VALUES %s
            """
            execute_values(cur, insert_query, values, page_size=len(values))
    return len(prepared)


def _normalize_record(record: dict) -> dict:
    return {
        "chapter": int(record.get("chapter") or CHAPTER_NUMBER),
        "subchapter": record.get("subchapter"),
        "label": record.get("label"),
        "path": _parse_path(record.get("path")),
        "parent_label": _normalize_parent_label(record.get("parent_label")),
        "content": _normalize_content(record.get("content") or ""),
    }


def _compare_records(parsed: list[dict], db_rows: list[dict]) -> tuple[list, list, list]:
    parsed_map: dict[tuple[int, str | None, str], dict] = {}
    for row in parsed:
        normalized = _normalize_record(row)
        key = (normalized["chapter"], normalized["subchapter"], normalized["label"])
        parsed_map[key] = normalized

    db_map: dict[tuple[int, str | None, str], dict] = {}
    for row in db_rows:
        normalized = _normalize_record(row)
        key = (normalized["chapter"], normalized["subchapter"], normalized["label"])
        db_map[key] = normalized

    db_only = [key for key in db_map.keys() if key not in parsed_map]
    parsed_only = [key for key in parsed_map.keys() if key not in db_map]
    mismatched: list[tuple[tuple[int, str | None, str], dict[str, tuple[object, object]]]] = []

    for key in sorted(set(db_map.keys()) & set(parsed_map.keys())):
        db_row = db_map[key]
        parsed_row = parsed_map[key]
        diffs: dict[str, tuple[object, object]] = {}
        for field in ("subchapter", "label", "path", "parent_label", "content"):
            if db_row.get(field) != parsed_row.get(field):
                diffs[field] = (db_row.get(field), parsed_row.get(field))
        if diffs:
            mismatched.append((key, diffs))
    return db_only, parsed_only, mismatched


def _write_verify_report(
    output_path: Path,
    db_only: list,
    parsed_only: list,
    mismatched: list,
) -> None:
    lines: list[str] = []
    lines.append("SUMMARY")
    lines.append(f"db_only={len(db_only)}")
    lines.append(f"parsed_only={len(parsed_only)}")
    lines.append(f"mismatched={len(mismatched)}")
    lines.append("")

    if db_only:
        lines.append("DB_ONLY")
        for chapter, subchapter, label in db_only:
            lines.append(f"chapter={chapter} subchapter={subchapter} label={label}")
        lines.append("")

    if parsed_only:
        lines.append("PARSED_ONLY")
        for chapter, subchapter, label in parsed_only:
            lines.append(f"chapter={chapter} subchapter={subchapter} label={label}")
        lines.append("")

    if mismatched:
        lines.append("MISMATCHED")
        for (chapter, subchapter, label), diffs in mismatched:
            lines.append(f"chapter={chapter} subchapter={subchapter} label={label}")
            for field, (db_value, parsed_value) in diffs.items():
                if field == "content":
                    lines.append("field=content db_value:")
                    lines.append(db_value or "")
                    lines.append("field=content parsed_value:")
                    lines.append(parsed_value or "")
                else:
                    lines.append(
                        f"field={field} db_value={db_value!r} parsed_value={parsed_value!r}"
                    )
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract shallow notes from Chapter 99 PDF and print JSON."
    )
    parser.add_argument("--pdf_url", help="PDF URL to download.")
    parser.add_argument("--pdf_path", help="Absolute path to a local PDF file.")
    parser.add_argument(
        "--note_list",
        required=True,
        help="Comma-separated note numbers, e.g. 2,16,20",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare parsed output against hts_notes table.",
    )
    parser.add_argument(
        "--import-data",
        action="store_true",
        help="Upsert parsed output into hts_notes table.",
    )
    parser.add_argument(
        "--database-dsn",
        help="Postgres DSN for --verify/--import-data.",
    )
    args = parser.parse_args()

    if args.verify or args.import_data:
        if not args.database_dsn:
            raise ValueError("--database-dsn is required for database operations.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    notes = _parse_note_list(args.note_list)
    output_dir = _make_output_dir()

    if args.pdf_path:
        pdf_path = Path(args.pdf_path)
        if not pdf_path.is_absolute():
            raise ValueError("pdf_path must be an absolute path.")
    elif args.pdf_url:
        pdf_path = _download_pdf(args.pdf_url, output_dir)
    else:
        raise ValueError("pdf_url or pdf_path is required.")

    data = PdfNoteAnalyzerShallow.get_notes(None, str(pdf_path), notes)

    json_path = output_dir / "notes.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "notes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["chapter", "subchapter", "label", "path", "parent_label", "content"]
        )
        for row in data:
            writer.writerow(
                [
                    row.get("chapter"),
                    row.get("subchapter"),
                    row.get("label"),
                    row.get("path"),
                    row.get("parent_label"),
                    row.get("content"),
                ]
            )

    print(f"PDF_PATH={pdf_path}")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"JSON_OUTPUT={json_path}")
    print(f"CSV_OUTPUT={csv_path}")

    if args.verify:
        db_rows = _fetch_db_notes(notes, args.database_dsn)
        db_only, parsed_only, mismatched = _compare_records(data, db_rows)
        report_path = output_dir / "verify_report.txt"
        _write_verify_report(report_path, db_only, parsed_only, mismatched)
        print(
            f"VERIFY_SUMMARY=db_only={len(db_only)} parsed_only={len(parsed_only)} mismatched={len(mismatched)}"
        )
        print(f"VERIFY_REPORT={report_path}")

    if args.import_data:
        imported = _upsert_notes(data, args.database_dsn)
        print(f"IMPORT_COUNT={imported}")


if __name__ == "__main__":
    main()
