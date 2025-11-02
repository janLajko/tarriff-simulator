"""HTS CSV ingestion utility.

Reads all CSV files in the HTS data directory, fixes known data issues, and
loads the normalized rows into PostgreSQL.

Fixes implemented:
1. Pads missing leading zeros in the first segment of HTS numbers (e.g. 410 → 0410).
2. Inherits ``General Rate of Duty`` and ``Column 2 Rate of Duty`` from the
   nearest ancestor when the current row leaves them blank.
3. Persists rows even when the HTS number itself is blank to preserve the
   hierarchy implied by indentation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:  # pragma: no cover - allows importing without psycopg2 installed
    psycopg2 = None  # type: ignore
    execute_values = None  # type: ignore

try:
    import requests
except ImportError:  # pragma: no cover - allows importing without requests installed
    requests = None  # type: ignore


HTS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS hts_codes (
    id SERIAL PRIMARY KEY,
    hts_number TEXT,
    indent INTEGER NOT NULL,
    description TEXT,
    unit_of_quantity TEXT,
    general_rate_of_duty TEXT,
    special_rate_of_duty TEXT,
    column_2_rate_of_duty TEXT,
    quota_quantity TEXT,
    additional_duties TEXT,
    status TEXT,
    parent_hts_number TEXT,
    row_order INTEGER NOT NULL,
    parent_row_order INTEGER
);
"""

HTS_STATUS_ENDPOINT = "https://hts.usitc.gov/reststop/getRates"
HTS_STATUS_ENDPOINT_99 = "https://hts.usitc.gov/reststop/getRates99"
_HTS_STATUS_CACHE: Dict[str, Dict[str, Optional[str]]] = {}


def _clean_status(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _derive_chapter_key(hts_number: str) -> Optional[str]:
    digits = "".join(ch for ch in hts_number if ch.isdigit())
    if not digits:
        return None
    return digits[:2] if len(digits) >= 2 else digits


def _fetch_status_payload(endpoint: str, params: Dict[str, str]) -> Any:
    query = urlencode(params)
    url = f"{endpoint}?{query}"
    headers = {
        "User-Agent": "tariff-simulate-v2/hts-importer (+https://hts.usitc.gov/)",
        "Accept": "application/json",
    }

    if requests is not None:
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network error handling
            raise RuntimeError(f"Failed to fetch HTS status data from {endpoint}: {exc}") from exc
        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Invalid JSON received from {endpoint}") from exc

    try:
        request = Request(url, headers=headers, method="GET")
        with urlopen(request, timeout=30) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            body = response.read().decode(charset)
    except (HTTPError, URLError, TimeoutError) as exc:  # pragma: no cover - network error handling
        raise RuntimeError(f"Failed to fetch HTS status data from {endpoint}: {exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Invalid JSON received from {endpoint}") from exc


def _collect_status_entries(payload: Any) -> Dict[str, Optional[str]]:
    status_map: Dict[str, Optional[str]] = {}

    def _visit(node: Any) -> None:
        if isinstance(node, dict):
            htsno = node.get("htsno")
            if isinstance(htsno, str):
                normalized = normalize_hts_number(htsno)
                if normalized:
                    if "status" in node:
                        status_map[normalized] = _clean_status(node.get("status"))
            for value in node.values():
                _visit(value)
        elif isinstance(node, list):
            for item in node:
                _visit(item)

    _visit(payload)
    return status_map


def _load_chapter_statuses(chapter_key: str, hts_number: str) -> Dict[str, Optional[str]]:
    endpoint = HTS_STATUS_ENDPOINT_99 if chapter_key == "99" else HTS_STATUS_ENDPOINT
    payload = _fetch_status_payload(endpoint, {"htsno": hts_number, "keyword": hts_number})
    statuses = _collect_status_entries(payload)
    _HTS_STATUS_CACHE[chapter_key] = statuses
    return statuses


def get_hts_status(hts_number: Optional[str]) -> Optional[str]:
    if not hts_number:
        return None
    chapter_key = _derive_chapter_key(hts_number)
    if not chapter_key:
        return None

    statuses = _HTS_STATUS_CACHE.get(chapter_key)
    if statuses is None:
        statuses = _load_chapter_statuses(chapter_key, hts_number)

    if hts_number in statuses:
        return statuses[hts_number]

    statuses = _load_chapter_statuses(chapter_key, hts_number)
    return statuses.get(hts_number)


@dataclass
class NormalizedRow:
    """Represents a normalized HTS CSV row ready for database insertion."""

    row_order: int
    parent_row_order: Optional[int]
    hts_number: Optional[str]
    indent: int
    description: Optional[str]
    unit_of_quantity: Optional[str]
    general_rate_of_duty: Optional[str]
    special_rate_of_duty: Optional[str]
    column_2_rate_of_duty: Optional[str]
    quota_quantity: Optional[str]
    additional_duties: Optional[str]
    status: Optional[str]

    def as_tuple(self) -> tuple:
        """Return row fields ordered for INSERT."""
        return (
            self.hts_number,
            self.indent,
            self.description,
            self.unit_of_quantity,
            self.general_rate_of_duty,
            self.special_rate_of_duty,
            self.column_2_rate_of_duty,
            self.quota_quantity,
            self.additional_duties,
            self.status,
            None,  # parent_hts_number placeholder, set after insert
            self.row_order,
            self.parent_row_order,
        )


def normalize_hts_number(raw_value: str) -> Optional[str]:
    """Normalize an HTS number, padding the first segment to four digits when needed."""
    if raw_value is None:
        return None
    value = raw_value.strip()
    if not value:
        return None
    parts = value.replace("\u3000", " ").split(".")
    head = parts[0].strip()
    if head.isdigit() and len(head) < 4:
        head = head.zfill(4)
    parts[0] = head
    normalized_parts = [segment.strip() for segment in parts if segment is not None]
    return ".".join(normalized_parts) if normalized_parts else None


def clean_field(value: Optional[str]) -> Optional[str]:
    """Convert empty CSV fields to None while trimming whitespace."""
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def parse_indent(value: Optional[str], *, line_number: int, source: Path) -> int:
    """Parse the indent level, defaulting blanks to zero."""
    if value is None or value.strip() == "":
        return 0
    try:
        indent = int(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid indent '{value}' at {source}:{line_number}") from exc
    if indent < 0:
        raise ValueError(f"Negative indent '{value}' at {source}:{line_number}")
    return indent


def iter_csv_rows(csv_path: Path) -> Iterable[dict]:
    """Yield dictionaries for each row of the CSV file."""
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def normalize_rows(csv_paths: Sequence[Path]) -> List[NormalizedRow]:
    """Normalize raw CSV rows, applying hierarchy-aware fixes."""
    normalized: List[NormalizedRow] = []
    hierarchy_stack: List[dict] = []
    line_counter = 0

    for csv_path in sorted(csv_paths):
        for row in iter_csv_rows(csv_path):
            line_counter += 1
            indent = parse_indent(row.get("Indent"), line_number=line_counter, source=csv_path)
            target_indent = indent

            while len(hierarchy_stack) > target_indent:
                hierarchy_stack.pop()

            if target_indent > len(hierarchy_stack):
                if not hierarchy_stack:
                    target_indent = len(hierarchy_stack)
                else:
                    filler = hierarchy_stack[-1]
                    while len(hierarchy_stack) < target_indent:
                        hierarchy_stack.append(
                            {
                                "row_order": filler.get("row_order"),
                                "hts_number": filler.get("hts_number"),
                                "general_rate_of_duty": filler.get("general_rate_of_duty"),
                                "column_2_rate_of_duty": filler.get("column_2_rate_of_duty"),
                            }
                        )

            parent_context = hierarchy_stack[-1] if hierarchy_stack else None

            while len(hierarchy_stack) < target_indent:
                hierarchy_stack.append(parent_context.copy() if parent_context else {})

            general_rate = clean_field(row.get("General Rate of Duty"))
            column2_rate = clean_field(row.get("Column 2 Rate of Duty"))

            if not general_rate:
                for ancestor in reversed(hierarchy_stack):
                    if ancestor["general_rate_of_duty"]:
                        general_rate = ancestor["general_rate_of_duty"]
                        break
            if not column2_rate:
                for ancestor in reversed(hierarchy_stack):
                    if ancestor["column_2_rate_of_duty"]:
                        column2_rate = ancestor["column_2_rate_of_duty"]
                        break

            normalized_hts = normalize_hts_number(row.get("HTS Number"))

            normalized_row = NormalizedRow(
                row_order=len(normalized) + 1,
                parent_row_order=parent_context["row_order"] if parent_context else None,
                hts_number=normalized_hts,
                indent=indent,
                description=clean_field(row.get("Description")),
                unit_of_quantity=clean_field(row.get("Unit of Quantity")),
                general_rate_of_duty=general_rate,
                special_rate_of_duty=clean_field(row.get("Special Rate of Duty")),
                column_2_rate_of_duty=column2_rate,
                quota_quantity=clean_field(row.get("Quota Quantity")),
                additional_duties=clean_field(row.get("Additional Duties")),
                status=get_hts_status(normalized_hts),
            )

            normalized.append(normalized_row)

            current_context = {
                "row_order": normalized_row.row_order,
                "hts_number": normalized_row.hts_number,
                "general_rate_of_duty": normalized_row.general_rate_of_duty,
                "column_2_rate_of_duty": normalized_row.column_2_rate_of_duty,
            }
            hierarchy_stack.append(current_context)

    return normalized


def ensure_table(conn) -> None:
    """Ensure the destination table exists."""
    with conn.cursor() as cur:
        cur.execute(HTS_TABLE_DDL)
        cur.execute(
            "ALTER TABLE hts_codes ADD COLUMN IF NOT EXISTS row_order INTEGER"
        )
        cur.execute(
            "ALTER TABLE hts_codes ADD COLUMN IF NOT EXISTS parent_row_order INTEGER"
        )
        cur.execute(
            "ALTER TABLE hts_codes ADD COLUMN IF NOT EXISTS status TEXT"
        )


def truncate_table(conn) -> None:
    """Remove existing HTS rows before loading fresh data."""
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE hts_codes RESTART IDENTITY;")


def insert_rows(conn, rows: Sequence[NormalizedRow]) -> None:
    """Bulk insert normalized rows."""
    if not rows:
        return
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO hts_codes (
                hts_number,
                indent,
                description,
                unit_of_quantity,
                general_rate_of_duty,
                special_rate_of_duty,
                column_2_rate_of_duty,
                quota_quantity,
                additional_duties,
                status,
                parent_hts_number,
                row_order,
                parent_row_order
            )
            VALUES %s
            """,
            [row.as_tuple() for row in rows],
        )


def assign_parent_ids(conn) -> None:
    """Replace stored parent HTS numbers with the parent row's numeric ID."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE hts_codes AS child
            SET parent_hts_number = parent.id::text
            FROM hts_codes AS parent
            WHERE child.parent_row_order IS NOT NULL
              AND parent.row_order = child.parent_row_order
        """
        )


def discover_csv_paths(data_dir: Path) -> List[Path]:
    """Return all CSV files within the data directory."""
    return sorted(p for p in data_dir.glob("*.csv") if p.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import HTS CSV data into PostgreSQL.")
    parser.add_argument(
        "--dsn",
        default=os.getenv("DATABASE_DSN"),
        help="PostgreSQL DSN (e.g. postgresql://user:pass@localhost:5432/dbname). "
        "Defaults to DATABASE_DSN environment variable.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing HTS CSV files (default: this script's directory).",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Append to existing data instead of truncating the table first.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dsn:
        raise SystemExit("PostgreSQL DSN required via --dsn or DATABASE_DSN.")
    if psycopg2 is None or execute_values is None:  # pragma: no cover - defensive
        raise SystemExit("psycopg2 is required to run this script. Install via `pip install psycopg2-binary`.")

    csv_paths = discover_csv_paths(args.data_dir)
    if not csv_paths:
        raise SystemExit(f"No CSV files found in {args.data_dir}")

    normalized_rows = normalize_rows(csv_paths)

    with psycopg2.connect(args.dsn) as conn:
        conn.autocommit = False
        ensure_table(conn)
        if not args.keep_existing:
            truncate_table(conn)
        insert_rows(conn, normalized_rows)
        assign_parent_ids(conn)


if __name__ == "__main__":
    main()
