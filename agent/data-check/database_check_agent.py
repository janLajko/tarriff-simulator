#!/usr/bin/env python3
"""Compare Postgres table data between two databases.

Primary use-case: diff `sieepa_measures` between two Postgres instances.

This script:
- Detects primary key columns automatically (or accepts `--key`).
- Streams `(<pk>, md5(to_jsonb(row)::text))` from both DBs in PK order.
- Reports missing keys (only in A / only in B) and mismatched rows (hash differs).
- Optionally fetches per-column diffs for a small sample of mismatches.

Credentials should be provided via env vars or CLI args; do not hardcode secrets.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Set

try:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extras import RealDictCursor
except ImportError as exc:  # pragma: no cover
    psycopg2 = None  # type: ignore
    sql = None  # type: ignore
    RealDictCursor = None  # type: ignore
    _PSYCOPG2_IMPORT_ERROR = exc
else:
    _PSYCOPG2_IMPORT_ERROR = None


LOGGER = logging.getLogger("database_check_agent")


DSN_PASSWORD_RE = re.compile(r"(?i)(password=)([^\\s]+)")


def redact_dsn(dsn: str) -> str:
    dsn = dsn.strip()
    if not dsn:
        return dsn

    if "://" in dsn:
        # postgresql://user:pass@host:port/db?...
        # Avoid importing urllib for edge cases; keep it minimal.
        # Replace user:pass@ -> user:***@
        return re.sub(r"(?<=://)([^:@/]+):([^@/]+)@", r"\1:***@", dsn)

    # libpq style: host=... dbname=... user=... password=...
    return DSN_PASSWORD_RE.sub(r"\1***", dsn)


def utc_now_z() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def require_psycopg2() -> None:
    if psycopg2 is None:
        raise SystemExit(
            f"psycopg2 is required to run this script: {_PSYCOPG2_IMPORT_ERROR}"
        )


def connect(dsn: str, application_name: str):
    require_psycopg2()
    assert psycopg2 is not None
    conn = psycopg2.connect(dsn, application_name=application_name)
    conn.autocommit = False
    return conn


def fetch_table_columns(conn, schema: str, table: str) -> List[str]:
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """
    with conn.cursor() as cur:
        cur.execute(query, (schema, table))
        rows = cur.fetchall()
    return [r[0] for r in rows]


def fetch_primary_key_columns(conn, schema: str, table: str) -> List[str]:
    query = """
        SELECT a.attname
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_attribute a
          ON a.attrelid = c.oid AND a.attnum = ANY(i.indkey)
        WHERE i.indisprimary = TRUE
          AND n.nspname = %s
          AND c.relname = %s
        ORDER BY array_position(i.indkey, a.attnum)
    """
    with conn.cursor() as cur:
        cur.execute(query, (schema, table))
        rows = cur.fetchall()
    return [r[0] for r in rows]


def validate_key_columns(all_columns: Sequence[str], key_columns: Sequence[str]) -> None:
    all_set = set(all_columns)
    missing = [c for c in key_columns if c not in all_set]
    if missing:
        raise SystemExit(
            f"Key columns not found in table columns: {missing}. Available: {list(all_columns)}"
        )


def _key_tuple(row: Sequence[Any], key_len: int) -> Tuple[Any, ...]:
    return tuple(row[:key_len])


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        # JSON/JSONB can come through as dict; normalize key order.
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    if isinstance(value, list):
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    return value


def diff_rows(row_a: Dict[str, Any], row_b: Dict[str, Any], columns: Sequence[str]) -> Dict[str, Any]:
    diffs: Dict[str, Any] = {}
    for col in columns:
        a_val = _normalize_value(row_a.get(col))
        b_val = _normalize_value(row_b.get(col))
        if a_val != b_val:
            diffs[col] = {"a": a_val, "b": b_val}
    return diffs


def guess_heading_column(columns: Sequence[str]) -> Optional[str]:
    preferred = [
        "heading",
        "hts_heading",
        "hts_heading_code",
        "hts_code",
        "hs_code",
        "measure_heading",
    ]
    cols_lower = {c.lower(): c for c in columns}
    for p in preferred:
        if p in cols_lower:
            return cols_lower[p]
    for c in columns:
        cl = c.lower()
        if "heading" in cl:
            return c
    for c in columns:
        cl = c.lower()
        if cl in ("hts", "htscode") or ("hts" in cl and "code" in cl):
            return c
    return None


def _build_rowhash_query(
    schema: str,
    table: str,
    key_columns: Sequence[str],
    where_sql: Optional[str],
) -> "sql.Composed":
    assert sql is not None
    identifiers = [sql.Identifier(c) for c in key_columns]
    select_keys = sql.SQL(", ").join(identifiers)
    order_keys = sql.SQL(", ").join(identifiers)
    where_clause = sql.SQL("")
    if where_sql:
        where_clause = sql.SQL(" WHERE ") + sql.SQL(where_sql)

    # Important: `.format()` exists on `sql.SQL`, not on `sql.Composed`.
    # Build a single template and format once.
    template = sql.SQL(
        "SELECT {keys}, md5(to_jsonb(t)::text) AS row_hash "
        "FROM {schema}.{table} AS t"
        "{where_clause}"
        " ORDER BY {order_keys}"
    )
    return template.format(
        keys=select_keys,
        schema=sql.Identifier(schema),
        table=sql.Identifier(table),
        where_clause=where_clause,
        order_keys=order_keys,
    )


def stream_row_hashes(
    conn,
    schema: str,
    table: str,
    key_columns: Sequence[str],
    where_sql: Optional[str],
    fetch_size: int = 2000,
) -> Iterator[Tuple[Tuple[Any, ...], str]]:
    assert sql is not None
    key_len = len(key_columns)
    query = _build_rowhash_query(schema, table, key_columns, where_sql)
    cursor_name = f"rowhash_{os.getpid()}_{id(conn)}"
    cur = conn.cursor(name=cursor_name)
    cur.itersize = fetch_size
    cur.execute(query)
    try:
        for row in cur:
            key = _key_tuple(row, key_len)
            row_hash = row[key_len]
            yield key, row_hash
    finally:
        cur.close()


def fetch_row_by_key(
    conn,
    schema: str,
    table: str,
    key_columns: Sequence[str],
    key: Tuple[Any, ...],
):
    assert sql is not None
    assert RealDictCursor is not None
    conditions = sql.SQL(" AND ").join(
        sql.SQL("{} = %s").format(sql.Identifier(col)) for col in key_columns
    )
    # Build a single SQL template; don't call `.format()` on `sql.Composed`.
    query = sql.SQL(
        "SELECT * FROM {schema}.{table} WHERE {conditions} LIMIT 1"
    ).format(
        schema=sql.Identifier(schema),
        table=sql.Identifier(table),
        conditions=conditions,
    )
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, list(key))
        return cur.fetchone()


@dataclass
class TableDiffReport:
    schema: str
    table: str
    key_columns: List[str]
    compared_matching_keys: int
    mismatched_rows: int
    only_in_a: int
    only_in_b: int
    sample_only_in_a_keys: List[List[Any]]
    sample_only_in_b_keys: List[List[Any]]
    sample_mismatched: List[Dict[str, Any]]
    dsn_a: str
    dsn_b: str
    generated_at: str
    partial: bool = False


@dataclass
class HeadingCountryRateDiffReport:
    schema: str
    table: str
    heading_column: str
    country_column: str
    rate_column: str
    headings_in_a: int
    headings_in_b: int
    common_headings: int
    only_in_a_headings: int
    only_in_b_headings: int
    diff_headings: int
    sample_only_in_a_headings: List[Any]
    sample_only_in_b_headings: List[Any]
    sample_diff_headings: List[Dict[str, Any]]
    dsn_a: str
    dsn_b: str
    generated_at: str
    partial: bool = False


def fetch_heading_country_rate_map(
    conn,
    schema: str,
    table: str,
    heading_column: str,
    country_column: str,
    rate_column: str,
    where_sql: Optional[str],
    fetch_size: int = 2000,
) -> Dict[Any, Set[Tuple[Any, Any]]]:
    assert sql is not None
    where_clause = sql.SQL("")
    if where_sql:
        where_clause = sql.SQL(" WHERE ") + sql.SQL(where_sql)

    query = sql.SQL(
        "SELECT {heading} AS heading, {country} AS country_iso2, {rate} AS ad_valorem_rate "
        "FROM {schema}.{table}"
        "{where_clause}"
    ).format(
        heading=sql.Identifier(heading_column),
        country=sql.Identifier(country_column),
        rate=sql.Identifier(rate_column),
        schema=sql.Identifier(schema),
        table=sql.Identifier(table),
        where_clause=where_clause,
    )

    cursor_name = f"heading_rate_{os.getpid()}_{id(conn)}"
    cur = conn.cursor(name=cursor_name)
    cur.itersize = fetch_size
    mapping: Dict[Any, Set[Tuple[Any, Any]]] = {}
    cur.execute(query)
    try:
        for heading, country_iso2, ad_valorem_rate in cur:
            heading_key = heading
            rate_norm = _normalize_value(ad_valorem_rate)
            country_norm = _normalize_value(country_iso2)
            mapping.setdefault(heading_key, set()).add((country_norm, rate_norm))
    finally:
        cur.close()
    return mapping


def compare_heading_country_rate(
    conn_a,
    conn_b,
    schema: str,
    table: str,
    heading_column: str,
    country_column: str,
    rate_column: str,
    where_sql: Optional[str],
    fetch_size: int,
    sample_limit: int,
    stop_after_differences: Optional[int],
) -> HeadingCountryRateDiffReport:
    cols_a = fetch_table_columns(conn_a, schema, table)
    cols_b = fetch_table_columns(conn_b, schema, table)
    if set(cols_a) != set(cols_b):
        raise SystemExit(
            "Table columns differ between DBs; refuse to compare.\n"
            f"A columns({len(cols_a)}): {cols_a}\n"
            f"B columns({len(cols_b)}): {cols_b}"
        )

    for required in (heading_column, country_column, rate_column):
        if required not in cols_a:
            raise SystemExit(f"Column {required!r} not found in {schema}.{table}.")

    map_a = fetch_heading_country_rate_map(
        conn_a,
        schema=schema,
        table=table,
        heading_column=heading_column,
        country_column=country_column,
        rate_column=rate_column,
        where_sql=where_sql,
        fetch_size=fetch_size,
    )
    map_b = fetch_heading_country_rate_map(
        conn_b,
        schema=schema,
        table=table,
        heading_column=heading_column,
        country_column=country_column,
        rate_column=rate_column,
        where_sql=where_sql,
        fetch_size=fetch_size,
    )

    headings_a = set(map_a.keys())
    headings_b = set(map_b.keys())
    only_a = sorted(headings_a - headings_b)
    only_b = sorted(headings_b - headings_a)
    common = sorted(headings_a & headings_b)

    diff_headings = 0
    sample_diffs: List[Dict[str, Any]] = []
    partial = False

    def _inc_and_maybe_stop() -> bool:
        nonlocal diff_headings, partial
        diff_headings += 1
        if stop_after_differences is None:
            return False
        if diff_headings >= stop_after_differences:
            partial = True
            return True
        return False

    for heading in common:
        pairs_a = map_a.get(heading, set())
        pairs_b = map_b.get(heading, set())
        if pairs_a == pairs_b:
            continue
        if _inc_and_maybe_stop():
            break
        if len(sample_diffs) < sample_limit:
            sample_diffs.append(
                {
                    "heading": _normalize_value(heading),
                    "only_in_a_pairs": sorted(
                        [{"country_iso2": c, "ad_valorem_rate": r} for c, r in (pairs_a - pairs_b)],
                        key=lambda x: (str(x["country_iso2"]), str(x["ad_valorem_rate"])),
                    ),
                    "only_in_b_pairs": sorted(
                        [{"country_iso2": c, "ad_valorem_rate": r} for c, r in (pairs_b - pairs_a)],
                        key=lambda x: (str(x["country_iso2"]), str(x["ad_valorem_rate"])),
                    ),
                }
            )

    return HeadingCountryRateDiffReport(
        schema=schema,
        table=table,
        heading_column=heading_column,
        country_column=country_column,
        rate_column=rate_column,
        headings_in_a=len(headings_a),
        headings_in_b=len(headings_b),
        common_headings=len(common),
        only_in_a_headings=len(only_a),
        only_in_b_headings=len(only_b),
        diff_headings=diff_headings,
        sample_only_in_a_headings=[_normalize_value(h) for h in only_a[:sample_limit]],
        sample_only_in_b_headings=[_normalize_value(h) for h in only_b[:sample_limit]],
        sample_diff_headings=sample_diffs,
        dsn_a="",
        dsn_b="",
        generated_at=utc_now_z(),
        partial=partial,
    )


def compare_table(
    conn_a,
    conn_b,
    schema: str,
    table: str,
    key_columns: Sequence[str],
    where_sql: Optional[str],
    fetch_size: int,
    sample_limit: int,
    mismatch_detail_limit: int,
    include_rows_in_mismatch_sample: bool,
    stop_after_differences: Optional[int],
) -> TableDiffReport:
    columns_a = fetch_table_columns(conn_a, schema, table)
    columns_b = fetch_table_columns(conn_b, schema, table)
    if set(columns_a) != set(columns_b):
        raise SystemExit(
            "Table columns differ between DBs; refuse to compare.\n"
            f"A columns({len(columns_a)}): {columns_a}\n"
            f"B columns({len(columns_b)}): {columns_b}"
        )
    if columns_a != columns_b:
        LOGGER.warning("Column order differs between DBs; using DB A column order for diffs.")
    validate_key_columns(columns_a, key_columns)

    gen_a = stream_row_hashes(conn_a, schema, table, key_columns, where_sql, fetch_size=fetch_size)
    gen_b = stream_row_hashes(conn_b, schema, table, key_columns, where_sql, fetch_size=fetch_size)

    compared_matching = 0
    mismatched = 0
    only_in_a = 0
    only_in_b = 0
    sample_only_in_a: List[List[Any]] = []
    sample_only_in_b: List[List[Any]] = []
    sample_mismatched: List[Dict[str, Any]] = []
    partial = False

    def _maybe_stop() -> bool:
        if stop_after_differences is None:
            return False
        return (only_in_a + only_in_b + mismatched) >= stop_after_differences

    a_next = next(gen_a, None)
    b_next = next(gen_b, None)
    while a_next is not None or b_next is not None:
        if _maybe_stop():
            partial = True
            break

        if a_next is None:
            # Remaining keys only in B
            key_b, _hash_b = b_next  # type: ignore[misc]
            only_in_b += 1
            if len(sample_only_in_b) < sample_limit:
                sample_only_in_b.append(list(key_b))
            b_next = next(gen_b, None)
            continue

        if b_next is None:
            # Remaining keys only in A
            key_a, _hash_a = a_next
            only_in_a += 1
            if len(sample_only_in_a) < sample_limit:
                sample_only_in_a.append(list(key_a))
            a_next = next(gen_a, None)
            continue

        key_a, hash_a = a_next
        key_b, hash_b = b_next

        if key_a < key_b:
            only_in_a += 1
            if len(sample_only_in_a) < sample_limit:
                sample_only_in_a.append(list(key_a))
            a_next = next(gen_a, None)
            continue

        if key_b < key_a:
            only_in_b += 1
            if len(sample_only_in_b) < sample_limit:
                sample_only_in_b.append(list(key_b))
            b_next = next(gen_b, None)
            continue

        # Same key
        compared_matching += 1
        if hash_a != hash_b:
            mismatched += 1
            if len(sample_mismatched) < mismatch_detail_limit:
                entry: Dict[str, Any] = {
                    "key": list(key_a),
                    "hash_a": hash_a,
                    "hash_b": hash_b,
                }
                row_a = fetch_row_by_key(conn_a, schema, table, key_columns, key_a)
                row_b = fetch_row_by_key(conn_b, schema, table, key_columns, key_a)
                if row_a is None or row_b is None:
                    entry["row_fetch_error"] = "missing row during detail fetch"
                else:
                    entry["column_diffs"] = diff_rows(row_a, row_b, columns_a)
                    if include_rows_in_mismatch_sample:
                        entry["row_a"] = {k: _normalize_value(v) for k, v in row_a.items()}
                        entry["row_b"] = {k: _normalize_value(v) for k, v in row_b.items()}
                sample_mismatched.append(entry)
        a_next = next(gen_a, None)
        b_next = next(gen_b, None)

    return TableDiffReport(
        schema=schema,
        table=table,
        key_columns=list(key_columns),
        compared_matching_keys=compared_matching,
        mismatched_rows=mismatched,
        only_in_a=only_in_a,
        only_in_b=only_in_b,
        sample_only_in_a_keys=sample_only_in_a,
        sample_only_in_b_keys=sample_only_in_b,
        sample_mismatched=sample_mismatched,
        dsn_a="",
        dsn_b="",
        generated_at=utc_now_z(),
        partial=partial,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diff a Postgres table between two databases.")
    parser.add_argument("--dsn-a", default=os.environ.get("DB_A_DSN"), help="Database A DSN (or env DB_A_DSN).")
    parser.add_argument("--dsn-b", default=os.environ.get("DB_B_DSN"), help="Database B DSN (or env DB_B_DSN).")
    parser.add_argument("--schema", default="public", help="Table schema (default: public).")
    parser.add_argument("--table", default="sieepa_measures", help="Table name (default: sieepa_measures).")
    parser.add_argument(
        "--mode",
        default="rowhash",
        choices=["rowhash", "heading-rate"],
        help="Compare mode: rowhash (default) or heading-rate.",
    )
    parser.add_argument(
        "--key",
        action="append",
        default=None,
        help="Primary key column (repeatable). If omitted, auto-detect PK from DB A.",
    )
    parser.add_argument(
        "--heading-column",
        default=None,
        help="Heading column name for --mode heading-rate (auto-guess if omitted).",
    )
    parser.add_argument(
        "--country-column",
        default="country_iso2",
        help="Country column name for --mode heading-rate (default: country_iso2).",
    )
    parser.add_argument(
        "--rate-column",
        default="ad_valorem_rate",
        help="Rate column name for --mode heading-rate (default: ad_valorem_rate).",
    )
    parser.add_argument(
        "--where",
        default=None,
        help="Optional raw SQL WHERE clause (without 'WHERE'). Use carefully.",
    )
    parser.add_argument("--fetch-size", type=int, default=2000, help="Server-side cursor fetch size.")
    parser.add_argument("--sample-limit", type=int, default=20, help="Sample keys to include in report.")
    parser.add_argument("--mismatch-detail-limit", type=int, default=20, help="Max mismatched rows to detail.")
    parser.add_argument(
        "--include-rows",
        action="store_true",
        help="Include full row data for mismatches in the JSON report (can be large).",
    )
    parser.add_argument(
        "--stop-after-differences",
        type=int,
        default=None,
        help="Stop early after N total differences (only_in_a + only_in_b + mismatched).",
    )
    parser.add_argument("--report", default=None, help="Write JSON report to this file path.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, ...).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.dsn_a or not args.dsn_b:
        raise SystemExit("Provide --dsn-a/--dsn-b or set DB_A_DSN/DB_B_DSN.")

    dsn_a_redacted = redact_dsn(args.dsn_a)
    dsn_b_redacted = redact_dsn(args.dsn_b)
    LOGGER.info("Connecting A: %s", dsn_a_redacted)
    LOGGER.info("Connecting B: %s", dsn_b_redacted)

    conn_a = connect(args.dsn_a, application_name="database_check_agent_a")
    conn_b = connect(args.dsn_b, application_name="database_check_agent_b")
    try:
        if args.mode == "heading-rate":
            cols = fetch_table_columns(conn_a, args.schema, args.table)
            heading_column = args.heading_column or guess_heading_column(cols)
            if not heading_column:
                raise SystemExit(
                    f"Unable to auto-detect heading column for {args.schema}.{args.table}; pass --heading-column."
                )
            report = compare_heading_country_rate(
                conn_a=conn_a,
                conn_b=conn_b,
                schema=args.schema,
                table=args.table,
                heading_column=heading_column,
                country_column=args.country_column,
                rate_column=args.rate_column,
                where_sql=args.where,
                fetch_size=int(args.fetch_size),
                sample_limit=int(args.sample_limit),
                stop_after_differences=args.stop_after_differences,
            )
            report.dsn_a = dsn_a_redacted
            report.dsn_b = dsn_b_redacted
            LOGGER.info(
                "Done. headings_in_a=%s headings_in_b=%s common_headings=%s diff_headings=%s only_in_a_headings=%s only_in_b_headings=%s partial=%s",
                report.headings_in_a,
                report.headings_in_b,
                report.common_headings,
                report.diff_headings,
                report.only_in_a_headings,
                report.only_in_b_headings,
                report.partial,
            )
            if args.report:
                with open(args.report, "w", encoding="utf-8") as f:
                    json.dump(asdict(report), f, ensure_ascii=False, indent=2, default=str)
                LOGGER.info("Wrote report: %s", args.report)
            else:
                print(
                    json.dumps(
                        {
                            "schema": report.schema,
                            "table": report.table,
                            "heading_column": report.heading_column,
                            "country_column": report.country_column,
                            "rate_column": report.rate_column,
                            "headings_in_a": report.headings_in_a,
                            "headings_in_b": report.headings_in_b,
                            "common_headings": report.common_headings,
                            "diff_headings": report.diff_headings,
                            "only_in_a_headings": report.only_in_a_headings,
                            "only_in_b_headings": report.only_in_b_headings,
                            "partial": report.partial,
                            "generated_at": report.generated_at,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
        else:
            if args.key:
                key_columns = args.key
            else:
                key_columns = fetch_primary_key_columns(conn_a, args.schema, args.table)
                if not key_columns:
                    raise SystemExit(
                        f"No primary key found for {args.schema}.{args.table}; please pass --key <col> (repeatable)."
                    )
            LOGGER.info("Using key columns: %s", key_columns)
            report = compare_table(
                conn_a=conn_a,
                conn_b=conn_b,
                schema=args.schema,
                table=args.table,
                key_columns=key_columns,
                where_sql=args.where,
                fetch_size=int(args.fetch_size),
                sample_limit=int(args.sample_limit),
                mismatch_detail_limit=int(args.mismatch_detail_limit),
                include_rows_in_mismatch_sample=bool(args.include_rows),
                stop_after_differences=args.stop_after_differences,
            )
            report.dsn_a = dsn_a_redacted
            report.dsn_b = dsn_b_redacted
            LOGGER.info(
                "Done. compared_matching_keys=%s mismatched_rows=%s only_in_a=%s only_in_b=%s partial=%s",
                report.compared_matching_keys,
                report.mismatched_rows,
                report.only_in_a,
                report.only_in_b,
                report.partial,
            )
            if args.report:
                with open(args.report, "w", encoding="utf-8") as f:
                    json.dump(asdict(report), f, ensure_ascii=False, indent=2, default=str)
                LOGGER.info("Wrote report: %s", args.report)
            else:
                print(
                    json.dumps(
                        {
                            "schema": report.schema,
                            "table": report.table,
                            "key_columns": report.key_columns,
                            "compared_matching_keys": report.compared_matching_keys,
                            "mismatched_rows": report.mismatched_rows,
                            "only_in_a": report.only_in_a,
                            "only_in_b": report.only_in_b,
                            "partial": report.partial,
                            "generated_at": report.generated_at,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
        return 0
    finally:
        try:
            conn_a.close()
        finally:
            conn_b.close()


if __name__ == "__main__":
    raise SystemExit(main())
