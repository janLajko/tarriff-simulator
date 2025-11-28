"""Section 301 duty computation utilities backed by the Postgres data model."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
import time
from typing import Dict, List, Optional, Tuple

try:  # pragma: no cover - optional dependency for environments without psycopg2
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:  # pragma: no cover - allow import without postgres libs
    psycopg2 = None  # type: ignore[assignment]
    RealDictCursor = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


@dataclass
class Section301Ch99:
    ch99_id: str
    alias: str
    general_rate: Decimal
    ch99_description: str
    amount: Decimal
    is_potential: bool = False


@dataclass
class Section301Computation:
    module_id: str
    module_name: str
    applicable: bool
    amount: Decimal
    currency: str
    rate: Optional[str]
    ch99_list: List[Section301Ch99] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _normalize_hts(code: str) -> str:
    return "".join(ch for ch in code if ch.isdigit())


def _code_matches(scope_code: str, hts_code: str) -> bool:
    scope_norm = _normalize_hts(scope_code)
    hts_norm = _normalize_hts(hts_code)
    if not scope_norm:
        return False
    if len(scope_norm) > len(hts_norm):
        return False
    return hts_norm.startswith(scope_norm)


def _date_active(
    entry: date, start: Optional[date], end: Optional[date]
) -> bool:
    if start and entry < start:
        return False
    if end and entry > end:
        return False
    return True


class Section301Evaluator:
    def __init__(self, conn, entry_date: date, country: str):
        self.conn = conn
        self.entry_date = entry_date
        self.country = country.upper()
        self.measures = self._load_measures()
        self.heading_to_measure: Dict[str, Dict] = {
            m["heading"]: m for m in self.measures
        }
        self.scope_cache: Dict[int, Dict[str, List[Dict]]] = {}
        self.match_cache: Dict[Tuple[int, str], Tuple[bool, List[str]]] = {}

    def _load_measures(self) -> List[Dict]:
        query = """
            SELECT
                m.id,
                m.heading,
                m.ad_valorem_rate,
                m.effective_start_date,
                m.effective_end_date,
                COALESCE(h.description, '') AS description
            FROM s301_measures AS m
            LEFT JOIN LATERAL (
                SELECT description
                FROM hts_codes hc
                WHERE hc.hts_number = m.heading
                ORDER BY hc.row_order
                LIMIT 1
            ) AS h ON TRUE
            WHERE (m.country_iso2 = %s OR m.country_iso2 IS NULL)
              AND m.effective_start_date <= %s
              AND (m.effective_end_date IS NULL OR %s <= m.effective_end_date)
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (self.country, self.entry_date, self.entry_date))
            rows = cur.fetchall()
        return rows

    def _load_scopes(self, measure_id: int) -> Dict[str, List[Dict]]:
        if measure_id in self.scope_cache:
            return self.scope_cache[measure_id]
        query = """
            SELECT
                map.relation,
                scope.key,
                scope.key_type,
                scope.country_iso2,
                scope.source_label,
                scope.effective_start_date,
                scope.effective_end_date
            FROM s301_scope_measure_map AS map
            JOIN s301_scope AS scope ON scope.id = map.scope_id
            WHERE map.measure_id = %s
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (measure_id,))
            rows = cur.fetchall()
        grouped: Dict[str, List[Dict]] = {"include": [], "exclude": []}
        for row in rows:
            if row["relation"] not in grouped:
                continue
            grouped[row["relation"]].append(row)
        self.scope_cache[measure_id] = grouped
        return grouped

    def _measure_id_for_heading(self, heading: str) -> Optional[int]:
        entry = self.heading_to_measure.get(heading)
        if entry:
            return entry["id"]
        query = """
            SELECT id, heading, ad_valorem_rate, effective_start_date, effective_end_date
            FROM s301_measures
            WHERE heading = %s
              AND (country_iso2 = %s OR country_iso2 IS NULL)
              AND effective_start_date <= %s
              AND (effective_end_date IS NULL OR %s <= effective_end_date)
            ORDER BY effective_start_date DESC, id DESC
            LIMIT 1
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                query,
                (heading, self.country, self.entry_date, self.entry_date),
            )
            row = cur.fetchone()
        if row:
            self.heading_to_measure[row["heading"]] = row
            if row not in self.measures:
                self.measures.append(row)
            return row["id"]
        return None

    def measure_covers(
        self, measure_id: int, hts_code: str, visited: Optional[set[int]] = None
    ) -> Tuple[bool, List[str]]:
        key = (measure_id, _normalize_hts(hts_code))
        if key in self.match_cache:
            return self.match_cache[key]

        if visited is None:
            visited = set()
        if measure_id in visited:
            self.match_cache[key] = False
            return False
        visited.add(measure_id)

        scopes = self._load_scopes(measure_id)
        includes = scopes.get("include", [])
        excludes = scopes.get("exclude", [])

        def matches_scope(scope_row: Dict, recurse: bool) -> bool:
            country = (scope_row.get("country_iso2") or "").upper()
            if country and country != self.country:
                return False
            if not _date_active(
                self.entry_date,
                scope_row.get("effective_start_date"),
                scope_row.get("effective_end_date"),
            ):
                return False
            key_value = scope_row["key"]
            if key_value.startswith("99"):
                if not recurse:
                    return False
                child_id = self._measure_id_for_heading(key_value)
                if not child_id:
                    return False
                child_match, _ = self.measure_covers(child_id, hts_code, visited.copy())
                return child_match
            return _code_matches(key_value, hts_code)

        include_hit = any(matches_scope(scope, True) for scope in includes)
        if not include_hit:
            self.match_cache[key] = (False, [])
            return False, []

        matched_exclusions: List[str] = []
        for scope in excludes:
            if matches_scope(scope, True):
                matched_exclusions.append(str(scope.get("key") or ""))

        if matched_exclusions:
            self.match_cache[key] = (False, matched_exclusions)
            return False, matched_exclusions

        self.match_cache[key] = (True, [])
        return True, []


def compute_section301_duty(
    hts_number: str,
    country_of_origin: str,
    entry_date: date,
    import_value: Optional[Decimal] = None,
) -> Section301Computation:
    """Compute Section 301 duty information for a given HTS number."""

    if not isinstance(entry_date, date):
        raise TypeError("entry_date must be a datetime.date instance")

    start_time = time.perf_counter()

    if psycopg2 is None:  # pragma: no cover - runtime enforcement
        raise RuntimeError(
            "psycopg2 is required to compute Section 301 duties. "
            "Install psycopg2-binary or psycopg2 and ensure it is available."
        )

    origin = (country_of_origin or "").upper()
    logger.info(
        "Section301 computation start hts=%s country=%s entry=%s",
        hts_number,
        origin,
        entry_date,
    )
    if origin != "CN":
        logger.info(
            "Section301 skipped for non-CN origin in %.3fs",
            time.perf_counter() - start_time,
        )
        return Section301Computation(
            module_id="301",
            module_name="Section 301 Tariffs",
            applicable=False,
            amount=Decimal("0"),
            currency="USD",
            rate=None,
            notes=["Section 301 tariffs currently target products of China only."],
        )

    dsn = os.getenv("DATABASE_DSN")
    if not dsn:
        raise RuntimeError("DATABASE_DSN environment variable is required for Section 301 calculations.")

    eval_start = time.perf_counter()
    with psycopg2.connect(dsn) as conn:
        evaluator = Section301Evaluator(conn, entry_date, origin)
        applicable_measures: List[Dict] = []
        excluded_measures: List[Tuple[Dict, List[str]]] = []
        for measure in evaluator.measures:
            covers, exclusions = evaluator.measure_covers(measure["id"], hts_number)
            if covers:
                applicable_measures.append(measure)
            elif exclusions:
                excluded_measures.append((measure, exclusions))
    logger.info(
        "Section301 evaluated measures in %.3fs (loaded=%s applicable=%s excluded=%s)",
        time.perf_counter() - eval_start,
        len(getattr(evaluator, "measures", []) or []),
        len(applicable_measures),
        len(excluded_measures),
    )

    rated_measures: List[Dict] = []
    zero_rate_matches: List[Dict] = []

    def _coerce_decimal(raw_value: Optional[object]) -> Optional[Decimal]:
        if raw_value is None:
            return None
        if isinstance(raw_value, Decimal):
            return raw_value
        return Decimal(str(raw_value))

    def _coerce_rate(raw_rate: Optional[object]) -> Optional[Decimal]:
        if raw_rate is None:
            return None
        return _coerce_decimal(raw_rate)

    import_value_decimal = _coerce_decimal(import_value)

    for measure in applicable_measures:
        rate = _coerce_rate(measure.get("ad_valorem_rate"))
        if rate is None or rate == 0:
            zero_rate_matches.append(measure)
            continue
        measure_with_rate = dict(measure)
        measure_with_rate["ad_valorem_rate"] = rate
        rated_measures.append(measure_with_rate)

    offset_heading_hits: set[str] = set()
    for measure, exclusions in excluded_measures:
        rate = _coerce_rate(measure.get("ad_valorem_rate"))
        if rate is None or rate == 0:
            zero_rate_matches.append(measure)
            continue

        direct_exclusions = [
            heading for heading in exclusions if heading and not heading.startswith("99")
        ]
        if direct_exclusions:
            # Direct HTS-level exclusion; do not surface paired entries.
            continue

        ref_exclusions = [
            heading for heading in exclusions if heading and heading.startswith("99")
        ]
        if not ref_exclusions:
            continue

        measure_with_rate = dict(measure)
        measure_with_rate["ad_valorem_rate"] = rate
        rated_measures.append(measure_with_rate)

        for heading in ref_exclusions:
            offset_heading_hits.add(heading)
            offset_measure = evaluator.heading_to_measure.get(heading, {})
            rated_measures.append(
                {
                    "heading": heading,
                    "alias": heading,
                    "ad_valorem_rate": -rate,
                    "description": offset_measure.get("description") or "",
                }
            )

    if not rated_measures:
        notes = [
            "No active Section 301 measures matched the provided HTS number on the given entry date."
        ]
        if zero_rate_matches:
            zero_headings = ", ".join(
                m["heading"]
                for m in zero_rate_matches
                if m.get("heading") not in offset_heading_hits
            )
            if zero_headings:
                notes.append(
                    "Section 301 headings matched but have zero ad valorem rate: "
                    f"{zero_headings}."
                )
        logger.info(
            "Section301 returning no rated measures in %.3fs",
            time.perf_counter() - start_time,
        )
        return Section301Computation(
            module_id="301",
            module_name="Section 301 Tariffs",
            applicable=False,
            amount=Decimal("0"),
            currency="USD",
            rate=None,
            notes=notes,
        )

    total_rate = sum((m["ad_valorem_rate"] for m in rated_measures), Decimal("0"))
    def _entry_amount(rate: Decimal) -> Decimal:
        if import_value_decimal is None:
            return Decimal("0")
        return (import_value_decimal * rate) / Decimal("100")

    ch99_list = [
        Section301Ch99(
            ch99_id=m["heading"],
            alias=m["heading"],
            general_rate=m["ad_valorem_rate"],
            ch99_description=m.get("description") or "",
            amount=_entry_amount(m["ad_valorem_rate"]),
        )
        for m in rated_measures
    ]

    normalized_total = total_rate.normalize()
    rate_display = f"{normalized_total}%"

    def _format_rate(rate: Decimal) -> str:
        normalized = rate.normalize()
        prefix = "+" if normalized > 0 else ""
        return f"{prefix}{normalized}%"

    applied_details = ", ".join(
        f"{m['heading']} ({_format_rate(m['ad_valorem_rate'])})" for m in rated_measures
    )
    notes = [
        "Applied Section 301 measures: " + applied_details
    ]
    if zero_rate_matches:
        zero_headings = ", ".join(
            m["heading"]
            for m in zero_rate_matches
            if m.get("heading") not in offset_heading_hits
        )
        if zero_headings:
            notes.append(
                "Additional Section 301 headings matched with zero rate: "
                + zero_headings
            )

    amount = Decimal("0")
    if import_value_decimal is not None:
        amount = (import_value_decimal * total_rate) / Decimal("100")

    logger.info(
        "Section301 computed duty in %.3fs applicable_measures=%s rated=%s",
        time.perf_counter() - start_time,
        len(applicable_measures),
        len(rated_measures),
    )
    return Section301Computation(
        module_id="301",
        module_name="Section 301 Tariffs",
        applicable=True,
        amount=amount,
        currency="USD",
        rate=rate_display,
        ch99_list=ch99_list,
        notes=notes,
    )
