"""Section 301 duty computation utilities backed by the Postgres data model."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

try:  # pragma: no cover - optional dependency for environments without psycopg2
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:  # pragma: no cover - allow import without postgres libs
    psycopg2 = None  # type: ignore[assignment]
    RealDictCursor = None  # type: ignore[assignment]


@dataclass
class Section301Ch99:
    ch99_id: str
    alias: str
    general_rate: Decimal
    ch99_description: str


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
        self.match_cache: Dict[Tuple[int, str], bool] = {}

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
    ) -> bool:
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
                return self.measure_covers(child_id, hts_code, visited.copy())
            return _code_matches(key_value, hts_code)

        include_hit = any(matches_scope(scope, True) for scope in includes)
        if not include_hit:
            self.match_cache[key] = False
            return False

        for scope in excludes:
            if matches_scope(scope, True):
                self.match_cache[key] = False
                return False

        self.match_cache[key] = True
        return True


def compute_section301_duty(
    hts_number: str,
    country_of_origin: str,
    entry_date: date,
) -> Section301Computation:
    """Compute Section 301 duty information for a given HTS number."""

    if not isinstance(entry_date, date):
        raise TypeError("entry_date must be a datetime.date instance")

    if psycopg2 is None:  # pragma: no cover - runtime enforcement
        raise RuntimeError(
            "psycopg2 is required to compute Section 301 duties. "
            "Install psycopg2-binary or psycopg2 and ensure it is available."
        )

    origin = (country_of_origin or "").upper()
    if origin != "CN":
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

    with psycopg2.connect(dsn) as conn:
        evaluator = Section301Evaluator(conn, entry_date, origin)
        applicable_measures: List[Dict] = []
        for measure in evaluator.measures:
            if evaluator.measure_covers(measure["id"], hts_number):
                applicable_measures.append(measure)

    if not applicable_measures:
        return Section301Computation(
            module_id="301",
            module_name="Section 301 Tariffs",
            applicable=False,
            amount=Decimal("0"),
            currency="USD",
            rate=None,
            notes=[
                "No active Section 301 measures matched the provided HTS number on the given entry date."
            ],
        )

    total_rate = sum(
        ((m["ad_valorem_rate"] or Decimal("0")) for m in applicable_measures),
        Decimal("0"),
    )
    ch99_list = [
        Section301Ch99(
            ch99_id=m["heading"],
            alias=m["heading"],
            general_rate=m["ad_valorem_rate"],
            ch99_description=m.get("description") or "",
        )
        for m in applicable_measures
    ]

    rate_display = f"{total_rate.normalize()}%" if total_rate else None

    notes = [
        "Applied Section 301 measures: " + ", ".join(m["heading"] for m in applicable_measures)
    ]

    return Section301Computation(
        module_id="301",
        module_name="Section 301 Tariffs",
        applicable=True,
        amount=Decimal("0"),
        currency="USD",
        rate=rate_display,
        ch99_list=ch99_list,
        notes=notes,
    )
