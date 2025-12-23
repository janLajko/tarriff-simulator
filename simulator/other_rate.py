"""Other Chapter 99 note duty computation utilities backed by Postgres."""

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

EU_MEMBER_CODES = {
    "AT",
    "BE",
    "BG",
    "HR",
    "CY",
    "CZ",
    "DK",
    "EE",
    "FI",
    "FR",
    "DE",
    "GR",
    "HU",
    "IE",
    "IT",
    "LV",
    "LT",
    "LU",
    "MT",
    "NL",
    "PL",
    "PT",
    "RO",
    "SK",
    "SI",
    "ES",
    "SE",
}


@dataclass
class OtherCh99:
    ch99_id: str
    alias: str
    general_rate: Decimal
    ch99_description: str
    amount: Decimal
    is_potential: bool = False


@dataclass
class OtherComputation:
    module_id: str
    module_name: str
    applicable: bool
    amount: Decimal
    currency: str
    rate: Optional[str]
    ch99_list: List[OtherCh99] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


NOTE_MODULES = {
    33: ("note33", "Chapter 99 Note 33 Tariffs"),
    36: ("note36", "Chapter 99 Note 36 Tariffs"),
    37: ("note37", "Chapter 99 Note 37 Tariffs"),
    38: ("note38", "Chapter 99 Note 38 Tariffs"),
}


def _normalize_hts(code: str) -> str:
    return "".join(ch for ch in code if ch.isdigit())


def _normalize_country(value: str) -> str:
    cleaned = (value or "").strip().upper()
    if cleaned in EU_MEMBER_CODES:
        return "EU"
    return cleaned


def _code_matches(scope_code: str, hts_code: str) -> bool:
    scope_norm = _normalize_hts(scope_code)
    hts_norm = _normalize_hts(hts_code)
    if not scope_norm:
        return False
    if len(scope_norm) > len(hts_norm):
        return False
    return hts_norm.startswith(scope_norm)


def _date_active(entry: date, start: Optional[date], end: Optional[date]) -> bool:
    if start and entry < start:
        return False
    if end and entry > end:
        return False
    return True


def _coerce_decimal(value: Optional[object]) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _format_rate(rate: Decimal) -> str:
    normalized = rate.normalize()
    prefix = "+" if normalized > 0 else ""
    return f"{prefix}{normalized}%"


class OtherNoteEvaluator:
    def __init__(self, conn, entry_date: date, origin: str, date_of_landing: Optional[date]):
        self.conn = conn
        self.entry_date = entry_date
        self.origin_raw = (origin or "").strip().upper()
        self.origin = _normalize_country(origin)
        # self.note_number = note_number
        self.date_of_landing = date_of_landing
        self.measures = self._load_measures()
        self.heading_to_measure: Dict[str, Dict] = {m["heading"]: m for m in self.measures}
        self.heading_to_measure_norm: Dict[str, Dict] = {
            _normalize_hts(m["heading"]): m for m in self.measures if m.get("heading")
        }
        self.scope_cache: Dict[int, Dict[str, List[Dict]]] = {}
        self.match_cache: Dict[Tuple[int, str], Tuple[bool, List[str]]] = {}

    def _load_measures(self) -> List[Dict]:
        query = """
            SELECT
                m.id,
                m.heading,
                m.country_iso2,
                m.ad_valorem_rate,
                m.value_basis,
                m.origin_exclude_iso2,
                m.notes,
                m.effective_start_date,
                m.effective_end_date,
                COALESCE(m.is_potential, false) AS is_potential,
                COALESCE(h.description, '') AS description
            FROM otherch_measures AS m
            LEFT JOIN LATERAL (
                SELECT description
                FROM hts_codes hc
                WHERE hc.hts_number = m.heading
                ORDER BY hc.row_order
                LIMIT 1
            ) AS h ON TRUE
            WHERE
               m.effective_start_date <= %s
              AND (m.effective_end_date IS NULL OR %s <= m.effective_end_date)
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                query,
                (
                    # str(self.note_number),
                    self.entry_date,
                    self.entry_date,
                ),
            )
            return cur.fetchall()

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
            FROM otherch_scope_measure_map AS map
            JOIN otherch_scope AS scope ON scope.id = map.scope_id
            WHERE map.measure_id = %s
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (measure_id,))
            rows = cur.fetchall()
        grouped: Dict[str, List[Dict]] = {"include": [], "exclude": []}
        for row in rows:
            relation = row.get("relation")
            if relation not in grouped:
                continue
            grouped[relation].append(row)
        self.scope_cache[measure_id] = grouped
        return grouped

    def _scope_allows_country(self, scope: Dict) -> bool:
        scope_country = (scope.get("country_iso2") or "").upper()
        if not scope_country:
            return True
        if scope_country == "EU":
            return self.origin == "EU"
        if scope_country in EU_MEMBER_CODES:
            return self.origin_raw == scope_country
        return self.origin_raw == scope_country

    def _constraints_met(self, measure: Dict) -> Tuple[bool, Optional[str]]:
        if str(measure.get("heading") or "") == "9903.94.03" or str(measure.get("heading") or "") == "9903.94.06":
            if self.origin_raw not in {"CA", "MX"}:
                return False, f"Origin {self.origin_raw} not eligible for 9903.94.03"
        measure_country = (measure.get("country_iso2") or "").upper()
        if measure_country:
            if measure_country == "EU":
                if self.origin != "EU":
                    return False, f"Origin {self.origin_raw} not in EU"
            elif measure_country in EU_MEMBER_CODES:
                if self.origin_raw != measure_country:
                    return False, f"Origin {self.origin_raw} != {measure_country}"
            elif self.origin_raw != measure_country:
                return False, f"Origin {self.origin_raw} != {measure_country}"
        excluded = measure.get("origin_exclude_iso2") or []
        if excluded:
            excluded_norm = {str(item).strip().upper() for item in excluded if item}
            if self.origin_raw in excluded_norm:
                return False, f"Origin {self.origin_raw} excluded"
            if self.origin == "EU" and "EU" in excluded_norm:
                return False, "Origin EU excluded"
        notes = measure.get("notes") or {}
        entry_after = notes.get("entry_date_on_or_after")
        if entry_after:
            try:
                cutoff = date.fromisoformat(entry_after)
                if self.entry_date < cutoff:
                    return False, f"Entry date before {cutoff.isoformat()}"
            except Exception:
                pass
        entry_before = notes.get("entry_date_before")
        if entry_before:
            try:
                cutoff = date.fromisoformat(entry_before)
                if self.entry_date > cutoff:
                    return False, f"Entry date after {cutoff.isoformat()}"
            except Exception:
                pass
        loading_before = notes.get("date_of_loading_before")
        if loading_before and self.date_of_landing:
            try:
                cutoff = date.fromisoformat(loading_before)
                if self.date_of_landing > cutoff:
                    return False, f"Loading date after {cutoff.isoformat()}"
            except Exception:
                pass
        return True, None

    def measure_covers(self, measure_id: int, hts_code: str) -> Tuple[bool, List[str]]:
        key = (measure_id, hts_code)
        if key in self.match_cache:
            return self.match_cache[key]
        scopes = self._load_scopes(measure_id)
        includes = scopes.get("include") or []
        excludes = scopes.get("exclude") or []

        def matches_scope(scope: Dict) -> bool:
            if scope.get("key_type") == "note":
                return False
            if not _date_active(self.entry_date, scope.get("effective_start_date"), scope.get("effective_end_date")):
                return False
            if not self._scope_allows_country(scope):
                return False
            return _code_matches(scope.get("key") or "", hts_code)

        include_hit = any(matches_scope(scope) for scope in includes)
        if not include_hit:
            self.match_cache[key] = (False, [])
            return False, []

        matched_exclusions: List[str] = []
        for scope in excludes:
            if matches_scope(scope):
                matched_exclusions.append(str(scope.get("key") or ""))

        if matched_exclusions:
            self.match_cache[key] = (False, matched_exclusions)
            return False, matched_exclusions

        self.match_cache[key] = (True, [])
        return True, []


def compute_note_duty(
    note_number: int,
    hts_number: str,
    country_of_origin: str,
    entry_date: date,
    date_of_landing: Optional[date],
    import_value: Optional[Decimal],
    copper_percentage: Optional[Decimal] = None,
    base_rate_decimal: Optional[Decimal] = None,
) -> OtherComputation:
    if not isinstance(entry_date, date):
        raise TypeError("entry_date must be a datetime.date instance")
    if date_of_landing is not None and not isinstance(date_of_landing, date):
        raise TypeError("date_of_landing must be a datetime.date instance")

    start_time = time.perf_counter()

    if psycopg2 is None:  # pragma: no cover - runtime enforcement
        raise RuntimeError(
            "psycopg2 is required to compute Chapter 99 note duties. "
            "Install psycopg2-binary or psycopg2 and ensure it is available."
        )

    dsn = os.getenv("DATABASE_DSN")
    if not dsn:
        raise RuntimeError("DATABASE_DSN environment variable is required for Chapter 99 note calculations.")

    module_id, module_name = NOTE_MODULES[note_number]
    origin = _normalize_country(country_of_origin)

    logger.info(
        "Other note computation start note=%s hts=%s country=%s entry=%s",
        note_number,
        hts_number,
        origin,
        entry_date,
    )

    with psycopg2.connect(dsn) as conn:
        evaluator = OtherNoteEvaluator(conn, entry_date, origin, date_of_landing)
        applicable_measures: List[Dict] = []
        excluded_measures: List[Tuple[Dict, List[str]]] = []
        constraint_notes: List[str] = []
        for measure in evaluator.measures:
            allowed, reason = evaluator._constraints_met(measure)
            if not allowed:
                if reason:
                    constraint_notes.append(f"{measure['heading']}: {reason}")
                continue
            covers, exclusions = evaluator.measure_covers(measure["id"], hts_number)
            if covers:
                applicable_measures.append(measure)
            elif exclusions:
                excluded_measures.append((measure, exclusions))

        if applicable_measures:
            heading_set = {
                _normalize_hts(measure.get("heading") or "")
                for measure in applicable_measures
                if measure.get("heading")
            }

            def _exclude_by_heading(scope: Dict, active_headings: set[str]) -> bool:
                if scope.get("key_type") != "heading":
                    return False
                if not _date_active(
                    entry_date, scope.get("effective_start_date"), scope.get("effective_end_date")
                ):
                    return False
                if not evaluator._scope_allows_country(scope):
                    return False
                scope_heading = _normalize_hts(scope.get("key") or "")
                if scope_heading not in active_headings:
                    return False
                scope_measure = evaluator.heading_to_measure_norm.get(scope_heading)
                # print(scope_measure)
                if scope_measure and scope_measure.get("is_potential"):
                    return False
                return True

            filtered_measures: List[Dict] = []
            for measure in applicable_measures:
                measure_heading = _normalize_hts(measure.get("heading") or "")
                candidate_headings = heading_set - {measure_heading}
                if not candidate_headings:
                    filtered_measures.append(measure)
                    continue
                scopes = evaluator._load_scopes(measure["id"])
                excludes = scopes.get("exclude") or []
                if any(_exclude_by_heading(scope, candidate_headings) for scope in excludes):
                    continue
                filtered_measures.append(measure)

            applicable_measures = filtered_measures

    rated_measures: List[Dict] = []
    zero_rate_matches: List[Dict] = []
    seen_headings: set[str] = set()

    import_value_decimal = _coerce_decimal(import_value)
    copper_percentage_decimal = _coerce_decimal(copper_percentage)

    def _measure_base_value(heading: str) -> Optional[Decimal]:
        if heading == "9903.78.01":
            if import_value_decimal is None or copper_percentage_decimal is None:
                return None
            return import_value_decimal * copper_percentage_decimal
        return import_value_decimal

    for measure in applicable_measures:
        rate = _coerce_decimal(measure.get("ad_valorem_rate"))
        heading = (measure.get("heading") or "").strip()
        if base_rate_decimal is not None and heading in {
            "9903.94.40",
            "9903.94.42",
            "9903.94.44",
            "9903.94.50",
            "9903.94.52",
            "9903.94.54",
            "9903.94.60",
            "9903.94.62",
            "9903.94.64",
        }:
            if base_rate_decimal < Decimal("15"):
                continue
        if base_rate_decimal is not None and heading in {
            "9903.94.41",
            "9903.94.43",
            "9903.94.45",
            "9903.94.51",
            "9903.94.53",
            "9903.94.55",
            "9903.94.61",
            "9903.94.63",
            "9903.94.65",
        }:
            if base_rate_decimal > Decimal("15"):
                continue
            adjusted = Decimal("15") - base_rate_decimal
            if adjusted < 0:
                adjusted = Decimal("0")
            rate = adjusted
        if rate is None or rate == 0:
            zero_rate_matches.append(measure)
            continue
        if not heading or heading in seen_headings:
            continue
        measure_with_rate = dict(measure)
        measure_with_rate["ad_valorem_rate"] = rate
        rated_measures.append(measure_with_rate)
        seen_headings.add(heading)

    if not rated_measures and not zero_rate_matches:
        notes = [
            f"No active Chapter 99 note {note_number} measures matched the provided HTS number on the given entry date."
        ]
        if constraint_notes:
            notes.extend(constraint_notes)
        logger.info(
            "Other note %s returning no matched measures in %.3fs",
            note_number,
            time.perf_counter() - start_time,
        )
        return OtherComputation(
            module_id=module_id,
            module_name=module_name,
            applicable=False,
            amount=Decimal("0"),
            currency="USD",
            rate=None,
            notes=notes,
        )

    total_rate = sum((m["ad_valorem_rate"] for m in rated_measures), Decimal("0"))

    def _entry_amount(rate: Decimal, heading: str) -> Decimal:
        base_value = _measure_base_value(heading)
        if base_value is None:
            return Decimal("0")
        return (base_value * rate) / Decimal("100")

    amount = sum(
        (_entry_amount(m["ad_valorem_rate"], m.get("heading") or "") for m in rated_measures),
        Decimal("0"),
    )

    display_measures: List[Dict] = []
    seen_display: set[str] = set()
    for measure in rated_measures + zero_rate_matches:
        heading = measure.get("heading")
        if not heading or heading in seen_display:
            continue
        display_measures.append(measure)
        seen_display.add(heading)

    ch99_list = [
        OtherCh99(
            ch99_id=m["heading"],
            alias=m["heading"],
            general_rate=_coerce_decimal(m.get("ad_valorem_rate")) or Decimal("0"),
            ch99_description=m.get("description") or "",
            amount=_entry_amount(
                _coerce_decimal(m.get("ad_valorem_rate")) or Decimal("0"),
                m.get("heading") or "",
            ),
            is_potential=bool(m.get("is_potential")),
        )
        for m in display_measures
    ]

    applied_details = ", ".join(
        f"{m['heading']} ({_format_rate(_coerce_decimal(m.get('ad_valorem_rate')) or Decimal('0'))})"
        for m in display_measures
    )
    notes = [f"Applied Chapter 99 note {note_number} measures: {applied_details}"]
    if constraint_notes:
        notes.extend(constraint_notes)
    if zero_rate_matches:
        zero_headings = ", ".join(m["heading"] for m in zero_rate_matches if m.get("heading"))
        if zero_headings:
            notes.append("Additional matched headings with zero rate: " + zero_headings)

    rate_display = f"{total_rate.normalize()}%"

    logger.info(
        "Other note %s computed duty in %.3fs applicable_measures=%s rated=%s",
        note_number,
        time.perf_counter() - start_time,
        len(applicable_measures),
        len(rated_measures),
    )
    return OtherComputation(
        module_id=module_id,
        module_name=module_name,
        applicable=True,
        amount=amount,
        currency="USD",
        rate=rate_display,
        ch99_list=ch99_list,
        notes=notes,
    )
