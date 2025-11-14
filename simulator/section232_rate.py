"""Section 232 duty computation utilities backed by the Postgres data model."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency for environments without psycopg2
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:  # pragma: no cover - allow import without postgres libs
    psycopg2 = None  # type: ignore[assignment]
    RealDictCursor = None  # type: ignore[assignment]

from .basic_hts_rate import get_unit_config


@dataclass
class Section232Ch99:
    ch99_id: str
    alias: str
    general_rate: Decimal
    ch99_description: str
    amount: Decimal = Decimal("0")
    is_potential: bool = False
    # amount: Decimal


@dataclass
class Section232Computation:
    module_id: str
    module_name: str
    applicable: bool
    amount: Decimal
    currency: str
    rate: Optional[str]
    ch99_list: List[Section232Ch99] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


HUNDRED = Decimal("100")
STEEL_PREFIXES = ("9903.81", "9908.81")
ALUMINUM_PREFIXES = ("9903.85", "9908.85")
STEEL_CH73_HEADINGS = {
    "9903.81.87",
    "9903.81.88",
    "9903.81.89",
    "9903.81.90",
    "9903.81.93",
    "9903.81.94",
    "9903.81.95",
    "9903.81.96",
    "9903.81.97",
    "9903.81.99",
}
STEEL_ALWAYS_HEADINGS = {
    "9903.81.91",
    "9903.81.98",
}
ALUMINUM_CH76_HEADINGS = {
    "9903.85.02",
    "9903.85.04",
    "9903.85.07",
    "9903.85.12",
    "9903.85.13",
    "9903.85.14",
}
ALUMINUM_ALWAYS_HEADINGS = {
    "9903.85.08",
    "9903.85.15",
}


def _coerce_decimal(value: object) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _normalize_measurement_map(
    measurements: Optional[Mapping[str, object]],
) -> Dict[str, Decimal]:
    normalized: Dict[str, Decimal] = {}
    if not measurements:
        return normalized
    for raw_key, raw_value in measurements.items():
        key = str(raw_key or "").strip().lower()
        if not key:
            continue
        value = _coerce_decimal(raw_value)
        if value is None:
            continue
        normalized[key] = value
    return normalized


def _classify_component(heading: str) -> str:
    normalized = (heading or "").strip()
    for prefix in STEEL_PREFIXES:
        if normalized.startswith(prefix):
            return "steel"
    for prefix in ALUMINUM_PREFIXES:
        if normalized.startswith(prefix):
            return "aluminum"
    return "general"


def _resolve_component_share(
    component: str,
    config: Mapping[str, object],
    measurements: Mapping[str, Decimal],
) -> Tuple[Optional[Decimal], Optional[str]]:
    field_name = f"{component}_percentage"
    measurement_value = measurements.get(field_name)
    fallback_key = config.get(field_name)
    if measurement_value is None and isinstance(fallback_key, str):
        candidate_key = fallback_key.strip().lower()
        if candidate_key and candidate_key not in {"kg", "kgs", "kilogram", "kilograms"}:
            measurement_value = measurements.get(candidate_key)

    if measurement_value is None:
        if field_name in config:
            return None, field_name
        return Decimal("1"), None

    share = measurement_value
    if share > 1 and share <= 100:
        share = share / HUNDRED
    return share, None


def _determine_entry_share(
    hts_number: str,
    ch99_id: str,
    component: str,
    shares: Mapping[str, Optional[Decimal]],
) -> Tuple[Optional[Decimal], Optional[str]]:
    chapter = _hts_chapter(hts_number)
    normalized_ch99 = (ch99_id or "").strip()

    def _component_share(name: str) -> Tuple[Optional[Decimal], Optional[str]]:
        return shares.get(name), name

    if normalized_ch99 in STEEL_CH73_HEADINGS:
        if chapter == "73":
            return _component_share("steel")
        return Decimal("1"), None

    if normalized_ch99 in STEEL_ALWAYS_HEADINGS:
        return _component_share("steel")

    if normalized_ch99 in ALUMINUM_CH76_HEADINGS:
        if chapter == "76":
            return _component_share("aluminum")
        return Decimal("1"), None

    if normalized_ch99 in ALUMINUM_ALWAYS_HEADINGS:
        return _component_share("aluminum")

    if component == "steel":
        return _component_share("steel")
    if component == "aluminum":
        return _component_share("aluminum")
    return Decimal("1"), None


def _compute_section232_amount(
    hts_number: str,
    entries: Sequence[Section232Ch99],
    import_value: Optional[Decimal],
    measurements: Mapping[str, Decimal],
) -> Tuple[Decimal, List[str]]:
    if not entries:
        return Decimal("0"), []
    if import_value is None:
        for entry in entries:
            entry.amount = Decimal("0")
        return Decimal("0"), []

    config = get_unit_config(hts_number) or {}
    shares: Dict[str, Optional[Decimal]] = {}
    component_missing_fields: Dict[str, str] = {}
    for component in ("steel", "aluminum"):
        share_value, missing_field = _resolve_component_share(component, config, measurements)
        shares[component] = share_value
        if missing_field:
            component_missing_fields[component] = missing_field

    amount = Decimal("0")
    missing_fields: set[str] = set()
    for entry in entries:
        component = _classify_component(entry.ch99_id)
        share, required_component = _determine_entry_share(
            hts_number,
            entry.ch99_id,
            component,
            shares,
        )

        if share is None:
            if required_component:
                missing_field = component_missing_fields.get(required_component) or f"{required_component}_percentage"
                missing_fields.add(missing_field)
            entry.amount = Decimal("0")
            continue

        entry_amount = (import_value * share * entry.general_rate) / HUNDRED
        entry.amount = entry_amount
        amount += entry_amount

    return amount, sorted(missing_fields)


def _normalize_hts(code: str) -> str:
    return "".join(ch for ch in code if ch.isdigit())


def _hts_chapter(hts_code: str) -> Optional[str]:
    normalized = _normalize_hts(hts_code)
    if len(normalized) < 2:
        return None
    return normalized[:2]


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


class Section232Evaluator:
    def __init__(
        self,
        conn,
        entry_date: date,
        country: str,
        melt_pour_origin: Optional[str],
    ) -> None:
        self.conn = conn
        self.entry_date = entry_date
        self.country = (country or "").upper()
        self.melt_origin = (melt_pour_origin or "").upper() or None
        self.measures = self._load_measures()
        self.heading_to_measure: Dict[str, Dict] = {
            m["heading"]: m for m in self.measures
        }
        self.scope_cache: Dict[int, Dict[str, List[Dict]]] = {}
        self.match_cache: Dict[Tuple[int, str], Tuple[bool, List[str]]] = {}

    def _country_allows_measure(self, measure: Dict) -> bool:
        target_country = (measure.get("country_iso2") or "").upper()
        if target_country:
            return self.country == target_country
        excluded: Sequence[str] = measure.get("origin_exclude_iso2") or []
        excluded_upper = { (code or "").upper() for code in excluded }
        if not self.country:
            # If country is unknown, only allow global measures with no exclusions.
            return not excluded_upper
        return self.country not in excluded_upper

    def _load_measures(self) -> List[Dict]:
        query = """
            SELECT
                m.id,
                m.heading,
                m.country_iso2,
                m.melt_pour_origin_iso2,
                m.origin_exclude_iso2,
                m.ad_valorem_rate,
                m.effective_start_date,
                m.effective_end_date,
                m.is_potential,
                COALESCE(h.description, '') AS description
            FROM s232_measures AS m
            LEFT JOIN LATERAL (
                SELECT description
                FROM hts_codes hc
                WHERE hc.hts_number = m.heading
                ORDER BY hc.row_order
                LIMIT 1
            ) AS h ON TRUE
            WHERE m.effective_start_date <= %s
              AND (m.effective_end_date IS NULL OR %s <= m.effective_end_date)
              AND (m.melt_pour_origin_iso2 IS NULL OR m.melt_pour_origin_iso2 = %s)
        """
        params = (self.entry_date, self.entry_date, self.melt_origin)
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        filtered = [row for row in rows if self._country_allows_measure(row)]
        return filtered

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
            FROM s232_scope_measure_map AS map
            JOIN s232_scope AS scope ON scope.id = map.scope_id
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
            SELECT
                id,
                heading,
                country_iso2,
                melt_pour_origin_iso2,
                origin_exclude_iso2,
                ad_valorem_rate,
                effective_start_date,
                effective_end_date,
                is_potential
            FROM s232_measures
            WHERE heading = %s
              AND (melt_pour_origin_iso2 IS NULL OR melt_pour_origin_iso2 = %s)
              AND effective_start_date <= %s
              AND (effective_end_date IS NULL OR %s <= effective_end_date)
            ORDER BY effective_start_date DESC, id DESC
            LIMIT 1
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                query,
                (heading, self.melt_origin, self.entry_date, self.entry_date),
            )
            row = cur.fetchone()
        if row and self._country_allows_measure(row):
            self.heading_to_measure[row["heading"]] = row
            if row not in self.measures:
                self.measures.append(row)
            return row["id"]
        return None

    def measure_covers(
        self,
        measure_id: int,
        hts_code: str,
        visited: Optional[set[int]] = None,
    ) -> Tuple[bool, List[str]]:
        key = (measure_id, _normalize_hts(hts_code))
        if key in self.match_cache:
            return self.match_cache[key]

        if visited is None:
            visited = set()
        if measure_id in visited:
            self.match_cache[key] = (False, [])
            return False, []
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


def compute_section232_duty(
    hts_number: str,
    country_of_origin: str,
    entry_date: date,
    melt_pour_origin_iso2: Optional[str] = None,
    import_value: Optional[Decimal] = None,
    measurements: Optional[Mapping[str, Decimal]] = None,
    steel_percentage: Optional[Decimal] = None,
    aluminum_percentage: Optional[Decimal] = None,
) -> Section232Computation:
    """Compute Section 232 duty information for a given HTS number."""

    if not isinstance(entry_date, date):
        raise TypeError("entry_date must be a datetime.date instance")

    if psycopg2 is None:  # pragma: no cover - runtime enforcement
        raise RuntimeError(
            "psycopg2 is required to compute Section 232 duties. "
            "Install psycopg2-binary or psycopg2 and ensure it is available."
        )

    origin = (country_of_origin or "").upper()
    melt_origin = (melt_pour_origin_iso2 or "").upper() or None
    import_value_decimal = _coerce_decimal(import_value)
    normalized_measurements = _normalize_measurement_map(measurements)
    steel_share = _coerce_decimal(steel_percentage)
    if steel_share is not None:
        normalized_measurements["steel_percentage"] = steel_share
    aluminum_share = _coerce_decimal(aluminum_percentage)
    if aluminum_share is not None:
        normalized_measurements["aluminum_percentage"] = aluminum_share

    dsn = os.getenv("DATABASE_DSN")
    if not dsn:
        raise RuntimeError("DATABASE_DSN environment variable is required for Section 232 calculations.")

    with psycopg2.connect(dsn) as conn:
        evaluator = Section232Evaluator(conn, entry_date, origin, melt_origin)
        applicable_measures: List[Dict] = []
        excluded_measures: List[Tuple[Dict, List[str]]] = []
        for measure in evaluator.measures:
            covers, exclusions = evaluator.measure_covers(measure["id"], hts_number)
            if covers:
                applicable_measures.append(measure)
            elif exclusions:
                excluded_measures.append((measure, exclusions))

    rated_measures: List[Dict] = []
    zero_rate_matches: List[Dict] = []
    special_zero_measure: Optional[Dict] = None
    SPECIAL_ZERO_HEADING = "9903.81.92"

    def _coerce_rate(raw_rate: Optional[object]) -> Optional[Decimal]:
        if raw_rate is None:
            return None
        if isinstance(raw_rate, Decimal):
            return raw_rate
        return Decimal(str(raw_rate))

    for measure in applicable_measures:
        rate = _coerce_rate(measure.get("ad_valorem_rate"))
        if rate is None or rate == 0:
            zero_rate_matches.append(measure)
            if (
                melt_origin == "US"
                and (measure.get("heading") or "") == SPECIAL_ZERO_HEADING
            ):
                special_zero_measure = measure
            continue
        measure_with_rate = dict(measure)
        measure_with_rate["ad_valorem_rate"] = rate
        rated_measures.append(measure_with_rate)

    if special_zero_measure:
        zero_decimal = Decimal("0")
        ch99_description = special_zero_measure.get("description") or ""
        ch99_entry = Section232Ch99(
            ch99_id=SPECIAL_ZERO_HEADING,
            alias=SPECIAL_ZERO_HEADING,
            general_rate=zero_decimal,
            ch99_description=ch99_description,
            is_potential=bool(special_zero_measure.get("is_potential")),
        )
        notes = [
            f"Applied Section 232 measures: {SPECIAL_ZERO_HEADING} (0%)"
        ]
        return Section232Computation(
            module_id="232",
            module_name="Section 232 Tariffs",
            applicable=True,
            amount=zero_decimal,
            currency="USD",
            rate="0%",
            ch99_list=[ch99_entry],
            notes=notes,
        )

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
                    "is_potential": offset_measure.get("is_potential"),
                }
            )

    if not rated_measures:
        notes = [
            "No active Section 232 measures matched the provided HTS number on the given entry date."
        ]
        if zero_rate_matches:
            zero_headings = ", ".join(
                m["heading"]
                for m in zero_rate_matches
                if m.get("heading") not in offset_heading_hits
            )
            if zero_headings:
                notes.append(
                    "Section 232 headings matched but have zero ad valorem rate: "
                    f"{zero_headings}."
                )
        return Section232Computation(
            module_id="232",
            module_name="Section 232 Tariffs",
            applicable=False,
            amount=Decimal("0"),
            currency="USD",
            rate=None,
            notes=notes,
        )

    total_rate = sum((m["ad_valorem_rate"] for m in rated_measures), Decimal("0"))
    ch99_list = [
        Section232Ch99(
            ch99_id=m["heading"],
            alias=m["heading"],
            general_rate=m["ad_valorem_rate"],
            ch99_description=m.get("description") or "",
            is_potential=bool(m.get("is_potential")),
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
        "Applied Section 232 measures: " + applied_details
    ]
    if zero_rate_matches:
        zero_headings = ", ".join(
            m["heading"]
            for m in zero_rate_matches
            if m.get("heading") not in offset_heading_hits
        )
        if zero_headings:
            notes.append(
                "Additional Section 232 headings matched with zero rate: "
                + zero_headings
            )

    amount, missing_measurement_fields = _compute_section232_amount(
        hts_number,
        ch99_list,
        import_value_decimal,
        normalized_measurements,
    )
    if missing_measurement_fields:
        missing_list = ", ".join(sorted(set(missing_measurement_fields)))
        notes.append(
            "Section 232 amount requires additional measurements: "
            f"{missing_list} (provide as fractional values, e.g., 0.25 for 25%)."
        )

    return Section232Computation(
        module_id="232",
        module_name="Section 232 Tariffs",
        applicable=True,
        amount=amount,
        currency="USD",
        rate=rate_display,
        ch99_list=ch99_list,
        notes=notes,
    )
