"""Section IEEPA duty computation utilities backed by the Postgres data model."""

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
class SectionIEEPACh99:
    ch99_id: str
    alias: str
    general_rate: Decimal
    ch99_description: str
    amount: Decimal
    is_potential: bool = False


@dataclass
class SectionIEEPAComputation:
    module_id: str
    module_name: str
    applicable: bool
    amount: Decimal
    currency: str
    rate: Optional[str]
    ch99_list: List[SectionIEEPACh99] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


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


def _date_active(entry: date, start: Optional[date], end: Optional[date]) -> bool:
    if start and entry < start:
        return False
    if end and entry > end:
        return False
    return True


PARTIAL_HEADING_SET = {
    "9903.01.25",
    "9903.01.35",
    "9903.01.39",
    "9903.01.63",
}
PARTIAL_RANGE_START = 99030201
PARTIAL_RANGE_END = 99030273


def _coerce_decimal(value: Optional[object]) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _extract_share(measurements: Optional[Dict[str, Decimal]], key: str) -> Optional[Decimal]:
    if not measurements:
        return None
    raw = measurements.get(key.lower())
    share = _coerce_decimal(raw)
    if share is None:
        return None
    if share < 0:
        return None
    if share > 1 and share <= 100:
        share = share / Decimal("100")
    if share > 1:
        return None
    return share


def _is_partial_ieepa_heading(heading: str) -> bool:
    normalized = heading.strip()
    if normalized in PARTIAL_HEADING_SET:
        return True
    digits = _normalize_hts(normalized)
    if len(digits) < 8:
        return False
    try:
        num = int(digits[:8])
    except ValueError:
        return False
    return PARTIAL_RANGE_START <= num <= PARTIAL_RANGE_END


class SectionIEEPAEvaluator:
    def __init__(
        self,
        conn,
        entry_date: date,
        country: str,
        melt_pour_origin: Optional[str],
    ):
        self.conn = conn
        self.entry_date = entry_date
        self.country = (country or "").upper()
        self.melt_origin = (melt_pour_origin or "").upper()
        self.measures = self._load_measures()
        self.heading_to_measure: Dict[str, Dict] = {
            m["heading"]: m for m in self.measures
        }
        self.id_to_measure: Dict[int, Dict] = {m["id"]: m for m in self.measures}
        self.scope_cache: Dict[int, Dict[str, List[Dict]]] = {}
        self.match_cache: Dict[Tuple[int, str], Tuple[bool, List[str]]] = {}

    def _country_allows_measure(self, measure: Dict) -> bool:
        """Mirror Section 232 country filtering: explicit country must match, otherwise honor exclusions."""
        target_country = (measure.get("country_iso2") or "").upper()
        if target_country:
            return self.country == target_country
        excluded = measure.get("origin_exclude_iso2") or []
        excluded_upper = {(code or "").upper() for code in excluded}
        if not self.country:
            return not excluded_upper
        return self.country not in excluded_upper

    def _load_measures(self) -> List[Dict]:
        query = """
            SELECT
                m.id,
                m.heading,
                m.country_iso2,
                m.ad_valorem_rate,
                m.value_basis,
                m.melt_pour_origin_iso2,
                m.origin_exclude_iso2,
                m.effective_start_date,
                m.effective_end_date,
                m.is_potential,
                COALESCE(h.description, '') AS description
            FROM sieepa_measures AS m
            LEFT JOIN LATERAL (
                SELECT description
                FROM hts_codes hc
                WHERE hc.hts_number = m.heading
                ORDER BY hc.row_order
                LIMIT 1
            ) AS h ON TRUE
            WHERE m.effective_start_date <= %s
              AND (%s <= COALESCE(m.effective_end_date, %s))
        """
        params: List[object] = [self.entry_date, self.entry_date, self.entry_date]
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [row for row in rows if self._country_allows_measure(row)]

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
            FROM sieepa_scope_measure_map AS map
            JOIN sieepa_scope AS scope ON scope.id = map.scope_id
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
            FROM sieepa_measures
            WHERE heading = %s
              AND (%s <= COALESCE(effective_end_date, %s))
              AND effective_start_date <= %s
            ORDER BY effective_start_date DESC, id DESC
            LIMIT 1
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (heading, self.entry_date, self.entry_date, self.entry_date))
            row = cur.fetchone()
        if row and self._country_allows_measure(row):
            self.heading_to_measure[row["heading"]] = row
            if row not in self.measures:
                self.measures.append(row)
            if row["id"] not in self.id_to_measure:
                self.id_to_measure[row["id"]] = row
            return row["id"]
        return None

    def constraints_met(self, measure: Dict) -> Tuple[bool, Optional[str]]:
        if not self._country_allows_measure(measure):
            return False, f"Origin {self.country or 'UNKNOWN'} not eligible for {measure.get('heading', '')}"
        excluded = measure.get("origin_exclude_iso2") or []
        normalized_excludes = {str(entry or "").upper() for entry in excluded}
        if self.country and self.country in normalized_excludes:
            return False, f"Excluded origin {self.country}"

        required_melt = (measure.get("melt_pour_origin_iso2") or "").upper()
        if required_melt:
            if not self.melt_origin:
                return False, f"Requires melt/pour origin {required_melt}"
            if self.melt_origin != required_melt:
                return False, f"Requires melt/pour origin {required_melt}"
        return True, None

    def measure_covers(
        self, measure_id: int, hts_code: str, visited: Optional[set[int]] = None
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

        measure = self.id_to_measure.get(measure_id)
        heading = (measure.get("heading") if measure else "") or ""
        prefix_only = heading.startswith("9903.01") or heading.startswith("9903.02")
        rate_raw = measure.get("ad_valorem_rate") if measure else None
        try:
            rate_decimal = Decimal(str(rate_raw)) if rate_raw is not None else None
        except Exception:
            rate_decimal = None
        prefix_override = prefix_only and rate_decimal is not None and rate_decimal > 0

        if prefix_override:
            include_hit = True
        elif includes:
            include_hit = any(matches_scope(scope, True) for scope in includes)
        else:
            include_hit = False
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


def compute_sectionieepa_duty(
    hts_number: str,
    country_of_origin: str,
    entry_date: date,
    import_value: Optional[Decimal] = None,
    melt_pour_origin_iso2: Optional[str] = None,
    measurements: Optional[Dict[str, Decimal]] = None,
) -> SectionIEEPAComputation:
    """Compute Section IEEPA duty information for a given HTS number."""

    if not isinstance(entry_date, date):
        raise TypeError("entry_date must be a datetime.date instance")

    if psycopg2 is None:  # pragma: no cover - runtime enforcement
        raise RuntimeError(
            "psycopg2 is required to compute Section IEEPA duties. "
            "Install psycopg2-binary or psycopg2 and ensure it is available."
        )

    dsn = os.getenv("DATABASE_DSN")
    if not dsn:
        raise RuntimeError("DATABASE_DSN environment variable is required for Section IEEPA calculations.")

    origin = (country_of_origin or "").upper()
    melt_origin = (melt_pour_origin_iso2 or "").upper()

    with psycopg2.connect(dsn) as conn:
        evaluator = SectionIEEPAEvaluator(conn, entry_date, origin, melt_origin)
        applicable_measures: List[Dict] = []
        excluded_measures: List[Tuple[Dict, List[str]]] = []
        constraint_notes: List[str] = []
        for measure in evaluator.measures:
            allowed, reason = evaluator.constraints_met(measure)
            if not allowed:
                if reason:
                    constraint_notes.append(f"{measure['heading']}: {reason}")
                continue
            covers, exclusions = evaluator.measure_covers(measure["id"], hts_number)
            if covers:
                applicable_measures.append(measure)
            elif exclusions:
                excluded_measures.append((measure, exclusions))

    steel_share = _extract_share(measurements, "steel_percentage")
    aluminum_share = _extract_share(measurements, "aluminum_percentage")
    rated_measures: List[Dict] = []
    zero_rate_matches: List[Dict] = []

    import_value_decimal = _coerce_decimal(import_value)

    def _taxable_share(heading: str) -> Decimal:
        if not _is_partial_ieepa_heading(heading):
            return Decimal("1")
        s = steel_share or Decimal("0")
        a = aluminum_share or Decimal("0")
        total = s + a
        if total > Decimal("1"):
            total = Decimal("1")
        portion = Decimal("1") - total
        if portion < 0:
            portion = Decimal("0")
        return portion

    for measure in applicable_measures:
        rate = _coerce_decimal(measure.get("ad_valorem_rate"))
        if rate is None or rate == 0:
            zero_rate_matches.append(measure)
            continue
        measure_with_rate = dict(measure)
        measure_with_rate["ad_valorem_rate"] = rate
        rated_measures.append(measure_with_rate)

    offset_heading_hits: set[str] = set()
    for measure, exclusions in excluded_measures:
        rate = _coerce_decimal(measure.get("ad_valorem_rate"))
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
                    "is_potential": offset_measure.get("is_potential") or False,
                }
            )

    if not rated_measures:
        notes = [
            "No active Section IEEPA measures matched the provided HTS number on the given entry date."
        ]
        if zero_rate_matches:
            zero_headings = ", ".join(
                m["heading"]
                for m in zero_rate_matches
                if m.get("heading") not in offset_heading_hits
            )
            if zero_headings:
                notes.append(
                    "Section IEEPA headings matched but have zero ad valorem rate: "
                    f"{zero_headings}."
                )
        notes.extend(constraint_notes)
        return SectionIEEPAComputation(
            module_id="ieepa",
            module_name="Section IEEPA Derivative Tariffs",
            applicable=False,
            amount=Decimal("0"),
            currency="USD",
            rate=None,
            notes=notes,
        )

    total_rate = sum((m["ad_valorem_rate"] for m in rated_measures), Decimal("0"))

    def _entry_amount(rate: Decimal, heading: str) -> Decimal:
        if import_value_decimal is None:
            return Decimal("0")
        share = _taxable_share(heading)
        return (import_value_decimal * share * rate) / Decimal("100")

    ch99_list: List[SectionIEEPACh99] = []
    for m in rated_measures:
        heading = m["heading"]
        rate_val = m["ad_valorem_rate"]
        ch99_list.append(
            SectionIEEPACh99(
                ch99_id=heading,
                alias=m.get("alias") or heading,
                general_rate=rate_val,
                ch99_description=m.get("description") or "",
                amount=_entry_amount(rate_val, heading),
                is_potential=bool(m.get("is_potential")),
            )
        )

    def _format_rate(rate: Decimal) -> str:
        normalized = rate.normalize()
        prefix = "+" if normalized > 0 else ""
        return f"{prefix}{normalized}%"

    applied_details = ", ".join(
        f"{m['heading']} ({_format_rate(m['ad_valorem_rate'])})" for m in rated_measures
    )
    notes = ["Applied Section IEEPA measures: " + applied_details]
    if zero_rate_matches:
        zero_headings = ", ".join(
            m["heading"]
            for m in zero_rate_matches
            if m.get("heading") not in offset_heading_hits
        )
        if zero_headings:
            notes.append(
                "Additional Section IEEPA headings matched with zero rate: "
                + zero_headings
            )
    notes.extend(constraint_notes)

    amount = sum((entry.amount for entry in ch99_list), Decimal("0"))

    if import_value_decimal is not None and import_value_decimal != 0:
        effective_rate = (amount / import_value_decimal) * Decimal("100")
        rate_display = f"{effective_rate.normalize()}%"
    else:
        normalized_total = total_rate.normalize()
        rate_display = f"{normalized_total}%"

    return SectionIEEPAComputation(
        module_id="ieepa",
        module_name="Section IEEPA Derivative Tariffs",
        applicable=True,
        amount=amount,
        currency="USD",
        rate=rate_display,
        ch99_list=ch99_list,
        notes=notes,
    )
