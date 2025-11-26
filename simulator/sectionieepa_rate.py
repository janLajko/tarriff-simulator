"""Section IEEPA duty computation utilities backed by the Postgres data model."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
import logging
import time
from typing import Dict, List, Optional, Tuple

try:  # pragma: no cover - optional dependency for environments without psycopg2
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:  # pragma: no cover - allow import without postgres libs
    psycopg2 = None  # type: ignore[assignment]
    RealDictCursor = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency for cross-module scope checks
    from .section232_rate import Section232Evaluator
except Exception:  # pragma: no cover
    Section232Evaluator = None  # type: ignore[assignment]

try:  # pragma: no cover - optional cache dependency
    from theine import Cache as TheineCache
except Exception:  # pragma: no cover
    TheineCache = None  # type: ignore[assignment]


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


logger = logging.getLogger(__name__)


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

ALWAYS_INCLUDE_HEADINGS = {
    "9903.01.01",
    "9903.01.10",
    "9903.01.24",
    "9903.01.25",
    "9903.01.26",
    "9903.01.27",
    "9903.01.35",
    "9903.01.39",
    "9903.01.63",
    "9903.01.77",
    "9903.01.84"
}
ALWAYS_INCLUDE_RANGE_START = 99030201  # exclusive
ALWAYS_INCLUDE_RANGE_END = 99030274  # exclusive

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
EU_MEMBER_NAMES = {
    "AUSTRIA",
    "BELGIUM",
    "BULGARIA",
    "CROATIA",
    "CYPRUS",
    "CZECHIA",
    "CZECH REPUBLIC",
    "DENMARK",
    "ESTONIA",
    "FINLAND",
    "FRANCE",
    "GERMANY",
    "GREECE",
    "HUNGARY",
    "IRELAND",
    "ITALY",
    "LATVIA",
    "LITHUANIA",
    "LUXEMBOURG",
    "MALTA",
    "NETHERLANDS",
    "POLAND",
    "PORTUGAL",
    "ROMANIA",
    "SLOVAKIA",
    "SLOVENIA",
    "SPAIN",
    "SWEDEN",
}

IEEPA_CACHE_TTL_SECONDS = 24 * 60 * 60

if TheineCache is not None:
    _ieepa_cache = TheineCache(maxsize=32, ttl=IEEPA_CACHE_TTL_SECONDS)
else:  # simple fallback cache with TTL
    _ieepa_cache = None
    _ieepa_cache_store: Dict[str, Tuple[float, Dict]] = {}


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


def _normalize_country_iso2_or_name(value: str) -> str:
    normalized = (value or "").strip().upper()
    if not normalized:
        return ""
    if normalized in EU_MEMBER_CODES or normalized in EU_MEMBER_NAMES:
        return "EU"
    return normalized


def _cache_get(key: str) -> Optional[Dict]:
    if _ieepa_cache is not None:
        try:
            return _ieepa_cache.get(key)  # type: ignore[call-arg]
        except Exception:
            return None
    entry = _ieepa_cache_store.get(key)
    if not entry:
        return None
    expires_at, value = entry
    if time.monotonic() > expires_at:
        _ieepa_cache_store.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: Dict) -> None:
    if _ieepa_cache is not None:
        try:
            _ieepa_cache.set(key, value)  # type: ignore[call-arg]
        except Exception:
            return
        return
    _ieepa_cache_store[key] = (time.monotonic() + IEEPA_CACHE_TTL_SECONDS, value)


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
    return PARTIAL_RANGE_START < num < PARTIAL_RANGE_END


def _is_always_include_heading(heading: str) -> bool:
    normalized = heading.strip()
    if normalized in ALWAYS_INCLUDE_HEADINGS:
        return True
    digits = _normalize_hts(normalized)
    if len(digits) < 8:
        return False
    try:
        num = int(digits[:8])
    except ValueError:
        return False
    return ALWAYS_INCLUDE_RANGE_START < num < ALWAYS_INCLUDE_RANGE_END


class SectionIEEPAEvaluator:
    def __init__(
        self,
        conn,
        entry_date: date,
        country: str,
        melt_pour_origin: Optional[str],
        section232_evaluator: Optional[object] = None,
    ):
        self.conn = conn
        self.entry_date = entry_date
        self.country = _normalize_country_iso2_or_name(country)
        self.melt_origin = _normalize_country_iso2_or_name(melt_pour_origin)
        self.scope_cache: Dict[int, Dict[str, List[Dict]]] = {}
        self.measures = self._load_measures()
        self.heading_to_measure: Dict[str, Dict] = {
            m["heading"]: m for m in self.measures
        }
        self.id_to_measure: Dict[int, Dict] = {m["id"]: m for m in self.measures}
        self.match_cache: Dict[Tuple[int, str, bool], Tuple[bool, List[str]]] = {}
        self.section232_evaluator = section232_evaluator

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
        cache_key = self.country or "__ALL__"
        cached = _cache_get(cache_key)
        if cached is None:
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
                WHERE m.country_iso2 IS NULL OR m.country_iso2 = %s
            """
            params: List[object] = [self.country]
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                measure_rows = cur.fetchall()
            measure_ids = [row["id"] for row in measure_rows]
            scope_map: Dict[int, Dict[str, List[Dict]]] = {}
            if measure_ids:
                scope_query = """
                    SELECT
                        map.measure_id,
                        map.relation,
                        scope.key,
                        scope.key_type,
                        scope.country_iso2,
                        scope.source_label,
                        scope.effective_start_date,
                        scope.effective_end_date
                    FROM sieepa_scope_measure_map AS map
                    JOIN sieepa_scope AS scope ON scope.id = map.scope_id
                    WHERE map.measure_id = ANY(%s)
                """
                with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(scope_query, (measure_ids,))
                    scope_rows = cur.fetchall()
                for row in scope_rows:
                    rel = row.get("relation")
                    if rel not in {"include", "exclude"}:
                        continue
                    mid = row["measure_id"]
                    bucket = scope_map.setdefault(mid, {"include": [], "exclude": []})
                    bucket[rel].append(row)
            cached = {"measures": measure_rows, "scopes": scope_map}
            _cache_set(cache_key, cached)
        rows = cached.get("measures", [])
        scope_map_cached = cached.get("scopes", {})

        filtered = [
            row
            for row in rows
            if self._country_allows_measure(row)
            and row.get("effective_start_date") <= self.entry_date
            and self.entry_date <= (row.get("effective_end_date") or self.entry_date)
        ]
        for mid, scopes in scope_map_cached.items():
            self.scope_cache[mid] = scopes
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Section IEEPA _load_measures country=%s entry_date=%s raw=%s filtered=%s",
                self.country,
                self.entry_date,
                len(rows) if rows else 0,
                len(filtered),
            )
            for row in filtered:
                if row["heading"] in {"9903.02.29", "9903.02.30", "9903.02.31"}:
                    logger.debug(
                        "IEEPA measure kept heading=%s country_iso2=%s excludes=%s rate=%s",
                        row.get("heading"),
                        row.get("country_iso2"),
                        row.get("origin_exclude_iso2"),
                        row.get("ad_valorem_rate"),
                    )
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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "IEEPA _measure_id_for_heading filtered out heading=%s country=%s row_country=%s",
                heading,
                self.country,
                row.get("country_iso2") if row else None,
            )
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
        self,
        measure_id: int,
        hts_code: str,
        visited: Optional[set[int]] = None,
        require_scope_match: bool = False,
    ) -> Tuple[bool, List[str]]:
        normalized_hts = _normalize_hts(hts_code)
        key = (measure_id, normalized_hts, require_scope_match)
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

        def matches_scope(
            scope_row: Dict,
            recurse: bool,
            force_scope: bool = False,
            force_ch99: bool = False,
            is_exclusion: bool = False,
        ) -> bool:
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
                if self.section232_evaluator and key_value.startswith(("9903.81", "9903.85")):
                    child_id = self.section232_evaluator._measure_id_for_heading(key_value)  # type: ignore[attr-defined]
                    if not child_id:
                        return False
                    child_match, _ = self.section232_evaluator.measure_covers(child_id, hts_code)  # type: ignore[attr-defined]
                    return child_match
                if force_ch99:
                    return True
                if not recurse:
                    return False
                child_id = self._measure_id_for_heading(key_value)
                if not child_id:
                    return False
                child_match, _ = self.measure_covers(
                    child_id,
                    hts_code,
                    visited.copy(),
                    require_scope_match=False if is_exclusion else require_scope_match,
                )
                return child_match
            return _code_matches(key_value, hts_code)

        measure = self.id_to_measure.get(measure_id)
        heading = (measure.get("heading") if measure else "") or ""
        always_include = _is_always_include_heading(heading)

        if always_include and not require_scope_match:
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
            key_value = scope.get("key", "") or ""
            force_scope = key_value.startswith("99")
            if matches_scope(
                scope,
                True,
                force_scope=force_scope,
                force_ch99=False,
                is_exclusion=True,
            ):
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
    base_duty_rate_percent: Optional[Decimal] = None,
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
        s232_eval = None
        if Section232Evaluator is not None:
            try:
                s232_eval = Section232Evaluator(
                    conn,
                    entry_date,
                    origin,
                    melt_pour_origin_iso2,
                    melt_pour_origin_iso2,
                )
            except Exception:
                s232_eval = None
        evaluator = SectionIEEPAEvaluator(conn, entry_date, origin, melt_origin, section232_evaluator=s232_eval)
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
            elif logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "IEEPA measure no match heading=%s country=%s exclusions=%s",
                    measure.get("heading"),
                    measure.get("country_iso2"),
                    exclusions,
                )
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "IEEPA applicable_measures count=%s items=%s",
                len(applicable_measures),
                [(m.get("heading"), m.get("country_iso2")) for m in applicable_measures],
            )

    steel_share = _extract_share(measurements, "steel_percentage")
    aluminum_share = _extract_share(measurements, "aluminum_percentage")
    rated_measures: List[Dict] = []
    zero_rate_matches: List[Dict] = []

    import_value_decimal = _coerce_decimal(import_value)
    base_rate_decimal: Optional[Decimal] = None
    if base_duty_rate_percent is not None:
        try:
            base_rate_decimal = Decimal(str(base_duty_rate_percent))
        except Exception:
            base_rate_decimal = None

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
        heading = (measure.get("heading") or "").strip()
        if base_rate_decimal is not None and heading in {"9903.02.72", "9903.02.73"}:
            if heading == "9903.02.72" and base_rate_decimal < Decimal("15"):
                continue
            if heading == "9903.02.73" and base_rate_decimal >= Decimal("15"):
                continue
        if base_rate_decimal is not None and heading in {"9903.02.19", "9903.02.72"}:
            adjusted = Decimal("15") - base_rate_decimal
            if adjusted < 0:
                adjusted = Decimal("0")
            rate = adjusted
        if rate is None or rate == 0:
            zero_rate_matches.append(measure)
            continue
        measure_with_rate = dict(measure)
        measure_with_rate["ad_valorem_rate"] = rate
        rated_measures.append(measure_with_rate)

    offset_heading_hits: set[str] = set()

    if not rated_measures:
        zero_rate_ch99: List[SectionIEEPACh99] = []
        for measure in zero_rate_matches:
            heading = measure.get("heading") or ""
            if heading in offset_heading_hits:
                continue
            zero_rate_ch99.append(
                SectionIEEPACh99(
                    ch99_id=heading,
                    alias=measure.get("alias") or heading,
                    general_rate=_coerce_decimal(measure.get("ad_valorem_rate")) or Decimal("0"),
                    ch99_description=measure.get("description") or "",
                    amount=Decimal("0"),
                    is_potential=bool(measure.get("is_potential")),
                )
            )

        notes: List[str] = []
        if zero_rate_ch99:
            zero_headings = ", ".join(entry.ch99_id for entry in zero_rate_ch99)
            notes.append(
                "Section IEEPA headings matched but have zero ad valorem rate: "
                + zero_headings
            )
        else:
            notes.append(
                "No active Section IEEPA measures matched the provided HTS number on the given entry date."
            )
        notes.extend(constraint_notes)
        return SectionIEEPAComputation(
            module_id="ieepa",
            module_name="Section IEEPA Derivative Tariffs",
            applicable=bool(zero_rate_ch99),
            amount=Decimal("0"),
            currency="USD",
            rate="0%" if zero_rate_ch99 else None,
            ch99_list=zero_rate_ch99,
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

    if zero_rate_matches:
        existing = {entry.ch99_id for entry in ch99_list}
        for measure in zero_rate_matches:
            heading = measure.get("heading") or ""
            if heading in offset_heading_hits:
                continue
            if heading in existing:
                continue
            existing.add(heading)
            ch99_list.append(
                SectionIEEPACh99(
                    ch99_id=heading,
                    alias=measure.get("alias") or heading,
                    general_rate=Decimal("0"),
                    ch99_description=measure.get("description") or "",
                    amount=Decimal("0"),
                    is_potential=bool(measure.get("is_potential")),
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
