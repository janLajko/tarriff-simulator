"""Tariff simulation FastAPI application."""

from __future__ import annotations

import logging
logging.basicConfig(level=logging.DEBUG)
import json
import os
import time
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Union

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from .basic_hts_rate import (
    BasicRateComputation,
    GENERAL_RATE_CASE_ID,
    MissingMeasurementError,
    compute_basic_duty,
    fetch_basic_hts_record,
    get_unit_config,
)
from .anti_scraping import (
    EncryptionConfigError,
    EncryptionExecutionError,
    encrypt_payload,
)
from .section301_rate import Section301Computation, compute_section301_duty
from .section232_rate import Section232Computation, compute_section232_duty
from .sectionieepa_rate import SectionIEEPAComputation, compute_sectionieepa_duty
from .other_rate import OtherComputation, compute_note_duty

logger = logging.getLogger(__name__)
HTS_CACHE_TTL_SECONDS = 24 * 60 * 60
_hts_record_cache: dict[str, dict] = {}
_hts_cache_expires_at: float = 0.0


DATABASE_DSN = (
    os.getenv("DATABASE_DSN")
    or os.getenv("POSTGRES_DSN")
    or os.getenv("PG_DSN")
)

DISCLAIMER = (
    "Rates and applicability depend on full legal context and notes; confirm before filing."
)

EXTRA_MEASURES_DIR = Path(__file__).resolve().parent.parent / "agent" / "othercharpter-agent" / "output"
EXTRA_MEASURES_NOTES = (33, 36, 37, 38)
_extra_measures_exclusions: Optional[Dict[str, set[str]]] = None
NOTE_PREFIX_TO_MODULE = {
    "9903.94": "Passenger vehicles or light trucks",
    "9903.78": "Semi-finished copper products and copper derivative products",
    "9903.76": "Wood products",
    "9903.74": "Medium- and heavy-duty vehicles",
}
NOTE_MODULE_NAMES = {
    "Passenger vehicles or light trucks": "Chapter 99 Note 33 Tariffs",
    "Semi-finished copper products and copper derivative products": "Chapter 99 Note 36 Tariffs",
    "Wood products": "Chapter 99 Note 37 Tariffs",
    "Medium- and heavy-duty vehicles": "Chapter 99 Note 38 Tariffs",
}


@contextmanager
def db_connection():
    if not DATABASE_DSN:
        raise RuntimeError(
            "DATABASE_DSN (or POSTGRES_DSN/PG_DSN) environment variable is not configured."
        )
    conn = psycopg2.connect(DATABASE_DSN)
    try:
        yield conn
    finally:
        conn.close()


class SimulationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    hts_code: str = Field(
        ...,
        min_length=4,
        validation_alias=AliasChoices("hts_code", "hts_number"),
    )
    country_of_origin: str = Field(
        ..., min_length=2, description="Country of origin for the import."
    )
    entry_date: Optional[date] = Field(
        default=None, description="Entry date for the simulated import."
    )
    date_of_landing: Optional[date] = Field(
        default=None, description="Date when goods landed in the U.S."
    )
    import_value: Optional[Decimal] = Field(
        default=None,
        description="Declared import value; mapped to the usd measurement when provided.",
    )
    copper_percentage: Optional[Decimal] = Field(
        default=None,
        description="Percentage of import value attributed to copper content for 9903.78.01.",
    )
    measurements: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Measurement inputs keyed by unit name (e.g. usd, kg).",
    )
    melt_pour_origin_iso2: Optional[str] = Field(
        default=None,
        description="ISO-2 code describing where steel was melted and poured, when applicable.",
    )
    steel_percentage: Optional[Decimal] = Field(
        default=None,
        description="Fractional (0-1) or percentage (0-100) share of steel content used for Section 232.",
    )
    aluminum_percentage: Optional[Decimal] = Field(
        default=None,
        description="Fractional (0-1) or percentage (0-100) share of aluminum content used for Section 232.",
    )
    steel_pour_country: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Optional indicator describing which material categories require melt/pour origin checks.",
    )
    aluminum_pour_country: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Optional indicator describing which material categories require melt/pour origin checks.",
    )


class RequestEcho(BaseModel):
    hts_code: str
    country_of_origin: str
    entry_date: date
    date_of_landing: Optional[date] = None
    import_value: Optional[Decimal] = None
    measurements: Dict[str, Decimal] = Field(default_factory=dict)
    melt_pour_origin_iso2: Optional[str] = None


class Ch99Entry(BaseModel):
    ch99_id: str
    alias: Optional[str] = None
    general_rate: Decimal
    ch99_description: Optional[str] = None
    amount: Decimal = Field(default=Decimal("0"))
    is_potential: bool = False


class TariffModule(BaseModel):
    module_id: str
    module_name: str
    ch99_list: List[Ch99Entry]
    notes: List[str] = Field(default_factory=list)
    applicable: bool = False
    amount: Decimal = Field(default=Decimal("0"))
    currency: str = "USD"
    rate: Optional[str] = None


class MetaInfo(BaseModel):
    currency: str
    rate_type: str
    evaluated_at: datetime
    disclaimer: str
    calculated: bool
    basic_duty_amount: Decimal
    formula: Optional[str] = None
    required_units: List[str] = Field(default_factory=list)
    provided_inputs: Dict[str, Decimal] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class BasicRateComputationPayload(BaseModel):
    case_id: str
    calculated: bool
    amount: Decimal
    currency: str
    formula: Optional[str] = None
    required_units: List[str] = Field(default_factory=list)
    provided_inputs: Dict[str, Decimal] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)

    @classmethod
    def from_dataclass(
        cls, computation: BasicRateComputation
    ) -> "BasicRateComputationPayload":
        return cls(
            case_id=computation.case_id,
            calculated=computation.calculated,
            amount=computation.amount,
            currency=computation.currency,
            formula=computation.formula,
            required_units=list(computation.required_units),
            provided_inputs=dict(computation.provided_inputs),
            notes=list(computation.notes),
        )


class SimulationResponse(BaseModel):
    request: RequestEcho
    modules: List[TariffModule]
    meta: MetaInfo
    basic_rate_computations: List[BasicRateComputationPayload] = Field(default_factory=list)


class EncryptedEnvelope(BaseModel):
    version: int
    algorithm: str
    ciphertext: str
    nonce: str
    issued_at: datetime
    key_id: Optional[str] = None


app = FastAPI(title="Tariff Simulator API", version="0.3.0")

allowed_origins = ["https://tariff-simulator-frontend.vercel.app","https://tariff.gingercontrol.com"]
runtime_origin = os.getenv("SIMULATOR_FRONTEND_ORIGIN")
if runtime_origin:
    allowed_origins.append(runtime_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _normalize_measurements(payload: SimulationRequest) -> Dict[str, Decimal]:
    measurements: Dict[str, Decimal] = {
        str(key).strip().lower(): value for key, value in payload.measurements.items()
    }
    if payload.import_value is not None:
        measurements.setdefault("usd", payload.import_value)
    if payload.copper_percentage is not None:
        measurements["copper_percentage"] = payload.copper_percentage
    if payload.steel_percentage is not None:
        measurements["steel_percentage"] = payload.steel_percentage
    if payload.aluminum_percentage is not None:
        measurements["aluminum_percentage"] = payload.aluminum_percentage
    return measurements


def _normalize_hts10(value: str) -> Optional[str]:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) >= 10:
        return digits[:10]
    return None


def _ensure_hts_cache() -> None:
    """Load all HTS records into process cache with a TTL."""

    global _hts_record_cache, _hts_cache_expires_at

    now = time.monotonic()
    if _hts_record_cache and now < _hts_cache_expires_at:
        return

    cache_start = time.perf_counter()
    new_cache: dict[str, dict] = {}
    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT hts_number,
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
                FROM hts_codes
                """
            )
            rows = cur.fetchall()

    for raw in rows:
        record = dict(raw)
        hts_number = str(record.get("hts_number") or "").strip()
        if hts_number:
            new_cache[hts_number] = record
        norm = _normalize_hts10(hts_number)
        if norm and norm not in new_cache:
            new_cache[norm] = record

    _hts_record_cache = new_cache
    _hts_cache_expires_at = now + HTS_CACHE_TTL_SECONDS
    logger.info(
        "HTS cache refreshed size=%s duration=%.3fs", len(_hts_record_cache), time.perf_counter() - cache_start
    )


def _get_cached_hts_record(hts_code: str) -> Optional[dict]:
    if not hts_code:
        return None
    normalized = _normalize_hts10(hts_code)
    record = _hts_record_cache.get(hts_code.strip())
    if record:
        return record
    if normalized:
        return _hts_record_cache.get(normalized)
    return None


def _determine_rate_type(required_units: List[str]) -> str:
    if not required_units:
        return "no_duty"
    units = [unit.lower() for unit in required_units]
    has_usd = "usd" in units
    if has_usd and len(units) == 1:
        return "ad_valorem_fraction"
    if has_usd and len(units) > 1:
        return "mixed_specific_ad_valorem"
    return "specific_rate"


def _build_meta_info(
    computation: BasicRateComputation,
    unit_config: Dict[str, object],
) -> MetaInfo:
    evaluated_at = datetime.now().astimezone()
    rate_type = _determine_rate_type(computation.required_units)
    hts_general = str(unit_config.get("hts_general") or "").strip().lower()
    if hts_general:
        normalized = hts_general.replace(" ", "_")
        if "ad_valorem" in normalized and "usd" in computation.required_units:
            rate_type = "ad_valorem_fraction"
        else:
            rate_type = normalized
    formula = computation.formula
    notes = list(computation.notes)
    return MetaInfo(
        currency=computation.currency,
        rate_type=rate_type,
        evaluated_at=evaluated_at,
        disclaimer=DISCLAIMER,
        calculated=computation.calculated,
        basic_duty_amount=computation.amount,
        formula=formula,
        required_units=list(computation.required_units),
        provided_inputs=dict(computation.provided_inputs),
        notes=notes,
    )


def _load_extra_measure_exclusions() -> Dict[str, set[str]]:
    global _extra_measures_exclusions
    if _extra_measures_exclusions is not None:
        return _extra_measures_exclusions
    exclusions: Dict[str, set[str]] = {}
    for note_number in EXTRA_MEASURES_NOTES:
        path = EXTRA_MEASURES_DIR / f"note{note_number}_extra_measures.json"
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning("Failed to load extra measures file %s: %s", path, exc)
            continue
        measures = payload.get("measures") or []
        for measure in measures:
            heading = str(measure.get("heading") or "").strip()
            if not heading:
                continue
            for scope in measure.get("scopes") or []:
                if scope.get("relation") != "exclude":
                    continue
                keys_value = scope.get("keys") or ""
                if not keys_value:
                    continue
                if isinstance(keys_value, list):
                    raw_keys = keys_value
                else:
                    raw_keys = str(keys_value).split(",")
                keys = [str(key).strip() for key in raw_keys if str(key).strip()]
                if not keys:
                    continue
                exclusions.setdefault(heading, set()).update(keys)
    _extra_measures_exclusions = exclusions
    return exclusions


def _build_modules(
    *sections: Section301Computation | Section232Computation | SectionIEEPAComputation | OtherComputation,
) -> List[TariffModule]:
    modules: List[TariffModule] = []
    note_headings: Dict[int, set[str]] = {}
    for computation in sections:
        module_id = getattr(computation, "module_id", "")
        if module_id.startswith("note"):
            suffix = module_id[4:]
            if suffix.isdigit():
                headings = {
                    str(entry.ch99_id).strip()
                    for entry in computation.ch99_list
                    if entry.ch99_id
                }
                if headings:
                    note_headings[int(suffix)] = headings
    excluded_headings: set[str] = set()
    if note_headings:
        extra_exclusions = _load_extra_measure_exclusions()
        all_note_headings: set[str] = set().union(*note_headings.values())
        for heading, exclusions in extra_exclusions.items():
            if all_note_headings.intersection(exclusions):
                excluded_headings.add(heading)

    def _note_module_for_heading(heading: str, fallback: str) -> str:
        for prefix, module_id in NOTE_PREFIX_TO_MODULE.items():
            if heading.startswith(prefix):
                return module_id
        return fallback

    def _format_rate(rate: Decimal) -> str:
        normalized = rate.normalize()
        prefix = "+" if normalized > 0 else ""
        return f"{prefix}{normalized}%"

    for computation in sections:
        entries = computation.ch99_list
        if computation.module_id in {"232", "ieepa"} and excluded_headings:
            entries = [
                entry
                for entry in entries
                if entry.ch99_id and entry.ch99_id.strip() not in excluded_headings
            ]
        if computation.module_id.startswith("note"):
            grouped_entries: Dict[str, List[Ch99Entry]] = {}
            for entry in entries:
                heading = str(entry.ch99_id or "").strip()
                if not heading:
                    continue
                module_id = _note_module_for_heading(heading, computation.module_id)
                grouped_entries.setdefault(module_id, []).append(entry)
            if not grouped_entries:
                continue
            constraint_notes = [
                note
                for note in computation.notes
                if not note.startswith("Applied Chapter 99 note")
            ]
            for module_id, group in grouped_entries.items():
                if not group:
                    continue
                ch99_entries = [
                    Ch99Entry(
                        ch99_id=entry.ch99_id,
                        alias=entry.alias,
                        general_rate=entry.general_rate,
                        ch99_description=entry.ch99_description,
                        amount=entry.amount,
                        is_potential=entry.is_potential,
                    )
                    for entry in group
                ]
                applied_details = ", ".join(
                    f"{entry.ch99_id} ({_format_rate(entry.general_rate)})"
                    for entry in group
                )
                notes = []
                if module_id.startswith("note") and applied_details:
                    note_number = module_id[4:]
                    if note_number.isdigit():
                        notes.append(
                            f"Applied Chapter 99 note {note_number} measures: {applied_details}"
                        )
                if constraint_notes:
                    notes.extend(constraint_notes)
                total_rate = sum((entry.general_rate for entry in group), Decimal("0"))
                rate_display = f"{total_rate.normalize()}%"
                amount = sum((entry.amount for entry in group), Decimal("0"))
                modules.append(
                    TariffModule(
                        module_id=module_id,
                        module_name=NOTE_MODULE_NAMES.get(module_id, computation.module_name),
                        ch99_list=ch99_entries,
                        notes=notes,
                        applicable=bool(group),
                        amount=amount,
                        currency=computation.currency,
                        rate=rate_display,
                    )
                )
            continue
        ch99_entries = [
            Ch99Entry(
                ch99_id=entry.ch99_id,
                alias=entry.alias,
                general_rate=entry.general_rate,
                ch99_description=entry.ch99_description,
                amount=entry.amount,
                is_potential=entry.is_potential
            )
            for entry in entries
        ]
        modules.append(
            TariffModule(
                module_id=computation.module_id,
                module_name=computation.module_name,
                ch99_list=ch99_entries,
                notes=list(computation.notes),
                applicable=computation.applicable,
                amount=computation.amount,
                currency=computation.currency,
                rate=computation.rate,
            )
        )
    return modules


def _build_request_echo(
    payload: SimulationRequest,
    normalized_measurements: Dict[str, Decimal],
    hts_code: str,
    entry_date: date,
    melt_pour_origin_iso2: Optional[str],
) -> RequestEcho:
    import_value = (
        payload.import_value
        if payload.import_value is not None
        else normalized_measurements.get("usd")
    )
    return RequestEcho(
        hts_code=hts_code,
        country_of_origin=payload.country_of_origin.strip().upper(),
        entry_date=entry_date,
        date_of_landing=payload.date_of_landing,
        import_value=import_value,
        measurements=normalized_measurements,
        melt_pour_origin_iso2=melt_pour_origin_iso2,
    )


@app.post("/simulate", response_model=EncryptedEnvelope)
def simulate_tariff(payload: SimulationRequest) -> EncryptedEnvelope:
    request_start = time.perf_counter()
    entry = payload.entry_date or date.today()
    country = payload.country_of_origin.strip().upper()
    general_melt_origin = (
        payload.melt_pour_origin_iso2.strip().upper()
        if payload.melt_pour_origin_iso2
        else None
    )
    steel_melt_origin = (
        payload.steel_pour_country.strip().upper()
        if payload.steel_pour_country
        else None
    )
    aluminum_melt_origin = (
        payload.aluminum_pour_country.strip().upper()
        if payload.aluminum_pour_country
        else None
    )
    melt_origin = general_melt_origin or steel_melt_origin or aluminum_melt_origin
    measurements = _normalize_measurements(payload)
    import_value_amount = measurements.get("usd")

    logger.info(
        "simulate_tariff start hts=%s country=%s entry=%s",
        payload.hts_code,
        country,
        entry,
    )

    try:
        db_start = time.perf_counter()
        _ensure_hts_cache()
        record = _get_cached_hts_record(payload.hts_code)
        source = "cache"
        if record is None:
            with db_connection() as conn:
                record = fetch_basic_hts_record(conn, payload.hts_code.strip())
            source = "db"
            if record:
                # Keep cache fresh for lookups we might have missed during bulk load.
                normalized = _normalize_hts10(payload.hts_code)
                if normalized:
                    _hts_record_cache[normalized] = record
        logger.info(
            "simulate_tariff fetched HTS record in %.3fs (%s)",
            time.perf_counter() - db_start,
            source,
        )
    except Exception as exc:  # pragma: no cover - database connectivity
        logger.exception("Failed to fetch HTS record from database.")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch HTS data from database.",
        ) from exc

    if not record:
        raise HTTPException(
            status_code=404,
            detail="HTS number not found in hts_codes table.",
        )

    canonical_hts = str(record["hts_number"])
    try:
        basic_start = time.perf_counter()
        basic_computations = compute_basic_duty(
            canonical_hts,
            measurements,
            country_of_origin=country,
            special_rate_of_duty=record.get("special_rate_of_duty"),
        )
        logger.info("simulate_tariff basic duty computed in %.3fs", time.perf_counter() - basic_start)
        print(basic_computations)
    except MissingMeasurementError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(exc),
                "missing_measurements": exc.missing,
            },
        ) from exc

    unit_config_raw = get_unit_config(canonical_hts) or {}
    general_basic_computation = next(
        (comp for comp in basic_computations if comp.case_id == GENERAL_RATE_CASE_ID),
        basic_computations[0],
    )
    base_duty_rate_percent: Optional[Decimal] = None
    if import_value_amount not in (None, Decimal("0")):
        try:
            base_duty_rate_percent = (general_basic_computation.amount / Decimal(import_value_amount)) * Decimal("100")
        except Exception:
            base_duty_rate_percent = None
    meta_info = _build_meta_info(general_basic_computation, unit_config_raw)
    basic_rate_payloads = [
        BasicRateComputationPayload.from_dataclass(computation)
        for computation in basic_computations
    ]

    def _timed_call(name: str, func, *args, **kwargs):
        started = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            logger.info("simulate_tariff %s computed in %.3fs", name, time.perf_counter() - started)

    with ThreadPoolExecutor(max_workers=4) as executor:
        fut_301 = executor.submit(
            _timed_call,
            "section301",
            compute_section301_duty,
            canonical_hts,
            country,
            entry,
            import_value=import_value_amount,
        )
        fut_ieepa = executor.submit(
            _timed_call,
            "sectionieepa",
            compute_sectionieepa_duty,
            canonical_hts,
            country,
            entry,
            date_of_landing=payload.date_of_landing,
            import_value=import_value_amount,
            melt_pour_origin_iso2=melt_origin,
            measurements=measurements,
            base_duty_rate_percent=base_duty_rate_percent,
        )
        fut_232 = executor.submit(
            _timed_call,
            "section232",
            compute_section232_duty,
            canonical_hts,
            country,
            entry,
            steel_melt_origin,
            aluminum_melt_origin,
            import_value=import_value_amount,
            measurements=measurements,
            steel_percentage=payload.steel_percentage,
            aluminum_percentage=payload.aluminum_percentage,
        )
        fut_note33 = executor.submit(
            _timed_call,
            "note33",
            compute_note_duty,
            note_number=33,
            hts_number=canonical_hts,
            country_of_origin=country,
            entry_date=entry,
            date_of_landing=payload.date_of_landing,
            import_value=import_value_amount,
            copper_percentage=payload.copper_percentage,
            base_rate_decimal=base_duty_rate_percent,
        )
        section_301_result = fut_301.result()
        section_232_result = fut_232.result()
        section_ieepa_result = fut_ieepa.result()
        note33_result = fut_note33.result()
    modules = _build_modules(section_301_result, section_232_result, section_ieepa_result, note33_result)
    request_echo = _build_request_echo(
        payload, measurements, canonical_hts, entry, melt_origin
    )

    envelope_payload = SimulationResponse(
        request=request_echo,
        modules=modules,
        meta=meta_info,
        basic_rate_computations=basic_rate_payloads,
    ).model_dump(mode="json")

    print(envelope_payload)

    try:
        encrypt_start = time.perf_counter()
        encrypted = encrypt_payload(envelope_payload)
        logger.info("simulate_tariff payload encrypted in %.3fs", time.perf_counter() - encrypt_start)
    except (EncryptionConfigError, EncryptionExecutionError) as exc:
        logger.exception("Failed to encrypt response payload.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    total_elapsed = time.perf_counter() - request_start
    logger.info("simulate_tariff completed in %.3fs", total_elapsed)
    return EncryptedEnvelope.model_validate(encrypted)
