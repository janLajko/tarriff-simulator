"""Tariff simulation FastAPI application."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Optional, Union

import psycopg2
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

logger = logging.getLogger(__name__)

DATABASE_DSN = (
    os.getenv("DATABASE_DSN")
    or os.getenv("POSTGRES_DSN")
    or os.getenv("PG_DSN")
)

DISCLAIMER = (
    "Rates and applicability depend on full legal context and notes; confirm before filing."
)


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
    pour_country: Optional[Union[str, List[str]]] = Field(
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

allowed_origins = ["https://tariff-simulator-frontend.vercel.app"]
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
    if payload.steel_percentage is not None:
        measurements["steel_percentage"] = payload.steel_percentage
    if payload.aluminum_percentage is not None:
        measurements["aluminum_percentage"] = payload.aluminum_percentage
    return measurements


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


def _build_modules(
    *sections: Section301Computation | Section232Computation,
) -> List[TariffModule]:
    modules: List[TariffModule] = []
    for computation in sections:
        ch99_entries = [
            Ch99Entry(
                ch99_id=entry.ch99_id,
                alias=entry.alias,
                general_rate=entry.general_rate,
                ch99_description=entry.ch99_description,
                amount=entry.amount,
                is_potential=entry.is_potential
            )
            for entry in computation.ch99_list
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
    entry = payload.entry_date or date.today()
    country = payload.country_of_origin.strip().upper()
    melt_origin = (
        payload.pour_country.strip().upper()
        if payload.pour_country
        else None
    )
    measurements = _normalize_measurements(payload)
    import_value_amount = measurements.get("usd")

    try:
        with db_connection() as conn:
            record = fetch_basic_hts_record(conn, payload.hts_code.strip())
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
        basic_computations = compute_basic_duty(
            canonical_hts,
            measurements,
            country_of_origin=country,
            special_rate_of_duty=record.get("special_rate_of_duty"),
        )
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
    meta_info = _build_meta_info(general_basic_computation, unit_config_raw)
    basic_rate_payloads = [
        BasicRateComputationPayload.from_dataclass(computation)
        for computation in basic_computations
    ]
    section_301_result = compute_section301_duty(
        canonical_hts,
        country,
        entry,
        import_value=import_value_amount,
    )
    section_232_result = compute_section232_duty(
        canonical_hts,
        country,
        entry,
        melt_origin,
        import_value=import_value_amount,
        measurements=measurements,
        steel_percentage=payload.steel_percentage,
        aluminum_percentage=payload.aluminum_percentage,
    )
    modules = _build_modules(section_301_result, section_232_result)
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
        encrypted = encrypt_payload(envelope_payload)
    except (EncryptionConfigError, EncryptionExecutionError) as exc:
        logger.exception("Failed to encrypt response payload.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return EncryptedEnvelope.model_validate(encrypted)
