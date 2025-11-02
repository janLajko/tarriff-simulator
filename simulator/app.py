"""Tariff simulation FastAPI application."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Optional

import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from .basic_hts_rate import (
    BasicRateComputation,
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


class RequestEcho(BaseModel):
    hts_code: str
    country_of_origin: str
    entry_date: date
    date_of_landing: Optional[date] = None
    import_value: Optional[Decimal] = None
    measurements: Dict[str, Decimal] = Field(default_factory=dict)


class Ch99Entry(BaseModel):
    ch99_id: str
    alias: Optional[str] = None
    general_rate: Decimal
    ch99_description: Optional[str] = None


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


class SimulationResponse(BaseModel):
    request: RequestEcho
    modules: List[TariffModule]
    meta: MetaInfo


class EncryptedEnvelope(BaseModel):
    version: int
    algorithm: str
    ciphertext: str
    nonce: str
    issued_at: datetime
    key_id: Optional[str] = None


app = FastAPI(title="Tariff Simulator API", version="0.3.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _normalize_measurements(payload: SimulationRequest) -> Dict[str, Decimal]:
    measurements: Dict[str, Decimal] = {
        str(key).strip().lower(): value for key, value in payload.measurements.items()
    }
    if payload.import_value is not None:
        measurements.setdefault("usd", payload.import_value)
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


def _build_modules(section301: Section301Computation) -> List[TariffModule]:
    ch99_entries = [
        Ch99Entry(
            ch99_id=entry.ch99_id,
            alias=entry.alias,
            general_rate=entry.general_rate,
            ch99_description=entry.ch99_description,
        )
        for entry in section301.ch99_list
    ]
    module = TariffModule(
        module_id=section301.module_id,
        module_name=section301.module_name,
        ch99_list=ch99_entries,
        notes=list(section301.notes),
        applicable=section301.applicable,
        amount=section301.amount,
        currency=section301.currency,
        rate=section301.rate,
    )
    return [module]


def _build_request_echo(
    payload: SimulationRequest,
    normalized_measurements: Dict[str, Decimal],
    hts_code: str,
    entry_date: date,
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
    )


@app.post("/simulate", response_model=EncryptedEnvelope)
def simulate_tariff(payload: SimulationRequest) -> EncryptedEnvelope:
    entry = payload.entry_date or date.today()
    country = payload.country_of_origin.strip().upper()
    measurements = _normalize_measurements(payload)

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
        basic_computation = compute_basic_duty(
            canonical_hts,
            measurements,
        )
    except MissingMeasurementError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(exc),
                "missing_measurements": exc.missing,
            },
        ) from exc

    unit_config_raw = get_unit_config(canonical_hts) or {}
    meta_info = _build_meta_info(basic_computation, unit_config_raw)
    section_301_result = compute_section301_duty(canonical_hts, country, entry)
    modules = _build_modules(section_301_result)
    request_echo = _build_request_echo(payload, measurements, canonical_hts, entry)

    envelope_payload = SimulationResponse(
        request=request_echo,
        modules=modules,
        meta=meta_info,
    ).model_dump(mode="json")

    try:
        encrypted = encrypt_payload(envelope_payload)
    except (EncryptionConfigError, EncryptionExecutionError) as exc:
        logger.exception("Failed to encrypt response payload.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return EncryptedEnvelope.model_validate(encrypted)
