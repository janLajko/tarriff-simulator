"""Utilities for looking up and computing basic HTS duty amounts."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import psycopg2
import psycopg2.extras as pgx

DATA_DIR = Path(__file__).resolve().parent
HTS_UNIT_PATH = DATA_DIR / "hts_unit.json"
SPI_PATH = DATA_DIR / "spi.json"
CURRENCY = "USD"
GENERAL_RATE_CASE_ID = "GENERAL_RATE"
SPECIAL_RATE_FREE_CASE_ID = "SPECIAL_RATE_FREE"


class BasicRateError(Exception):
    """Base error for HTS basic rate lookups."""


class MissingMeasurementError(BasicRateError):
    """Raised when required measurement inputs are missing."""

    def __init__(self, missing: Iterable[str]) -> None:
        missing_list = sorted(set(str(item) for item in missing))
        super().__init__(f"Missing required measurement values: {', '.join(missing_list)}")
        self.missing = missing_list


class FormulaEvaluationError(BasicRateError):
    """Raised when a configured formula cannot be evaluated."""


@dataclass
class BasicRateComputation:
    """Represents the result of evaluating the basic duty formula."""

    case_id: str
    calculated: bool
    amount: Decimal
    currency: str
    formula: Optional[str]
    required_units: list[str]
    provided_inputs: Dict[str, Decimal]
    notes: list[str]


def _decimal(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError) as exc:  # pragma: no cover - defensive
        raise FormulaEvaluationError(f"Unable to coerce '{value}' to Decimal.") from exc


def _load_unit_config() -> Dict[str, Dict[str, object]]:
    if not HTS_UNIT_PATH.exists():  # pragma: no cover - defensive
        raise FileNotFoundError(f"HTS unit configuration not found: {HTS_UNIT_PATH}")
    with HTS_UNIT_PATH.open("r", encoding="utf-8") as handle:
        raw_entries = json.load(handle)

    config: Dict[str, Dict[str, object]] = {}

    def _ingest(hts: object, value: object) -> None:
        if not isinstance(value, dict):
            return
        code = str(hts or "").strip()
        if not code:
            return
        units = value.get("unit") or []
        normalized_units = [str(item).strip().lower() for item in units if str(item).strip()]
        config[code] = {
            "units": normalized_units,
            "formula": (value.get("formula") or "").strip(),
            "general_rate_of_duty": value.get("general_rate_of_duty"),
            "hts_general": value.get("hts_general"),
            "notes": value.get("notes"),
        }

    if isinstance(raw_entries, dict):
        for hts, value in raw_entries.items():
            _ingest(hts, value)
    elif isinstance(raw_entries, list):
        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            for hts, value in entry.items():
                _ingest(hts, value)

    return config


@lru_cache(maxsize=1)
def _load_spi_programs() -> Dict[str, tuple[str, frozenset[str]]]:
    if not SPI_PATH.exists():
        return {}
    with SPI_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    entries = payload.get("special_program_indicators") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return {}

    program_index: Dict[str, tuple[str, frozenset[str]]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        code = str(entry.get("code") or "").strip().upper()
        if not code:
            continue
        name = str(entry.get("program_name") or "").strip()
        raw_countries = entry.get("countries")
        iso_codes: set[str] = set()
        if isinstance(raw_countries, list):
            for country in raw_countries:
                if not isinstance(country, dict):
                    continue
                iso_alpha2 = country.get("iso_alpha2")
                if not isinstance(iso_alpha2, str):
                    continue
                normalized_iso = iso_alpha2.strip().upper()
                if normalized_iso:
                    iso_codes.add(normalized_iso)
        program_index[code] = (name, frozenset(iso_codes))
    return program_index


_FREE_CODES_PATTERN = re.compile(r"free\s*\(([^()]*)\)", re.IGNORECASE)
_PROGRAM_CODE_CLEANER = re.compile(r"[^A-Z0-9+*]")


def _normalize_country_code(country: Optional[str]) -> Optional[str]:
    if not country or not isinstance(country, str):
        return None
    cleaned = "".join(ch for ch in country.strip().upper() if ch.isalnum())
    if len(cleaned) < 2:
        return None
    return cleaned


def _extract_free_program_codes(special_rate: str) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    for match in _FREE_CODES_PATTERN.finditer(special_rate):
        raw_block = match.group(1)
        for raw_code in raw_block.split(","):
            cleaned = _PROGRAM_CODE_CLEANER.sub("", raw_code.upper())
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                codes.append(cleaned)
    return codes


def _resolve_program_countries(program_code: str, program_index: Dict[str, tuple[str, frozenset[str]]]) -> set[str]:
    normalized_code = str(program_code or "").strip().upper()
    if not normalized_code:
        return set()

    candidates = [normalized_code]
    trimmed = normalized_code.rstrip("*+")
    if trimmed and trimmed != normalized_code:
        candidates.append(trimmed)

    for candidate in candidates:
        program = program_index.get(candidate)
        if program:
            countries = program[1]
            if countries:
                return set(countries)

    if len(normalized_code) == 2 and normalized_code.isalpha():
        return {normalized_code}
    return set()


def _describe_program(program_code: str, program_index: Dict[str, tuple[str, frozenset[str]]]) -> str:
    normalized_code = str(program_code or "").strip().upper()
    if not normalized_code:
        return ""
    program = program_index.get(normalized_code)
    if program and program[0]:
        return f"{normalized_code} ({program[0]})"
    trimmed = normalized_code.rstrip("*+")
    if trimmed and trimmed != normalized_code:
        fallback = program_index.get(trimmed)
        if fallback and fallback[0]:
            return f"{normalized_code} ({fallback[0]})"
    return normalized_code


def _build_special_free_note(country: Optional[str], special_rate: Optional[str]) -> Optional[str]:
    normalized_country = _normalize_country_code(country)
    if not normalized_country:
        return None

    if not special_rate or not isinstance(special_rate, str):
        return None

    if "free" not in special_rate.lower():
        return None

    codes = _extract_free_program_codes(special_rate)
    if not codes:
        return None

    program_index = _load_spi_programs()

    matched_codes: list[str] = []
    for code in codes:
        countries = _resolve_program_countries(code, program_index)
        if normalized_country in countries:
            matched_codes.append(code)

    if not matched_codes:
        return None

    descriptions = []
    seen_desc: set[str] = set()
    for code in matched_codes:
        description = _describe_program(code, program_index) or code
        if description not in seen_desc:
            seen_desc.add(description)
            descriptions.append(description)

    detail = ", ".join(descriptions)
    return (
        f"Special rate 'Free' applies for country {normalized_country}; "
        f"qualifying program codes: {detail}."
    )


@lru_cache(maxsize=1)
def get_unit_config_map() -> Dict[str, Dict[str, object]]:
    """Return the cached mapping of HTS numbers to unit/formula metadata."""

    return _load_unit_config()


def get_unit_config(hts_number: str) -> Optional[Dict[str, object]]:
    """Return unit configuration for the given HTS number if available."""

    config = get_unit_config_map()
    return config.get(hts_number)


def _evaluate_formula(formula: str, values: Mapping[str, Decimal]) -> Decimal:
    """Safely evaluate an arithmetic formula using Decimal values."""

    if not formula:
        return Decimal("0")
    tree = ast.parse(formula, mode="eval")

    def _eval(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            raise FormulaEvaluationError(f"Unsupported operator: {ast.dump(node.op)}")
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise FormulaEvaluationError(f"Unsupported unary operator: {ast.dump(node.op)}")
        if isinstance(node, ast.Constant):
            return _decimal(node.value)
        if isinstance(node, ast.Name):
            key = node.id.lower()
            if key not in values:
                raise FormulaEvaluationError(f"Unknown variable referenced in formula: {node.id}")
            return values[key]
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise FormulaEvaluationError(f"Unsupported callable in formula: {ast.dump(node.func)}")
            function_name = node.func.id.lower()
            if node.keywords:
                raise FormulaEvaluationError("Keyword arguments are not supported in formulas.")
            args = [_eval(arg) for arg in node.args]
            if function_name == "max":
                if not args:
                    raise FormulaEvaluationError("Function 'max' requires at least one argument.")
                return max(args)
            # TODO: extend supported functions if new formulas require them.
            raise FormulaEvaluationError(f"Unsupported function call in formula: {node.func.id}")
        raise FormulaEvaluationError(f"Unsupported expression element: {ast.dump(node)}")

    return _eval(tree)


def _finalize_basic_rate_results(
    general_result: BasicRateComputation,
    *,
    special_applies: bool,
    special_note: Optional[str],
) -> list[BasicRateComputation]:
    """Attach a special-rate entry when applicable and return all computations."""

    results: list[BasicRateComputation] = [general_result]
    if not special_applies:
        return results

    special_notes = list(general_result.notes)
    if special_note and special_note not in special_notes:
        special_notes.append(special_note)

    special_result = BasicRateComputation(
        case_id=SPECIAL_RATE_FREE_CASE_ID,
        calculated=False,
        amount=Decimal("0"),
        currency=CURRENCY,
        formula=None,
        required_units=[],
        provided_inputs={},
        notes=special_notes,
    )
    results.append(special_result)
    return results


def compute_basic_duty(
    hts_number: str,
    measurements: Mapping[str, object],
    *,
    country_of_origin: Optional[str] = None,
    special_rate_of_duty: Optional[str] = None,
) -> list[BasicRateComputation]:
    """Compute the basic duty for the supplied HTS number.

    Parameters
    ----------
    hts_number:
        The HTS code submitted by the client.
    measurements:
        Mapping of unit name → numeric value supplied by the client. Keys are
        matched case-insensitively against the configured unit list.
    country_of_origin:
        Optional ISO-3166 alpha-2 country code for evaluating special program
        eligibility when a duty-free rate is available.
    special_rate_of_duty:
        Raw ``Special Rate of Duty`` text used to determine whether the
        provided country qualifies for a duty-free program.
    """

    config = get_unit_config(hts_number)
    notes: list[str] = []
    special_note = _build_special_free_note(country_of_origin, special_rate_of_duty)
    special_applies = special_note is not None

    if not config:
        notes.append("HTS not present in configuration; no basic duty applies(" + hts_number + ")")
        general_result = BasicRateComputation(
            case_id=GENERAL_RATE_CASE_ID,
            calculated=False,
            amount=Decimal("0"),
            currency=CURRENCY,
            formula=None,
            required_units=[],
            provided_inputs={},
            notes=list(notes),
        )
        return _finalize_basic_rate_results(
            general_result,
            special_applies=special_applies,
            special_note=special_note,
        )

    required_units: list[str] = list(config.get("units") or [])
    formula = str(config.get("formula") or "").strip()
    config_notes = config.get("notes")
    if config_notes:
        notes.append(str(config_notes))

    normalized_inputs = {
        str(key).strip().lower(): _decimal(value) for key, value in (measurements or {}).items()
    }

    if not required_units:
        # No units required; treat constant formula if numeric else assume zero.
        if not formula:
            general_result = BasicRateComputation(
                case_id=GENERAL_RATE_CASE_ID,
                calculated=False,
                amount=Decimal("0"),
                currency=CURRENCY,
                formula=None,
                required_units=[],
                provided_inputs={},
                notes=list(notes),
            )
            return _finalize_basic_rate_results(
                general_result,
                special_applies=special_applies,
                special_note=special_note,
            )
        try:
            amount = _evaluate_formula(formula, {})
            general_result = BasicRateComputation(
                case_id=GENERAL_RATE_CASE_ID,
                calculated=True,
                amount=amount,
                currency=CURRENCY,
                formula=formula,
                required_units=[],
                provided_inputs={},
                notes=list(notes),
            )
            return _finalize_basic_rate_results(
                general_result,
                special_applies=special_applies,
                special_note=special_note,
            )
        except FormulaEvaluationError:
            notes.append("Formula requires manual handling.")
            general_result = BasicRateComputation(
                case_id=GENERAL_RATE_CASE_ID,
                calculated=False,
                amount=Decimal("0"),
                currency=CURRENCY,
                formula=formula,
                required_units=[],
                provided_inputs={},
                notes=list(notes),
            )
            return _finalize_basic_rate_results(
                general_result,
                special_applies=special_applies,
                special_note=special_note,
            )

    missing_units = [unit for unit in required_units if unit not in normalized_inputs]
    if missing_units:
        if special_applies:
            missing_error = MissingMeasurementError(missing_units)
            notes.append(str(missing_error))
            available_inputs = {
                unit: normalized_inputs[unit] for unit in required_units if unit in normalized_inputs
            }
            general_result = BasicRateComputation(
                case_id=GENERAL_RATE_CASE_ID,
                calculated=False,
                amount=Decimal("0"),
                currency=CURRENCY,
                formula=formula or None,
                required_units=required_units,
                provided_inputs=available_inputs,
                notes=list(notes),
            )
            return _finalize_basic_rate_results(
                general_result,
                special_applies=True,
                special_note=special_note,
            )
        raise MissingMeasurementError(missing_units)

    ordered_inputs = {unit: normalized_inputs[unit] for unit in required_units}

    if not formula or formula.lower() in {"see notes", "free"}:
        notes.append("Formula unavailable for automatic calculation.")
        general_result = BasicRateComputation(
            case_id=GENERAL_RATE_CASE_ID,
            calculated=False,
            amount=Decimal("0"),
            currency=CURRENCY,
            formula=formula or None,
            required_units=required_units,
            provided_inputs=ordered_inputs,
            notes=list(notes),
        )
        return _finalize_basic_rate_results(
            general_result,
            special_applies=special_applies,
            special_note=special_note,
        )

    try:
        amount = _evaluate_formula(formula, ordered_inputs)
    except FormulaEvaluationError as exc:
        notes.append(str(exc))
        general_result = BasicRateComputation(
            case_id=GENERAL_RATE_CASE_ID,
            calculated=False,
            amount=Decimal("0"),
            currency=CURRENCY,
            formula=formula,
            required_units=required_units,
            provided_inputs=ordered_inputs,
            notes=list(notes),
        )
        return _finalize_basic_rate_results(
            general_result,
            special_applies=special_applies,
            special_note=special_note,
        )

    general_result = BasicRateComputation(
        case_id=GENERAL_RATE_CASE_ID,
        calculated=True,
        amount=amount,
        currency=CURRENCY,
        formula=formula,
        required_units=required_units,
        provided_inputs=ordered_inputs,
        notes=list(notes),
    )
    return _finalize_basic_rate_results(
        general_result,
        special_applies=special_applies,
        special_note=special_note,
    )


def fetch_basic_hts_record(conn: psycopg2.extensions.connection, hts_number: str) -> Optional[Dict[str, object]]:
    """Fetch the base HTS metadata row from PostgreSQL."""

    with conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
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
            WHERE hts_number = %s
            ORDER BY row_order
            LIMIT 1
            """,
            (hts_number,),
        )
        row = cur.fetchone()

        if row:
            return dict(row)

        digits_only = "".join(ch for ch in hts_number if ch.isdigit())
        if not digits_only:
            return None

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
            WHERE regexp_replace(hts_number, '\\D', '', 'g') = %s
            ORDER BY row_order
            LIMIT 1
            """,
            (digits_only,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
