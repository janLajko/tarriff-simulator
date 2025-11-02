"""Utilities for looking up and computing basic HTS duty amounts."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import psycopg2
import psycopg2.extras as pgx

DATA_DIR = Path(__file__).resolve().parent
HTS_UNIT_PATH = DATA_DIR / "hts_unit.json"
CURRENCY = "USD"


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
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        for hts, value in entry.items():
            if not isinstance(value, dict):
                continue
            units = value.get("unit") or []
            normalized_units = [str(item).strip().lower() for item in units if str(item).strip()]
            config[hts] = {
                "units": normalized_units,
                "formula": (value.get("formula") or "").strip(),
                "general_rate_of_duty": value.get("general_rate_of_duty"),
                "hts_general": value.get("hts_general"),
                "notes": value.get("notes"),
            }
    return config


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
        raise FormulaEvaluationError(f"Unsupported expression element: {ast.dump(node)}")

    return _eval(tree)


def compute_basic_duty(hts_number: str, measurements: Mapping[str, object]) -> BasicRateComputation:
    """Compute the basic duty for the supplied HTS number.

    Parameters
    ----------
    hts_number:
        The HTS code submitted by the client.
    measurements:
        Mapping of unit name → numeric value supplied by the client. Keys are
        matched case-insensitively against the configured unit list.
    """

    config = get_unit_config(hts_number)
    normalized_inputs = {str(key).strip().lower(): _decimal(value) for key, value in (measurements or {}).items()}
    notes: list[str] = []

    if not config:
        notes.append("HTS not present in configuration; no basic duty applies.")
        return BasicRateComputation(
            calculated=False,
            amount=Decimal("0"),
            currency=CURRENCY,
            formula=None,
            required_units=[],
            provided_inputs={},
            notes=notes,
        )

    required_units: list[str] = list(config.get("units") or [])
    formula = str(config.get("formula") or "").strip()
    config_notes = config.get("notes")
    if config_notes:
        notes.append(str(config_notes))

    if not required_units:
        # No units required; treat constant formula if numeric else assume zero.
        if not formula:
            return BasicRateComputation(
                calculated=False,
                amount=Decimal("0"),
                currency=CURRENCY,
                formula=None,
                required_units=[],
                provided_inputs={},
                notes=notes,
            )
        try:
            amount = _evaluate_formula(formula, {})
            return BasicRateComputation(
                calculated=True,
                amount=amount,
                currency=CURRENCY,
                formula=formula,
                required_units=[],
                provided_inputs={},
                notes=notes,
            )
        except FormulaEvaluationError:
            notes.append("Formula requires manual handling.")
            return BasicRateComputation(
                calculated=False,
                amount=Decimal("0"),
                currency=CURRENCY,
                formula=formula,
                required_units=[],
                provided_inputs={},
                notes=notes,
            )

    missing_units = [unit for unit in required_units if unit not in normalized_inputs]
    if missing_units:
        raise MissingMeasurementError(missing_units)

    ordered_inputs = {unit: normalized_inputs[unit] for unit in required_units}

    if not formula or formula.lower() in {"see notes", "free"}:
        notes.append("Formula unavailable for automatic calculation.")
        return BasicRateComputation(
            calculated=False,
            amount=Decimal("0"),
            currency=CURRENCY,
            formula=formula or None,
            required_units=required_units,
            provided_inputs=ordered_inputs,
            notes=notes,
        )

    try:
        amount = _evaluate_formula(formula, ordered_inputs)
    except FormulaEvaluationError as exc:
        notes.append(str(exc))
        return BasicRateComputation(
            calculated=False,
            amount=Decimal("0"),
            currency=CURRENCY,
            formula=formula,
            required_units=required_units,
            provided_inputs=ordered_inputs,
            notes=notes,
        )

    return BasicRateComputation(
        calculated=True,
        amount=amount,
        currency=CURRENCY,
        formula=formula,
        required_units=required_units,
        provided_inputs=ordered_inputs,
        notes=notes,
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
