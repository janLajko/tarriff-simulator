import csv
import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from simulator import basic_hts_rate
except ModuleNotFoundError as exc:  # pragma: no cover - protective guard
    if exc.name == "psycopg2":
        pytest.skip(
            "psycopg2 must be installed to run basic HTS rate integration tests.",
            allow_module_level=True,
        )
    raise


pytestmark = [
    pytest.mark.skipif(
        basic_hts_rate.psycopg2 is None,
        reason="psycopg2 is required for integration tests",
    ),
    pytest.mark.skipif(
        not os.getenv("DATABASE_DSN"),
        reason="DATABASE_DSN environment variable must be set for integration tests",
    ),
]

CSV_PATH = Path(__file__).parent / "basic_rate_test.csv"


def _load_cases() -> List[dict]:
    cases: List[dict] = []
    with CSV_PATH.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for index, row in enumerate(reader, start=2):
            raw_payload = row.get("input", "")
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse payload in {CSV_PATH.name}:{index}") from exc

            hts_code = (row.get("hts_code") or payload.get("hts_code") or "").strip()
            if not hts_code:
                raise ValueError(f"Missing HTS code in {CSV_PATH.name}:{index}")

            expected_raw = str(row.get("expected_result") or "").strip()
            if not expected_raw:
                raise ValueError(f"Missing expected_result in {CSV_PATH.name}:{index}")
            expected_amount = Decimal(expected_raw)

            cases.append(
                {
                    "id": f"{CSV_PATH.name}:{index}",
                    "hts_code": hts_code,
                    "payload": payload,
                    "expected": expected_amount,
                }
            )
    return cases


CSV_CASES = _load_cases()


@pytest.fixture(scope="module")
def db_connection():
    dsn = os.environ["DATABASE_DSN"]
    conn = basic_hts_rate.psycopg2.connect(dsn)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="module")
def special_rate_lookup(db_connection):
    lookup: Dict[str, Optional[str]] = {}
    with db_connection.cursor() as cur:
        cur.execute("SELECT hts_number, special_rate_of_duty FROM hts_codes")
        for hts_number, special_rate in cur.fetchall():
            if hts_number:
                normalized = str(hts_number).strip()
                if normalized:
                    normalized_rate = _normalize_special_rate(special_rate)
                    lookup[normalized] = normalized_rate
                    digits_only = "".join(ch for ch in normalized if ch.isdigit())
                    if digits_only and digits_only not in lookup:
                        lookup[digits_only] = normalized_rate
    return lookup


def _find_special_rate(hts_code: str, lookup: Dict[str, Optional[str]]) -> Optional[str]:
    rate = lookup.get(hts_code)
    if rate is not None:
        return rate
    digits_only = "".join(ch for ch in hts_code if ch.isdigit())
    if digits_only:
        return lookup.get(digits_only)
    return None


def _normalize_country(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().upper()
    return cleaned or None


def _prepare_measurements(payload: dict) -> Dict[str, object]:
    raw_measurements = payload.get("measurements")
    normalized: Dict[str, object] = {}
    if isinstance(raw_measurements, dict):
        normalized.update(
            {str(key).strip().lower(): value for key, value in raw_measurements.items()}
        )
    import_value = payload.get("import_value")
    if "usd" not in normalized and import_value is not None:
        normalized["usd"] = import_value
    return normalized


def _normalize_special_rate(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return stripped
    return str(value)


@pytest.mark.parametrize("case", CSV_CASES, ids=lambda info: info["id"])
def test_compute_basic_duty_against_csv(
    case: dict, special_rate_lookup: Dict[str, Optional[str]]
):
    payload = case["payload"]
    measurements = _prepare_measurements(payload)
    country = _normalize_country(payload.get("country_of_origin"))

    special_rate = _find_special_rate(case["hts_code"], special_rate_lookup)

    result = basic_hts_rate.compute_basic_duty(
        case["hts_code"],
        measurements,
        country_of_origin=country,
        special_rate_of_duty=special_rate,
    )

    print(f"{case['id']}: {result}")

    assert result.amount == case["expected"], f"{case['id']}: unexpected duty amount"
    assert result.currency == basic_hts_rate.CURRENCY, f"{case['id']}: unexpected currency"
    if case["expected"] > 0:
        assert result.calculated is True, f"{case['id']}: expected calculated duty"
