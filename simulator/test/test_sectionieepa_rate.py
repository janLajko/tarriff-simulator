import csv
import os
import sys
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from simulator import sectionieepa_rate


pytestmark = [
    pytest.mark.skipif(
        sectionieepa_rate.psycopg2 is None,
        reason="psycopg2 is required for integration tests",
    ),
    pytest.mark.skipif(
        not os.getenv("DATABASE_DSN"),
        reason="DATABASE_DSN environment variable must be set for integration tests",
    ),
]

CSV_PATH = Path(__file__).parent / "ieepa" / "ieepa_test_data.csv"
LOG_PATH = Path(__file__).parent / "ieepa" / "ieepa_test_output.log"
DEFAULT_ENTRY_DATE = date(2025, 1, 1)

_NORMALIZE = sectionieepa_rate._normalize_hts


def _parse_entry_date(raw: str) -> date:
    value = (raw or "").strip()
    if not value:
        return DEFAULT_ENTRY_DATE
    return datetime.strptime(value, "%Y/%m/%d").date()


def _coerce_decimal(raw: str) -> Decimal:
    value = (raw or "").strip()
    if not value:
        return Decimal("0")
    return Decimal(value)


def _gather_fixture_rows() -> list[dict]:
    rows: list[dict] = []
    with CSV_PATH.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for index, raw in enumerate(reader, start=2):
            duty_str = (raw.get("duty") or "").strip()
            if not duty_str:
                raise ValueError(f"Missing duty value in {CSV_PATH.name}:{index}")
            input_hts = (raw.get("inputhts") or "").strip()
            if not input_hts:
                raise ValueError(f"Missing input HTS in {CSV_PATH.name}:{index}")
            heading = (raw.get("ieepahts") or "").strip()
            if not heading:
                raise ValueError(f"Missing duty heading in {CSV_PATH.name}:{index}")

            info = {
                "id": f"{CSV_PATH.name}:{index}",
                "file": CSV_PATH,
                "line": index,
                "input_hts": input_hts,
                "normalized_input": _NORMALIZE(input_hts),
                "entry_date": _parse_entry_date(raw.get("entrydate", "")),
                "country": (raw.get("country") or "").strip().upper(),
                "steel_percentage": _coerce_decimal(raw.get("steel_percentage", "")),
                "aluminum_percentage": _coerce_decimal(raw.get("aluminum_percentage", "")),
                "expected_heading": heading,
                "duty": Decimal(duty_str),
            }
            rows.append(info)
    return rows


RAW_ROWS = _gather_fixture_rows()


def _ambiguous_keys(rows: list[dict]) -> set[tuple]:
    duties_by_key: defaultdict = defaultdict(set)
    for info in rows:
        key = (
            info["normalized_input"],
            info["entry_date"],
            info["country"],
            info["steel_percentage"],
            info["aluminum_percentage"],
        )
        duties_by_key[key].add(info["duty"])
    return {key for key, duties in duties_by_key.items() if len(duties) > 1}


AMBIGUOUS_KEYS = _ambiguous_keys(RAW_ROWS)
for info in RAW_ROWS:
    key = (
        info["normalized_input"],
        info["entry_date"],
        info["country"],
        info["steel_percentage"],
        info["aluminum_percentage"],
    )
    info["ambiguous"] = key in AMBIGUOUS_KEYS


def _fixture_id(info: dict) -> str:
    return info["id"]


@pytest.mark.parametrize("row_info", RAW_ROWS, ids=_fixture_id)
def test_compute_sectionieepa_duty_against_fixture(row_info: dict):
    if row_info.get("ambiguous"):
        pytest.skip(
            f"{row_info['id']} has multiple expected duty outcomes for identical inputs."
        )

    measurements = {
        "steel_percentage": row_info["steel_percentage"],
        "aluminum_percentage": row_info["aluminum_percentage"],
    }

    result = sectionieepa_rate.compute_sectionieepa_duty(
        hts_number=row_info["input_hts"],
        country_of_origin=row_info["country"],
        entry_date=row_info["entry_date"],
        import_value=None,
        melt_pour_origin_iso2=None,
        measurements=measurements,
    )

    # Persist and print the raw computation result for inspection.
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_line = f"{row_info['id']}: {result}\n"
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(log_line)
    print(log_line.strip())

    expected_duty = row_info["duty"]
    expected_heading = row_info["expected_heading"]

    total_rate = sum((ch.general_rate for ch in result.ch99_list), Decimal("0"))
    assert total_rate == expected_duty, f"{row_info['id']}: unexpected summed duty rate"

    if expected_heading.upper() == "NONE":
        assert expected_duty == 0, f"{row_info['id']}: expected zero duty when heading is NONE"
        assert result.applicable is False, f"{row_info['id']}: expected module to be inactive"
        assert not result.ch99_list, f"{row_info['id']}: expected no CH99 entries"
        assert result.rate in (None, "0%"), f"{row_info['id']}: unexpected rate display"
        return

    assert result.applicable is True, f"{row_info['id']}: expected duty to apply"
    assert any(
        ch.ch99_id == expected_heading for ch in result.ch99_list
    ), f"{row_info['id']}: missing expected CH99 heading {expected_heading}"
    assert result.rate is not None, f"{row_info['id']}: expected rate display"
    actual_rate = Decimal(result.rate.rstrip("%"))
    assert actual_rate == expected_duty, f"{row_info['id']}: unexpected duty rate"
