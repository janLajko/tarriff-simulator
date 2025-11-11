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

from simulator import section232_rate


pytestmark = [
    pytest.mark.skipif(
        section232_rate.psycopg2 is None,
        reason="psycopg2 is required for integration tests",
    ),
    pytest.mark.skipif(
        not os.getenv("DATABASE_DSN"),
        reason="DATABASE_DSN environment variable must be set for integration tests",
    ),
]

CSV_DIR = Path(__file__).parent / "232"
CSV_PATTERN = "99038192_*.csv"
DEFAULT_ENTRY_DATE = date(2024, 7, 1)

_NORMALIZE = section232_rate._normalize_hts


def _parse_entry_date(raw: str) -> date:
    value = (raw or "").strip()
    if not value:
        return DEFAULT_ENTRY_DATE
    return datetime.strptime(value, "%Y/%m/%d").date()


def _gather_fixture_rows() -> list[dict]:
    rows: list[dict] = []
    for csv_path in sorted(CSV_DIR.glob(CSV_PATTERN)):
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for index, raw in enumerate(reader, start=2):
                duty_str = (raw.get("duty") or "").strip()
                if not duty_str:
                    raise ValueError(f"Missing duty value in {csv_path.name}:{index}")
                input_hts = (raw.get("inputhts") or "").strip()
                if not input_hts:
                    raise ValueError(f"Missing input HTS in {csv_path.name}:{index}")
                heading = (raw.get("hts") or "").strip()
                if not heading:
                    raise ValueError(f"Missing duty heading in {csv_path.name}:{index}")

                info = {
                    "id": f"{csv_path.name}:{index}",
                    "file": csv_path,
                    "line": index,
                    "input_hts": input_hts,
                    "normalized_input": _NORMALIZE(input_hts),
                    "entry_date": _parse_entry_date(raw.get("entrydate", "")),
                    "country": (raw.get("country") or "").strip().upper(),
                    "melt_origin": (raw.get("melt_origin") or "").strip().upper() or None,
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
            info["melt_origin"],
        )
        duties_by_key[key].add(info["duty"])
    return {key for key, duties in duties_by_key.items() if len(duties) > 1}


AMBIGUOUS_KEYS = _ambiguous_keys(RAW_ROWS)
for info in RAW_ROWS:
    key = (
        info["normalized_input"],
        info["entry_date"],
        info["country"],
        info["melt_origin"],
    )
    info["ambiguous"] = key in AMBIGUOUS_KEYS


def _fixture_id(info: dict) -> str:
    return info["id"]


@pytest.mark.parametrize("row_info", RAW_ROWS, ids=_fixture_id)
def test_compute_section232_duty_against_fixture(row_info: dict):
    if row_info.get("ambiguous"):
        pytest.skip(
            f"{row_info['id']} has multiple expected duty outcomes for identical inputs."
        )

    result = section232_rate.compute_section232_duty(
        row_info["input_hts"],
        row_info["country"],
        row_info["entry_date"],
        row_info["melt_origin"],
    )

    print(f"{row_info['id']}: {result}")

    expected_duty = row_info["duty"]
    expected_heading = row_info["expected_heading"]

    total_rate = sum((ch.general_rate for ch in result.ch99_list), Decimal("0"))
    assert total_rate == expected_duty, f"{row_info['id']}: unexpected summed duty rate"

    if not result.ch99_list:
        assert expected_duty == 0, f"{row_info['id']}: expected zero duty when no CH99 entries"
        assert result.applicable is False, f"{row_info['id']}: expected module to be inactive"
        assert result.rate in (None, "0%"), f"{row_info['id']}: unexpected rate display"
        return

    assert result.applicable is True, f"{row_info['id']}: expected duty to apply"
    assert any(
        ch.ch99_id == expected_heading for ch in result.ch99_list
    ), f"{row_info['id']}: missing expected CH99 heading {expected_heading}"
    assert result.rate is not None, f"{row_info['id']}: expected rate display"
    actual_rate = Decimal(result.rate.rstrip("%"))
    assert actual_rate == expected_duty, f"{row_info['id']}: unexpected duty rate"
