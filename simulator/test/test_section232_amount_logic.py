from decimal import Decimal
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import simulator.section232_rate as section232_rate


def _make_entry(heading: str, rate: str) -> section232_rate.Section232Ch99:
    return section232_rate.Section232Ch99(
        ch99_id=heading,
        alias=heading,
        general_rate=Decimal(rate),
        ch99_description="test",
    )


def test_compute_amount_without_component_shares(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(section232_rate, "get_unit_config", lambda _hts: {})
    entries = [_make_entry("9903.81.87", "25")]
    amount, missing = section232_rate._compute_section232_amount(
        "test-hts",
        entries,
        import_value=Decimal("1000"),
        measurements={},
    )
    assert amount == Decimal("250")
    assert missing == []
    assert entries[0].amount == Decimal("250")


def test_compute_amount_with_component_shares(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        section232_rate,
        "get_unit_config",
        lambda _hts: {
            "steel_percentage": "steel_percentage",
            "aluminum_percentage": "aluminum_percentage",
        },
    )
    entries = [
        _make_entry("9903.81.87", "25"),
        _make_entry("9903.85.05", "10"),
    ]
    measurements = {
        "steel_percentage": Decimal("0.6"),
        "aluminum_percentage": Decimal("0.4"),
    }
    amount, missing = section232_rate._compute_section232_amount(
        "test-hts",
        entries,
        import_value=Decimal("1000"),
        measurements=measurements,
    )
    assert amount == Decimal("190")
    assert missing == []
    assert entries[0].amount == Decimal("150")
    assert entries[1].amount == Decimal("40")


def test_compute_amount_missing_required_measurement(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        section232_rate,
        "get_unit_config",
        lambda _hts: {"steel_percentage": "steel_percentage"},
    )
    entries = [_make_entry("9903.81.87", "25")]
    amount, missing = section232_rate._compute_section232_amount(
        "test-hts",
        entries,
        import_value=Decimal("1000"),
        measurements={},
    )
    assert amount == Decimal("0")
    assert missing == ["steel_percentage"]
    assert entries[0].amount == Decimal("0")


def test_compute_amount_uses_payload_share_without_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(section232_rate, "get_unit_config", lambda _hts: {})
    entries = [_make_entry("9903.81.87", "25")]
    amount, missing = section232_rate._compute_section232_amount(
        "test-hts",
        entries,
        import_value=Decimal("1000"),
        measurements={"steel_percentage": Decimal("85")},
    )
    assert amount == Decimal("212.5")
    assert missing == []
    assert entries[0].amount == Decimal("212.5")
