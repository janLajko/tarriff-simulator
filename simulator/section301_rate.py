"""Placeholder Section 301 duty computation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import List, Optional


@dataclass
class Section301Ch99:
    ch99_id: str
    alias: str
    general_rate: Decimal
    ch99_description: str


@dataclass
class Section301Computation:
    module_id: str
    module_name: str
    applicable: bool
    amount: Decimal
    currency: str
    rate: Optional[str]
    ch99_list: List[Section301Ch99] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def compute_section301_duty(
    hts_number: str,
    country_of_origin: str,
    entry_date: date,
) -> Section301Computation:
    """Mocked Section 301 computation.

    This placeholder marks the HTS code as not currently subject to Section 301
    tariffs until the dedicated data pipeline is implemented.
    """

    notes = [
        "Section 301 duty calculation is mocked; integrate real logic when data is available."
    ]
    return Section301Computation(
        module_id="301",
        module_name="Section 301 Tariffs",
        applicable=False,
        amount=Decimal("0"),
        currency="USD",
        rate=None,
        notes=notes,
        ch99_list=[
            Section301Ch99(
                ch99_id="9903.88.01",
                alias="List 1",
                general_rate=Decimal("0.25"),
                ch99_description=(
                    "Products of China subject to additional duties as provided in U.S. note 20(b) "
                    "to this subchapter and listed in Annex A."
                ),
            ),
        ],
    )
