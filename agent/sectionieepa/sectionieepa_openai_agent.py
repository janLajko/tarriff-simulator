#!/usr/bin/env python3
"""Section ieepa processing agent.

This module implements the workflow described in ``agent.txt``:

1. Read the Section ieepa HTS headings (e.g. 9903.88.01) from ``hts_codes``.
2. Insert or update records in ``sieepa_measures`` for those headings.
3. Use an LLM to structure the descriptions and notes into include / exclude
   scopes with their effective periods.
4. Persist the scopes in ``sieepa_scope`` and link them to measures via
   ``sieepa_scope_measure_map``.

The code is intentionally straightforward—no elaborate abstractions—so the data
flow is easy to follow when extending or debugging.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

import requests

try:
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor
    from psycopg2 import errors as pg_errors
except ImportError as exc:  # pragma: no cover - allows linting without psycopg2
    psycopg2 = None  # type: ignore
    Json = None  # type: ignore
    RealDictCursor = None  # type: ignore
    _PSYCOPG2_IMPORT_ERROR = exc
    pg_errors = None  # type: ignore
else:
    _PSYCOPG2_IMPORT_ERROR = None


LOGGER = logging.getLogger("sectionieepa_agent")

BULK_QUERY_CHUNK = 50
T = TypeVar("T")

DEFAULT_HEADINGS = [
    "9903.01.25"
]

LLM_MEASURE_PROMPT = """You are a legal text structure analyzer for HTSUS Section ieepa derivative steel measures.
From the following text, extract:
1. **Except sections** – heading numbers listed after phrases like “Except as provided in headings …”.
2. **Include sections** – notes or subheadings mentioned after phrases such as “as provided for in …”.
3. **Effective period** – explicit start / end dates (e.g., “Effective with respect to entries on or after …” or “through …”). Dates that refer only to transportation or entry-cutoff timing (e.g., “goods loaded before August 7, 2025” or “entered before October 5, 2025”) do **not** set the measure’s start date; treat them as eligibility criteria unless the text explicitly says the measure becomes effective on that date.
4. **country_iso2** – ISO-2 code for the products covered by the measure. Use null when the measure applies to every country other than the excluded origins.
5. **melt_pour_origin_iso2** – ISO-2 code for where the steel must be melted and poured (e.g., “US” when the text says “melted and poured in the United States”). Use null if not stated.
6. **origin_exclude_iso2** – an array of ISO-2 codes for origin countries explicitly excluded (e.g., ["UK"]). When the text implies “all countries except the United Kingdom” for derivative products made from U.S.-melted steel, output ["UK"] even if the exclusion is implicit.
7. **is_potential** – true only when the text describes a conditional or potential scenario (e.g., treatment of goods that might later enter commerce, or goods admitted to a foreign trade zone before a cut-off time). False when the measure is in force without such contingencies. When `is_potential` is true, leave `effective_period.end_date` null unless the text explicitly states when the measure itself expires. Dates that merely reference entry/admission timing do **not** impose an end date.

Input text:
\"\"\"{description}\"\"\"

import!
1. When the text prints a range such as "9903.02.74-9903.02.77" or "9903.02.74–9903.02.77", expand it into each discrete heading (9903.02.74, 9903.02.75, 9903.02.76, 9903.02.77).


Return JSON only. Use this schema:
{{
  "except": [],
  "include": ["note16(m)"],
  "effective_period": {{
    "start_date": "June 15, 2024",
    "end_date": "November 29, 2025"
  }},
  "country_iso2": "UK",
  "melt_pour_origin_iso2": "US",
  "origin_exclude_iso2": ["UK"],
  "is_potential": false
}}

Examples:
1. Input:
\"\"\"Derivative iron or steel products provided for in the tariff subheadings enumerated in subdivision subdivisions (m), (n), (t) or (u) of note 16 to this subchapter, where the derivative iron or steel product was processed in another country from steel articles that were melted and poured in the United States.\"\"\"
Output:
{{
  "except": [],
  "include": ["note16(m)", "note16(n)", "note16(t)", "note16(u)"],
  "effective_period": {{"start_date": null, "end_date": null}},
  "country_iso2": null,
  "melt_pour_origin_iso2": "US",
  "origin_exclude_iso2": ["UK"],
  "is_potential": false
}}

2. Input:
\"\"\"Except for derivative iron or steel products described in headings 9903.81.96, 9903.81.97 or 9903.81.98, products of iron or steel of the United Kingdom provided for in the tariff headings or subheadings enumerated in subdivision (q) of note 16 to this subchapter.\"\"\"
Output:
{{
  "except": ["9903.81.96", "9903.81.97", "9903.81.98"],
  "include": ["note16(q)"],
  "effective_period": {{"start_date": null, "end_date": null}},
  "country_iso2": "UK",
  "melt_pour_origin_iso2": null,
  "origin_exclude_iso2": [],
  "is_potential": false
}}

3. Input:
\"\"\"Except as provided in headings 9903.81.91 or 9903.81.92, derivative products of iron or steel, as specified in subdivisions (l) and (m) of note 16 to this subchapter, admitted to a U.S. foreign trade zone under “privileged foreign status” as defined by 19 CFR 146.41, prior to 12:01 a.m. eastern daylight time on June 4, 2025.\"\"\"
Output:
{{
  "except": ["9903.81.91", "9903.81.92"],
  "include": ["note16(l)", "note16(m)"],
  "effective_period": {{"start_date": null, "end_date": null}},
  "country_iso2": null,
  "melt_pour_origin_iso2": null,
  "origin_exclude_iso2": [],
  "is_potential": true
}}

4. Input:
\"\"\"Except for goods loaded onto a vessel at the port of loading and in transit on the final mode of transit before 12:01 a.m. eastern daylight time on August 7, 2025, and entered for consumption or withdrawn from warehouse for consumption before 12:01 a.m. eastern daylight time on October 5, 2025, products provided for in heading 9903.02.02 are subject to the additional duty.\"\"\"
Output:
{{
  "except": [],
  "include": [],
  "effective_period": {{"start_date": null, "end_date": null}},
  "country_iso2": null,
  "melt_pour_origin_iso2": null,
  "origin_exclude_iso2": [],
  "is_potential": true
}}
"""

LLM_NOTE_PROMPT = """You are a structured extractor for HTSUS Section ieepa notes.
Output JSON only. No inference. No paraphrasing. Include only codes that are explicitly printed in the input text.

Objective:
From the input legal note text, produce three outputs:
1. **input_htscode** – copy exactly from `context.input_htscode` (a list provided by the caller). Do NOT infer, paraphrase, or add/remove codes. If this field is missing, return an empty list.
2. **scope** – headings or subheadings that are explicitly stated to be covered (inside the range) by the measure. Do **not** include anything that appears only inside "except"/"exclusion" phrasing.
3. **except** – headings or subheadings explicitly excluded (phrases such as "except … provided for in …").

Each object in scope or except must match the `sieepa_scope` table schema:

- keys                 // comma-separated exact heading or HTS8 codes (e.g. "0203.29.20,0203.29.40,0206.10.00") when conditions are identical
- key                  // single exact heading or HTS8 code (e.g. "9903.88.05" or "8501.10.40") when used alone
- key_type             // one of: "heading" (use for 4-digit headings and any 6-digit references), "hts8" (8-digit subheading), "hts10" (10-digit statistical reporting number)
- country_iso2         // ISO-2 country code mentioned in the text (e.g. "UK"), else null
- source_label         // e.g. "note20(a)" or "note20(b)-exclusion"
- effective_start_date // ISO date; use context.fallback_start_date if not stated
- effective_end_date   // ISO date or null

Extraction rules:
A. Set **input_htscode** equal to `context.input_htscode`. Never invent or summarize values.

B. Identify all **scope** items:
   - Include every HTS reference that is explicitly printed (4-digit headings, 6-digit subheadings, 8-digit HTS codes, or 10-digit statistical reporting numbers) under "applies to…" or similar language.
   - When the text prints a range such as "9903.02.74-9903.02.77" or "9903.02.74–9903.02.77", expand it into each discrete heading (9903.02.74, 9903.02.75, 9903.02.76, 9903.02.77).
   - Do NOT include `input_htscode` again.
   - **When multiple HTS codes share identical conditions (same key_type, country_iso2, source_label, effective_start_date, effective_end_date), combine them using "keys" field with comma-separated values instead of creating separate objects.**

C. Identify all **except** items:
   - Look for any statement that exempts certain products/headings from the additional duties that would otherwise apply to `input_htscode`.
   - Common patterns include:
     * "Except as provided in heading(s) X..."
     * "The additional duties imposed by heading(s) [including input_htscode] shall not apply to [products] provided for in heading(s) X"
     * "As provided in heading X, the additional duties... shall not apply to products classified in..."
   
   - **Understanding Full vs Partial Exemptions:**
     * **FULL EXEMPTION** (add to `except`): The products classified in the referenced heading are COMPLETELY exempt from the additional duties.
       - The entire product/article in that heading is not subject to the additional duties.
       - Example: "The additional duties shall not apply to medium- and heavy-duty vehicle parts provided for in headings 9903.74.08 and 9903.74.09"
         → Products in 9903.74.08 and 9903.74.09 are FULLY exempt
     
     * **PARTIAL EXEMPTION** (do NOT add to `except`): Only a specific portion/component of the products in the referenced heading is exempt, while the remaining portion is still subject to duties.
       - The text clarifies that duties still apply to some part/content of the product.
       - Example: "The additional duties shall not apply to products of iron or steel provided for in headings 9903.81.87 and 9903.81.88, but such additional duties shall apply to the non-steel content of such products"
         → Products in 9903.81.87 and 9903.81.88 are only PARTIALLY exempt (steel content exempt, non-steel content still taxed)
       - Example: "The additional duties shall not apply to the declared value of the steel content of the derivative iron or steel products provided for in headings 9903.81.89-9903.81.93, but shall apply to the non-steel content"
         → Products in 9903.81.89-9903.81.93 are only PARTIALLY exempt
   
   - **Decision Rule**: 
     * Use your understanding of the legal language to determine: Are the products in the referenced heading COMPLETELY exempt from the additional duties, or are they only PARTIALLY exempt?
     * Only add headings to `except` if they represent FULL exemptions.
     * Do NOT add headings that represent partial exemptions (where some portion is still taxable).

   - When the language says "As provided in heading 9903.01.32 ... shall not apply to products classified in the following subheadings", treat the enumerated HTS codes as the actual `except` entries and do not output the referencing heading itself (e.g., 9903.01.32).
   
   - **Checking Applicability**: 
     * For an exemption to apply to `input_htscode`, the text must indicate that the exemption applies to the additional duties imposed by a list of headings that includes `input_htscode`.
     * Ranges like "9903.02.01-9903.02.73" include ALL codes numerically between start and end (inclusive).
     * Example: If input_htscode is "9903.02.02" and text says "duties imposed by headings 9903.02.01-9903.02.73 shall not apply to X", then X is an exemption for 9903.02.02.
   
   - Mark their source_label with "-exclusion".
   - Apply the same combination rule as scope items when conditions are identical.

D. Ignore carve-outs that clearly belong to other headings/programs (e.g., "As provided in 9903.01.26 ... products of Canada"). Unless the sentence explicitly ties the relief to `input_htscode`, do not add those references to `scope` or `except`.

E. Set country_iso2 to the ISO-2 code for the country named in the text (e.g. "UK" for "products of the United Kingdom"). If no country is mentioned, leave it null.

F. Only populate effective_start_date and effective_end_date when explicit "effective" or date ranges appear in the text.  
   - Example: "effective January 1, 2023" → effective_start_date = "2023-01-01"  
   - Example: "effective January 1, 2019 through December 31, 2019" → effective_start_date = "2019-01-01", effective_end_date = "2019-12-31"  
   - Entry/admission timing clauses (e.g., goods admitted to a foreign trade zone before 12:01 a.m. EDT on March 12, 2025) do **not** impose an effective_end_date; they only describe which entries qualify.  
   - If no explicit "effective" or date reference appears, leave both fields null. Do not use fallback_start_date.

G. Remove duplicates.

H. When multiple note blocks are provided, use `context.note_labels` (same order as the input blocks) for their labels. `context.primary_note_labels` lists the notes that actually triggered this extraction. `context.supporting_note_labels` lists subdivisions that were included only because a primary note referenced them (e.g., "subdivision (l)"). For supporting notes:
   - Treat any HTS codes printed inside those blocks as exclusions for the referencing primary note unless the supporting block clearly introduces a different heading of its own.
   - Use the primary note label with "-exclusion" for the `source_label` (e.g., `note16(m)-exclusion`).
   - Do NOT add supporting-block codes to the scope of the current heading unless the text explicitly says they apply to that heading.

I. If a supporting block explicitly states a different heading (e.g., "The rates … in heading 9903.81.89"), you may output its own scope entries but keep the source_label equal to that supporting note label.

Context (provided by caller):
// context.input_htscode → base headings provided by the caller; copy directly into "input_htscode"
// context.note_labels → labels for each input block (in order)
// context.primary_note_labels → subset that actually triggered extraction
// context.supporting_note_labels → subdivisions referenced only to describe exclusions for the primaries
{context_json}

Input (one or more note blocks, each beginning with its note label on the first line):
\"\"\"{note_text}\"\"\"

Output JSON format (note the use of "keys" for combined codes and "key" for single codes):
{{
  "input_htscode": ["9903.88.01"],
  "scope": [
    {{ "keys": "0203.29.20,0203.29.40,0206.10.00,0208.10.00,0208.90.20,0208.90.25,0210.19.00", "key_type": "hts8", "country_iso2": "CN", "source_label": "note20(b)", "effective_start_date": "1900-01-01", "effective_end_date": null }},
    {{ "key": "2845.30.00", "key_type": "hts8", "country_iso2": "CN", "source_label": "note20(b)", "effective_start_date": "1900-01-01", "effective_end_date": null }}
  ],
  "except": [
    {{ "key": "9903.88.05", "key_type": "heading", "country_iso2": "CN", "source_label": "note20(a)-exclusion", "effective_start_date": "1900-01-01", "effective_end_date": null }}
  ]
}}
"""

NOTE_TOKEN_RE = re.compile(r"\(\s*([^)]+?)\s*\)")


def strip_json_code_block(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


@dataclass
class MeasureAnalysis:
    include: List[str]
    exclude: List[str]
    effective_start: Optional[date]
    effective_end: Optional[date]
    country_iso2: Optional[str]
    melt_pour_origin_iso2: Optional[str]
    origin_exclude_iso2: Optional[List[str]]
    is_potential: bool


@dataclass
class ScopeRecord:
    key: str
    key_type: str
    country_iso2: Optional[str]
    source_label: Optional[str]
    effective_start_date: date
    effective_end_date: Optional[date]


def parse_date(value: Optional[str]) -> Optional[date]:
    """Parse various date formats used in HTS texts."""
    if not value:
        return None
    value = value.strip()
    if not value:
        return None

    patterns = [
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y-%m-%d",
        "%m/%d/%Y",
    ]
    for pattern in patterns:
        try:
            return datetime.strptime(value, pattern).date()
        except ValueError:
            continue
    return None


def normalize_note_label(raw_label: str) -> str:
    """Normalize a note label into the canonical form used in hts_notes."""
    if not raw_label:
        raise ValueError("note label cannot be empty")
    s = raw_label.strip()
    if not s.lower().startswith("note"):
        s = "note " + s
    if "(" not in s:
        parts = s.split()
        head = parts[0].lower()
        tail = "".join(f"({p.strip()})" for p in parts[1:])
        s = head + tail
    tokens = list(NOTE_TOKEN_RE.findall(s))
    prefix_match = re.search(r"note\s*([0-9ivxlcdm]+)", s, re.IGNORECASE)
    if prefix_match:
        leading = prefix_match.group(1).strip()
        if leading and leading not in tokens:
            tokens.insert(0, leading)
    return "note" + "".join(f"({t})" for t in tokens if t)


def classify_code_type(code: str) -> str:
    """Heuristically classify a code string as heading or hts8."""
    digits_only = code.replace(".", "")
    if code.startswith("99"):
        return "heading"
    if len(digits_only) == 8:
        return "hts8"
    if len(digits_only) == 4:
        return "heading"
    return "heading"


class Section232Database:
    """Minimal database helper around psycopg2."""

    def __init__(self, dsn: str):
        if psycopg2 is None:
            raise RuntimeError(
                "psycopg2 is required to use Section232Database"
            ) from _PSYCOPG2_IMPORT_ERROR
        self._dsn = dsn
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False

    def close(self) -> None:
        self._conn.close()

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def _reconnect(self) -> None:
        try:
            if self._conn and not self._conn.closed:
                self._conn.close()
        except Exception:
            pass
        self._conn = psycopg2.connect(self._dsn)
        self._conn.autocommit = False

    def _run_with_reconnect(self, operation: Callable[[], T]) -> T:
        last_exc: Optional[Exception] = None
        for attempt in range(2):
            try:
                return operation()
            except psycopg2.OperationalError as exc:  # type: ignore[attr-defined]
                last_exc = exc
                message = str(exc).lower()
                if "server closed the connection unexpectedly" in message:
                    LOGGER.warning("Database connection lost; reconnecting (attempt %s)", attempt + 1)
                    self._reconnect()
                    continue
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError("Operation failed without raising an exception")

    def fetch_hts_code(self, heading: str) -> Optional[Dict[str, Any]]:
        query = """
            SELECT hts_number, description, status, additional_duties
            FROM hts_codes
            WHERE hts_number = %s
            ORDER BY row_order
            LIMIT 1
        """
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (heading,))
            return cur.fetchone()

    def ensure_measure(
        self,
        heading: str,
        country_iso2: Optional[str],
        ad_valorem_rate: Decimal,
        value_basis: str,
        melt_pour_origin_iso2: Optional[str],
        origin_exclude_iso2: Optional[Sequence[str]],
        notes: Optional[Dict[str, Any]],
        start_date: date,
        end_date: Optional[date],
        is_potential: bool,
    ) -> Optional[int]:
        query_select = """
            SELECT id FROM sieepa_measures
            WHERE heading = %s
              AND COALESCE(country_iso2, '') = COALESCE(%s, '')
              AND ad_valorem_rate = %s
              AND value_basis = %s
              AND COALESCE(melt_pour_origin_iso2, '') = COALESCE(%s, '')
              AND COALESCE(origin_exclude_iso2, ARRAY[]::text[]) =
                  COALESCE(%s::text[], ARRAY[]::text[])
              AND effective_start_date = %s
              AND COALESCE(effective_end_date, DATE '9999-12-31') =
                  COALESCE(%s, DATE '9999-12-31')
              AND is_potential = %s
            LIMIT 1
        """
        with self._conn.cursor() as cur:
            cur.execute(
                query_select,
                (
                    heading,
                    country_iso2,
                    ad_valorem_rate,
                    value_basis,
                    melt_pour_origin_iso2,
                    origin_exclude_iso2,
                    start_date,
                    end_date,
                    is_potential,
                ),
            )
            row = cur.fetchone()
            if row:
                return row[0]

        query_insert = """
            INSERT INTO sieepa_measures
            (heading, country_iso2, ad_valorem_rate, value_basis,
             melt_pour_origin_iso2, origin_exclude_iso2, notes,
             effective_start_date, effective_end_date, is_potential)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        payload_notes = Json(notes) if notes is not None else None
        with self._conn.cursor() as cur:
            cur.execute("SAVEPOINT sp_measure_insert")
            try:
                cur.execute(
                    query_insert,
                    (
                        heading,
                        country_iso2,
                        ad_valorem_rate,
                        value_basis,
                        melt_pour_origin_iso2,
                        origin_exclude_iso2,
                        payload_notes,
                        start_date,
                        end_date,
                        is_potential,
                    ),
                )
                measure_id = cur.fetchone()[0]
            except Exception as exc:
                if pg_errors and isinstance(exc, pg_errors.UniqueViolation):
                    cur.execute("ROLLBACK TO SAVEPOINT sp_measure_insert")
                    cur.execute("RELEASE SAVEPOINT sp_measure_insert")
                    LOGGER.info("Duplicate measure detected for %s; skipping insert", heading)
                    with self._conn.cursor() as cur_lookup:
                        cur_lookup.execute(
                            query_select,
                            (
                                heading,
                                country_iso2,
                                ad_valorem_rate,
                                value_basis,
                                melt_pour_origin_iso2,
                                origin_exclude_iso2,
                                start_date,
                                end_date,
                                is_potential,
                            ),
                        )
                        row = cur_lookup.fetchone()
                        if row:
                            return row[0]
                    LOGGER.warning("Failed to locate existing measure for %s after duplicate conflict", heading)
                    return None
                raise
            else:
                cur.execute("RELEASE SAVEPOINT sp_measure_insert")
        LOGGER.info("Created measure %s for heading %s", measure_id, heading)
        return measure_id

    def find_existing_measure_id(self, heading: str, country_iso2: Optional[str]) -> Optional[int]:
        query = """
            SELECT id
            FROM sieepa_measures
            WHERE heading = %s
              AND COALESCE(country_iso2, '') = COALESCE(%s, '')
            ORDER BY effective_start_date DESC, id DESC
            LIMIT 1
        """
        with self._conn.cursor() as cur:
            cur.execute(query, (heading, country_iso2))
            row = cur.fetchone()
            return row[0] if row else None

    def ensure_scope(self, record: ScopeRecord) -> int:
        query_select = """
            SELECT id FROM sieepa_scope
            WHERE key = %s
              AND key_type = %s
              AND COALESCE(country_iso2, '') = COALESCE(%s, '')
              AND effective_start_date = %s
              AND COALESCE(effective_end_date, DATE '9999-12-31') =
                  COALESCE(%s, DATE '9999-12-31')
            LIMIT 1
        """
        with self._conn.cursor() as cur:
            cur.execute(
                query_select,
                (
                    record.key,
                    record.key_type,
                    record.country_iso2,
                    record.effective_start_date,
                    record.effective_end_date,
                ),
            )
            row = cur.fetchone()
            if row:
                return row[0]

        query_insert = """
            INSERT INTO sieepa_scope
            (key, key_type, country_iso2, source_label,
             effective_start_date, effective_end_date)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        with self._conn.cursor() as cur:
            cur.execute("SAVEPOINT sp_scope_insert")
            try:
                cur.execute(
                    query_insert,
                    (
                        record.key,
                        record.key_type,
                        record.country_iso2,
                        record.source_label,
                        record.effective_start_date,
                        record.effective_end_date,
                    ),
                )
                scope_id = cur.fetchone()[0]
            except Exception as exc:
                if pg_errors and isinstance(exc, pg_errors.UniqueViolation):
                    cur.execute("ROLLBACK TO SAVEPOINT sp_scope_insert")
                    cur.execute("RELEASE SAVEPOINT sp_scope_insert")
                    LOGGER.info("Duplicate scope detected for key %s; skipping insert", record.key)
                    with self._conn.cursor() as cur_lookup:
                        cur_lookup.execute(
                            query_select,
                            (
                                record.key,
                                record.key_type,
                                record.country_iso2,
                                record.effective_start_date,
                                record.effective_end_date,
                            ),
                        )
                        row = cur_lookup.fetchone()
                        if row:
                            return row[0]
                    LOGGER.warning("Failed to locate existing scope for key %s after duplicate conflict", record.key)
                    return None
                raise
            else:
                cur.execute("RELEASE SAVEPOINT sp_scope_insert")
        LOGGER.info("Created scope %s for key %s", scope_id, record.key)
        return scope_id

    def ensure_scopes_bulk(self, records: Sequence[ScopeRecord]) -> Dict[Tuple[str, str, str, date, Optional[date]], int]:
        sentinel_end = date(9999, 12, 31)
        unique: Dict[Tuple[str, str, str, date, Optional[date]], ScopeRecord] = {}
        for record in records:
            key = (
                record.key,
                record.key_type,
                (record.country_iso2 or ""),
                record.effective_start_date,
                record.effective_end_date,
            )
            unique[key] = record
        if not unique:
            return {}

        def _build_conditions(items: List[Tuple[str, str, str, date, Optional[date]]]) -> Tuple[str, List[Any]]:
            clauses: List[str] = []
            params: List[Any] = []
            for entry in items:
                clauses.append(
                    "(key = %s AND key_type = %s AND COALESCE(country_iso2, '') = COALESCE(%s, '') "
                    "AND effective_start_date = %s AND COALESCE(effective_end_date, DATE '9999-12-31') = "
                    "COALESCE(%s, DATE '9999-12-31'))"
                )
                params.extend(entry)
            return " OR ".join(clauses), params

        existing: Dict[Tuple[str, str, str, date, Optional[date]], int] = {}
        keys_list = list(unique.keys())
        for chunk in chunked(keys_list, BULK_QUERY_CHUNK):
            where_clause, params = _build_conditions(list(chunk))
            query = (
                "SELECT id, key, key_type, COALESCE(country_iso2, '') AS country_iso2, "
                "effective_start_date, COALESCE(effective_end_date, DATE '9999-12-31') AS effective_end_date "
                "FROM sieepa_scope WHERE " + where_clause
            )

            def _fetch_existing() -> List[Tuple[Any, ...]]:
                with self._conn.cursor() as cur:
                    cur.execute(query, params)
                    return cur.fetchall()

            rows = self._run_with_reconnect(_fetch_existing)
            for row in rows:
                signature = (
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    None if row[5] == sentinel_end else row[5],
                )
                existing[signature] = row[0]

        missing = [sig for sig in keys_list if sig not in existing]
        if missing:
            for chunk in chunked(missing, BULK_QUERY_CHUNK):
                values = []
                insert_params: List[Any] = []
                for sig in chunk:
                    record = unique[sig]
                    values.append("(%s, %s, %s, %s, %s, %s)")
                    insert_params.extend(
                        [
                            record.key,
                            record.key_type,
                            record.country_iso2,
                            record.source_label,
                            record.effective_start_date,
                            record.effective_end_date,
                        ]
                    )
                insert_query = (
                    "INSERT INTO sieepa_scope "
                    "(key, key_type, country_iso2, source_label, effective_start_date, effective_end_date) "
                    "VALUES " + ", ".join(values) +
                    " RETURNING id, key, key_type, COALESCE(country_iso2, '') AS country_iso2, "
                    "effective_start_date, COALESCE(effective_end_date, DATE '9999-12-31') AS effective_end_date"
                )

                def _insert_missing() -> List[Tuple[Any, ...]]:
                    with self._conn.cursor() as cur:
                        cur.execute(insert_query, insert_params)
                        return cur.fetchall()

                rows = self._run_with_reconnect(_insert_missing)
                for row in rows:
                    signature = (
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        None if row[5] == sentinel_end else row[5],
                    )
                    existing[signature] = row[0]
                    LOGGER.info("Created scope %s for key %s", row[0], row[1])

        return existing

    def ensure_scope_measure_map(
        self,
        scope_id: int,
        measure_id: int,
        relation: str,
        note_label: Optional[str],
        text_criteria: Optional[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> int:
        start_date = None
        end_date = None
        query_select = """
            SELECT id FROM sieepa_scope_measure_map
            WHERE scope_id = %s
              AND measure_id = %s
              AND relation = %s
              AND COALESCE(note_label, '') = COALESCE(%s, '')
              AND COALESCE(text_criteria, '') = COALESCE(%s, '')
              AND effective_start_date = %s
              AND COALESCE(effective_end_date, DATE '9999-12-31') =
                  COALESCE(%s, DATE '9999-12-31')
            LIMIT 1
        """
        with self._conn.cursor() as cur:
            cur.execute(
                query_select,
                (
                    scope_id,
                    measure_id,
                    relation,
                    note_label,
                    text_criteria,
                    start_date,
                    end_date,
                ),
            )
            row = cur.fetchone()
            if row:
                return row[0]

        query_insert = """
            INSERT INTO sieepa_scope_measure_map
            (scope_id, measure_id, relation, note_label, text_criteria,
             effective_start_date, effective_end_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        with self._conn.cursor() as cur:
            cur.execute("SAVEPOINT sp_scope_map_insert")
            try:
                cur.execute(
                    query_insert,
                    (
                        scope_id,
                        measure_id,
                        relation,
                        note_label,
                        text_criteria,
                        start_date,
                        end_date,
                    ),
                )
                map_id = cur.fetchone()[0]
            except Exception as exc:
                if pg_errors and isinstance(exc, pg_errors.UniqueViolation):
                    cur.execute("ROLLBACK TO SAVEPOINT sp_scope_map_insert")
                    cur.execute("RELEASE SAVEPOINT sp_scope_map_insert")
                    LOGGER.info(
                        "Duplicate scope↔measure relation skipped (scope=%s, measure=%s, relation=%s)",
                        scope_id,
                        measure_id,
                        relation,
                    )
                    with self._conn.cursor() as cur_lookup:
                        cur_lookup.execute(
                            query_select,
                            (
                                scope_id,
                                measure_id,
                                relation,
                                note_label,
                                text_criteria,
                                start_date,
                                end_date,
                            ),
                        )
                        row = cur_lookup.fetchone()
                        if row:
                            return row[0]
                    LOGGER.warning(
                        "Failed to locate existing scope↔measure relation after duplicate conflict (scope=%s, measure=%s)",
                        scope_id,
                        measure_id,
                    )
                    return None
                raise
            else:
                cur.execute("RELEASE SAVEPOINT sp_scope_map_insert")
        LOGGER.info(
            "Created scope↔measure relation %s (scope=%s, measure=%s, %s)",
            map_id,
            scope_id,
            measure_id,
            relation,
        )
        return map_id

    def ensure_scope_measure_map_bulk(self, entries: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int, str, str, str, Optional[date], Optional[date]], int]:
        sentinel_end = date(9999, 12, 31)
        unique: Dict[Tuple[int, int, str, str, str, Optional[date], Optional[date]], Dict[str, Any]] = {}
        for entry in entries:
            key = (
                entry["scope_id"],
                entry["measure_id"],
                entry["relation"],
                (entry.get("note_label") or ""),
                (entry.get("text_criteria") or ""),
                entry.get("start_date"),
                entry.get("end_date"),
            )
            unique[key] = entry
        if not unique:
            return {}

        def _build_conditions(items: List[Tuple[int, int, str, str, str, Optional[date], Optional[date]]]) -> Tuple[str, List[Any]]:
            clauses: List[str] = []
            params: List[Any] = []
            for entry in items:
                clauses.append(
                    "(scope_id = %s AND measure_id = %s AND relation = %s AND "
                    "COALESCE(note_label, '') = COALESCE(%s, '') AND COALESCE(text_criteria, '') = COALESCE(%s, '') AND "
                    "effective_start_date = %s AND COALESCE(effective_end_date, DATE '9999-12-31') = COALESCE(%s, DATE '9999-12-31'))"
                )
                params.extend(entry)
            return " OR ".join(clauses), params

        existing: Dict[Tuple[int, int, str, str, str, Optional[date], Optional[date]], int] = {}
        keys_list = list(unique.keys())
        for chunk in chunked(keys_list, BULK_QUERY_CHUNK):
            where_clause, params = _build_conditions(list(chunk))
            query = (
                "SELECT id, scope_id, measure_id, relation, COALESCE(note_label, '') AS note_label, "
                "COALESCE(text_criteria, '') AS text_criteria, effective_start_date, "
                "COALESCE(effective_end_date, DATE '9999-12-31') AS effective_end_date "
                "FROM sieepa_scope_measure_map WHERE " + where_clause
            )

            def _fetch_existing() -> List[Tuple[Any, ...]]:
                with self._conn.cursor() as cur:
                    cur.execute(query, params)
                    return cur.fetchall()

            rows = self._run_with_reconnect(_fetch_existing)
            for row in rows:
                signature = (
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    None if row[7] == sentinel_end else row[7],
                )
                existing[signature] = row[0]

        missing = [sig for sig in keys_list if sig not in existing]
        if missing:
            for chunk in chunked(missing, BULK_QUERY_CHUNK):
                values = []
                insert_params: List[Any] = []
                for sig in chunk:
                    entry = unique[sig]
                    values.append("(%s, %s, %s, %s, %s, %s, %s)")
                    insert_params.extend(
                        [
                            entry["scope_id"],
                            entry["measure_id"],
                            entry["relation"],
                            entry.get("note_label"),
                            entry.get("text_criteria"),
                            entry.get("start_date"),
                            entry.get("end_date"),
                        ]
                    )
                insert_query = (
                    "INSERT INTO sieepa_scope_measure_map "
                    "(scope_id, measure_id, relation, note_label, text_criteria, effective_start_date, effective_end_date) "
                    "VALUES " + ", ".join(values) +
                    " RETURNING id, scope_id, measure_id, relation, COALESCE(note_label, '') AS note_label, "
                    "COALESCE(text_criteria, '') AS text_criteria, effective_start_date, "
                    "COALESCE(effective_end_date, DATE '9999-12-31') AS effective_end_date"
                )

                def _insert_missing() -> List[Tuple[Any, ...]]:
                    with self._conn.cursor() as cur:
                        cur.execute(insert_query, insert_params)
                        return cur.fetchall()

                rows = self._run_with_reconnect(_insert_missing)
                for row in rows:
                    signature = (
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                        row[6],
                        None if row[7] == sentinel_end else row[7],
                    )
                    existing[signature] = row[0]
                    LOGGER.info(
                        "Created scope↔measure relation %s (scope=%s, measure=%s, %s)",
                        row[0],
                        row[1],
                        row[2],
                        row[3],
                    )

        return existing

    def fetch_note_rows(self, label: str) -> List[Dict[str, Any]]:
        """Fetch a note and all of its descendant rows, mirroring get_note()."""
        base_query = """
            SELECT chapter, subchapter, label, content, raw_html, path
            FROM hts_notes
            WHERE lower(label) = lower(%s)
            ORDER BY subchapter, array_length(path, 1), path
        """
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(base_query, (label,))
            anchor = cur.fetchone()
            if not anchor:
                return []

            cur.execute(
                """
                SELECT chapter, subchapter, label, content, raw_html, path
                FROM hts_notes
                WHERE chapter=%s AND subchapter=%s AND path[1:%s] = %s
                ORDER BY id, path
                """,
                (
                    anchor["chapter"],
                    anchor["subchapter"],
                    len(anchor["path"]),
                    anchor["path"],
                ),
            )
            rows = cur.fetchall()
            if not rows:
                # fallback to the anchor row so caller sees something
                return [anchor]
            return rows

    def fetch_scope_relations(
        self,
        measure_id: int,
        relation: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        base_query = """
            SELECT
                map.id AS map_id,
                map.relation,
                map.note_label,
                map.text_criteria,
                map.effective_start_date AS relation_start_date,
                map.effective_end_date AS relation_end_date,
                scope.id AS scope_id,
                scope.key,
                scope.key_type,
                scope.country_iso2,
                scope.source_label,
                scope.effective_start_date,
                scope.effective_end_date
            FROM sieepa_scope_measure_map AS map
            JOIN sieepa_scope AS scope ON scope.id = map.scope_id
            WHERE map.measure_id = %s
        """
        params: List[Any] = [measure_id]
        if relation:
            base_query += " AND map.relation = %s"
            params.append(relation)
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(base_query, params)
            return cur.fetchall()


class Section232LLM:
    """Thin wrapper around an LLM endpoint (default: OpenAI chat completions)."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 36000,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.timeout = timeout

    def _post(self, message: str) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY (or explicit api_key) is required for LLM calls")

        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a precise legal text parser. Respond with JSON only."},
                {"role": "user", "content": message},
            ],
        }
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - network error handling
            raise RuntimeError(f"LLM HTTP error: {exc} → {response.text}") from exc
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Unexpected LLM response structure: {data}") from exc

    def extract_measure(self, description: str) -> MeasureAnalysis:
        message = LLM_MEASURE_PROMPT.format(description=description.strip())
        raw = strip_json_code_block(self._post(message))
        try:
            LOGGER.info("measure %s", raw)
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to decode measure analysis JSON: {raw}") from exc

        include = [item.strip() for item in payload.get("include", []) if item and item.strip()]
        exclude = [item.strip() for item in payload.get("except", []) if item and item.strip()]
        eff_payload = payload.get("effective_period") or {}
        start = parse_date(eff_payload.get("start_date"))
        end = parse_date(eff_payload.get("end_date"))
        
        def _normalize_iso(value: Any) -> Optional[str]:
            if value is None:
                return None
            text = str(value).strip().upper()
            return text or None

        def _normalize_iso_list(value: Any) -> Optional[List[str]]:
            if value is None:
                return None
            candidates: List[str] = []
            if isinstance(value, str):
                candidates = [value]
            else:
                try:
                    iterator = iter(value)
                except TypeError:
                    iterator = iter(())
                for item in iterator:
                    text = str(item).strip()
                    if text:
                        candidates.append(text)
            normalized = [entry.strip().upper() for entry in candidates if entry.strip()]
            return normalized or None

        country_iso2 = _normalize_iso(payload.get("country_iso2"))
        melt_origin = _normalize_iso(payload.get("melt_pour_origin_iso2"))
        origin_exclude = _normalize_iso_list(payload.get("origin_exclude_iso2"))

        def _normalize_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y"}
            return False
        is_potential = _normalize_bool(payload.get("is_potential"))

        return MeasureAnalysis(
            include=include,
            exclude=exclude,
            effective_start=start,
            effective_end=end,
            country_iso2=country_iso2,
            melt_pour_origin_iso2=melt_origin,
            origin_exclude_iso2=origin_exclude,
            is_potential=is_potential,
        )

    def extract_note(self, note_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        context_json = json.dumps(context, ensure_ascii=False, sort_keys=True)
        message = LLM_NOTE_PROMPT.format(
            note_text=note_text.strip(),
            context_json=context_json,
        )
        LOGGER.info("message:%s", message)
        raw = strip_json_code_block(self._post(message))
        LOGGER.info("raw:%s", raw)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to decode note analysis JSON: {raw}") from exc
        return payload


class Section232Agent:
    """Coordinates DB + LLM workflow for Section 232 measures."""

    def __init__(
        self,
        db: Section232Database,
        llm: Section232LLM,
        *,
        country_iso2: Optional[str] = None,
    ):
        self.db = db
        self.llm = llm
        self.country_iso2 = country_iso2
        self._note_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        self._measure_cache: Dict[str, Optional[int]] = {}
        self._in_progress: set[str] = set()

    def run(self, headings: Sequence[str]) -> None:
        for heading in headings:
            self.process_heading(heading)

    def process_heading(self, heading: str) -> Optional[int]:
        normalized = heading.strip()
        if not normalized:
            return None
        if normalized in self._measure_cache:
            LOGGER.debug("Heading %s already processed; reusing measure id %s", normalized, self._measure_cache[normalized])
            return self._measure_cache[normalized]
        if normalized in self._in_progress:
            LOGGER.warning("Heading %s is already being processed; skipping to avoid recursion", normalized)
            return None

        LOGGER.info("Processing heading %s", normalized)
        self._in_progress.add(normalized)
        try:
            row = self.db.fetch_hts_code(normalized)
            if not row:
                LOGGER.warning("Heading %s not found in hts_codes; skipping", normalized)
                self._measure_cache[normalized] = None
                return None

            status = (row.get("status") or "").strip().lower()
            if status == "expired":
                LOGGER.info("Heading %s is marked expired; skipping", normalized)
                self._measure_cache[normalized] = None
                return None

            description = (row.get("description") or "").strip()
            if not description:
                LOGGER.warning("Heading %s has empty description; skipping", normalized)
                self._measure_cache[normalized] = None
                return None

            analysis = self.llm.extract_measure(description)
            # LOGGER.info("analysis : %s", analysis)
            start_date = analysis.effective_start or date(1900, 1, 1)
            end_date = analysis.effective_end
            if analysis.is_potential and end_date:
                LOGGER.info(
                    "Heading %s marked potential; clearing effective_end_date %s",
                    normalized,
                    end_date,
                )
                end_date = None

            general_rate_of_duty = (row.get("general_rate_of_duty") or "").strip()
            rate = self._derive_rate(general_rate_of_duty)

            measure_country_iso2 = analysis.country_iso2 or self.country_iso2

            existing_measure_id = self.db.find_existing_measure_id(normalized, measure_country_iso2)
            if existing_measure_id:
                LOGGER.info(
                    "Heading %s already present in sieepa_measures (id=%s); reusing existing scope",
                    normalized,
                    existing_measure_id,
                )
                self._measure_cache[normalized] = existing_measure_id
                return existing_measure_id

            measure_id = self.db.ensure_measure(
                heading=normalized,
                country_iso2=measure_country_iso2,
                ad_valorem_rate=rate,
                value_basis="customs_value",
                melt_pour_origin_iso2=analysis.melt_pour_origin_iso2,
                origin_exclude_iso2=analysis.origin_exclude_iso2,
                notes=None,
                start_date=start_date,
                end_date=end_date,
                is_potential=analysis.is_potential,
            )
            if not measure_id:
                LOGGER.warning("Failed to ensure measure for %s; skipping scope linkage", normalized)
                self.db.rollback()
                self._measure_cache[normalized] = None
                return None

            try:
                self._apply_scope_links(
                    heading=normalized,
                    measure_id=measure_id,
                    country_iso2=measure_country_iso2,
                    start_date=start_date,
                    end_date=end_date,
                    includes=analysis.include,
                    excludes=analysis.exclude,
                )
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise

            self._measure_cache[normalized] = measure_id
            return measure_id
        finally:
            self._in_progress.discard(normalized)

    def _derive_rate(self, description: str) -> Decimal:
        desc = description or ""
        match_add = re.search(
            r"the duty provided in the applicable subheading\s*(?:\+|plus)\s*(\d+(?:\.\d+)?)\s*(?:percent|%)",
            desc,
            re.IGNORECASE,
        )
        if match_add:
            return Decimal(match_add.group(1)).quantize(Decimal("0.001"))

        if re.search(r"the duty provided in the applicable subheading", desc, re.IGNORECASE):
            return Decimal("0.000")

        match = re.search(r"(\d+(?:\.\d+)?)\s*percent", desc, re.IGNORECASE)
        if match:
            return Decimal(match.group(1)).quantize(Decimal("0.001"))

        fallback_num = re.search(r"(\d+(?:\.\d+)?)", desc)
        if fallback_num:
            value = Decimal(fallback_num.group(1)).quantize(Decimal("0.001"))
            if value <= Decimal("999.000"):
                return value
            LOGGER.warning(
                "Derived fallback rate %s exceeds numeric precision; using default 25%%",
                value,
            )

        return Decimal("25.000")

    def _apply_scope_links(
        self,
        *,
        heading: str,
        measure_id: int,
        country_iso2: Optional[str],
        start_date: date,
        end_date: Optional[date],
        includes: Iterable[str],
        excludes: Iterable[str],
    ) -> None:
        include_notes, include_others = self._split_references(includes)
        exclude_notes, exclude_others = self._split_references(excludes)

        if include_notes:
            self._process_note_references(
                note_refs=include_notes,
                relation="include",
                heading=heading,
                measure_id=measure_id,
                country_iso2=country_iso2,
                fallback_start=start_date,
                fallback_end=end_date,
            )
        for code in include_others:
            self._handle_non_note_reference(
                reference=code,
                relation="include",
                heading=heading,
                measure_id=measure_id,
                country_iso2=country_iso2,
                start_date=start_date,
                end_date=end_date,
            )

        if exclude_notes:
            self._process_note_references(
                note_refs=exclude_notes,
                relation="exclude",
                heading=heading,
                measure_id=measure_id,
                country_iso2=country_iso2,
                fallback_start=start_date,
                fallback_end=end_date,
            )
        for code in exclude_others:
            self._handle_non_note_reference(
                reference=code,
                relation="exclude",
                heading=heading,
                measure_id=measure_id,
                country_iso2=country_iso2,
                start_date=start_date,
                end_date=end_date,
            )

    def _split_references(
        self, references: Iterable[str]
    ) -> Tuple[List[Tuple[str, str]], List[str]]:
        note_refs: List[Tuple[str, str]] = []
        other_refs: List[str] = []
        for ref in references:
            if not ref:
                continue
            cleaned = ref.strip()
            if not cleaned:
                continue
            if self._is_note_reference(cleaned):
                normalized = normalize_note_label(cleaned)
                note_refs.append((cleaned, normalized))
            else:
                other_refs.append(cleaned)
        return note_refs, other_refs

    @staticmethod
    def _is_note_reference(reference: str) -> bool:
        lower = reference.lower()
        if lower.startswith("note"):
            return True
        if NOTE_TOKEN_RE.search(reference):
            return True
        return False

    def _handle_non_note_reference(
        self,
        *,
        reference: str,
        relation: str,
        heading: str,
        measure_id: int,
        country_iso2: Optional[str],
        start_date: date,
        end_date: Optional[date],
    ) -> None:
        ref = reference.strip()
        if not ref:
            return
        if self._is_note_reference(ref):
            LOGGER.debug("Reference %s expected to be handled as note; skipping non-note handler", ref)
            return

        key_type = classify_code_type(ref)
        scope = ScopeRecord(
            key=ref,
            key_type=key_type,
            country_iso2=country_iso2,
            source_label=f"{heading}-description",
            effective_start_date=start_date,
            effective_end_date=end_date,
        )
        scope_id = self.db.ensure_scope(scope)
        if not scope_id:
            return
        self.db.ensure_scope_measure_map(
            scope_id=scope_id,
            measure_id=measure_id,
            relation=relation,
            note_label=scope.source_label,
            text_criteria=None,
        )
        if scope.key.startswith("99"):
            self.process_heading(scope.key)

    def _process_note_references(
        self,
        *,
        note_refs: Sequence[Tuple[str, str]],
        relation: str,
        heading: str,
        measure_id: int,
        country_iso2: Optional[str],
        fallback_start: date,
        fallback_end: Optional[date],
    ) -> None:
        unique_refs: List[Tuple[str, str]] = []
        seen: set[str] = set()
        for original, normalized in note_refs:
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_refs.append((original, normalized))

        note_blocks: List[Tuple[str, str, str]] = []
        for original, normalized in unique_refs:
            note_rows = self.db.fetch_note_rows(normalized)
            if not note_rows:
                LOGGER.warning("Note %s not found in hts_notes; skipping", normalized)
                continue
            LOGGER.info(
                "Loaded %d note rows for %s (first path=%s)",
                len(note_rows),
                normalized,
                note_rows[0].get("path") if note_rows else None,
            )
            block_lines: List[str] = []
            for row in note_rows:
                label_text = (row.get("label") or "").strip()
                content_text = (row.get("content") or "").strip()
                parts: List[str] = []
                if label_text:
                    parts.append(label_text)
                if content_text:
                    parts.append(content_text)
                line = " ".join(parts).strip()
                if line:
                    block_lines.append(line)

            combined_text = "\n".join(block_lines).strip()
            if not combined_text:
                LOGGER.warning("Note %s rows present but content is empty", normalized)
                continue
            note_blocks.append((original, normalized, combined_text))

        if not note_blocks:
            return

        original_labels = [block[1] for block in note_blocks]
        note_blocks, supporting_labels = self._expand_referenced_subdivisions(note_blocks)

        context = {
            "input_htscode": [heading] if heading else [],
            "country_iso2": country_iso2,
            "fallback_start_date": fallback_start.isoformat(),
            "source_label_prefix": ", ".join(block[1] for block in note_blocks),
            "note_labels": [block[0] for block in note_blocks],
            "primary_note_labels": original_labels,
            "supporting_note_labels": supporting_labels,
        }

        cache_key = tuple(block[1] for block in note_blocks)
        if cache_key not in self._note_cache:
            LOGGER.info(
                "Querying LLM for notes %s",
                ", ".join(block[1] for block in note_blocks),
            )
            note_input = "\n\n".join(
                f"{block[0]}\n{block[2]}" for block in note_blocks
            )
            # LOGGER.info("note_input:%s", note_input)
            self._note_cache[cache_key] = self.llm.extract_note(note_input, context)
        payload = self._note_cache[cache_key]

        if len(note_blocks) == 1:
            default_scope_label = note_blocks[0][1]
            default_except_label = f"{note_blocks[0][1]}-exclusion"
        else:
            default_scope_label = ",".join(block[1] for block in note_blocks)
            default_except_label = ",".join(f"{block[1]}-exclusion" for block in note_blocks)

        scope_records, scope_children = self._convert_note_entries(
            payload.get("scope", []),
            default_label=default_scope_label,
            fallback_start=fallback_start,
            fallback_end=fallback_end,
        )
        except_records, except_children = self._convert_note_entries(
            payload.get("except", []),
            default_label=default_except_label,
            fallback_start=fallback_start,
            fallback_end=fallback_end,
        )

        scope_id_map = self._ensure_scopes_batch(scope_records + except_records)

        map_entries: List[Dict[str, Any]] = []
        for scope in scope_records:
            scope_id = scope_id_map.get(self._scope_signature(scope))
            if not scope_id:
                continue
            map_entries.append(
                {
                    "scope_id": scope_id,
                    "measure_id": measure_id,
                    "relation": "include",
                    "note_label": scope.source_label,
                    "text_criteria": None,
                    "start_date": None,
                    "end_date": None,
                }
            )

        for scope in except_records:
            scope_id = scope_id_map.get(self._scope_signature(scope))
            if not scope_id:
                continue
            map_entries.append(
                {
                    "scope_id": scope_id,
                    "measure_id": measure_id,
                    "relation": "exclude",
                    "note_label": scope.source_label,
                    "text_criteria": None,
                    "start_date": None,
                    "end_date": None,
                }
            )

        self.db.ensure_scope_measure_map_bulk(map_entries)

        for child in scope_children:
            self._link_child_heading(
                child_heading=child["key"],
                relation="include",
                parent_heading=heading,
                parent_measure_id=measure_id,
                note_label=child["label"],
                fallback_start=child["start"],
                fallback_end=child["end"],
            )

        for child in except_children:
            self._link_child_heading_reference(
                child_heading=child["key"],
                relation="exclude",
                parent_heading=heading,
                parent_measure_id=measure_id,
                note_label=child["label"],
                country_iso2=country_iso2,
                fallback_start=child["start"],
                fallback_end=child["end"],
            )

    @staticmethod
    def _expand_entry_keys(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        key = entry.get("key")
        if key:
            return [entry]

        keys_value = entry.get("keys")
        if not keys_value:
            return []

        candidates: List[str] = []
        if isinstance(keys_value, str):
            normalized = keys_value.replace(";", ",").replace("\n", ",")
            candidates = [item.strip() for item in normalized.split(",") if item.strip()]
        else:
            try:
                iterator = iter(keys_value)
            except TypeError:
                iterator = iter(())
            for item in iterator:
                text = str(item).strip()
                if text:
                    candidates.append(text)

        expanded: List[Dict[str, Any]] = []
        for candidate in candidates:
            cloned = dict(entry)
            cloned["key"] = candidate
            cloned.pop("keys", None)
            expanded.append(cloned)
        return expanded

    def _convert_note_entries(
        self,
        entries: Iterable[Dict[str, Any]],
        *,
        default_label: str,
        fallback_start: date,
        fallback_end: Optional[date],
    ) -> Tuple[List[ScopeRecord], List[Dict[str, Any]]]:
        records: List[ScopeRecord] = []
        child_refs: List[Dict[str, Any]] = []
        for raw_entry in entries:
            expanded_entries = self._expand_entry_keys(raw_entry)
            if not expanded_entries:
                expanded_entries = [raw_entry]

            for entry in expanded_entries:
                key = entry.get("key")
                if not key:
                    continue
                key_type = entry.get("key_type") or classify_code_type(key)
                label = entry.get("source_label") or default_label
                country = entry.get("country_iso2")
                if country == "":
                    country = None

                start = parse_date(entry.get("effective_start_date")) or fallback_start
                end = parse_date(entry.get("effective_end_date")) or fallback_end
                # skip inserting 99-series headings into scope; process separately
                if key.startswith("99"):
                    child_label = key
                    child_refs.append(
                        {
                            "key": key,
                            "label": child_label,
                            "start": start,
                            "end": end,
                        }
                    )
                    continue
                record = ScopeRecord(
                    key=key,
                    key_type=key_type,
                    country_iso2=country,
                    source_label=label,
                    effective_start_date=start,
                    effective_end_date=end,
                )
                records.append(record)
        return records, child_refs

    def _scope_signature(self, scope: ScopeRecord) -> Tuple[str, str, str, date, Optional[date]]:
        return (
            scope.key,
            scope.key_type,
            scope.country_iso2 or "",
            scope.effective_start_date,
            scope.effective_end_date,
        )

    def _ensure_scopes_batch(self, records: Sequence[ScopeRecord]) -> Dict[Tuple[str, str, str, date, Optional[date]], int]:
        if not records:
            return {}
        return self.db.ensure_scopes_bulk(records)

    def _link_child_heading(
        self,
        *,
        child_heading: str,
        relation: str,
        parent_heading: str,
        parent_measure_id: int,
        note_label: Optional[str],
        fallback_start: date,
        fallback_end: Optional[date],
    ) -> None:
        child_measure_id = self.process_heading(child_heading)
        if not child_measure_id:
            LOGGER.info(
                "Child heading %s could not be processed; skipping linkage to %s",
                child_heading,
                parent_heading,
            )
            return

        scope = ScopeRecord(
            key=child_heading,
            key_type="heading",
            country_iso2=None,
            source_label=note_label or child_heading,
            effective_start_date=fallback_start,
            effective_end_date=fallback_end,
        )
        scope_id = self.db.ensure_scope(scope)
        if not scope_id:
            return

        self.db.ensure_scope_measure_map(
            scope_id=scope_id,
            measure_id=parent_measure_id,
            relation=relation,
            note_label=scope.source_label,
            text_criteria=None,
        )
        LOGGER.info(
            "Recorded child heading reference %s → %s as %s",
            parent_heading,
            child_heading,
            relation,
        )

    def _link_child_heading_reference(
        self,
        *,
        child_heading: str,
        relation: str,
        parent_heading: str,
        parent_measure_id: int,
        note_label: Optional[str],
        country_iso2: Optional[str],
        fallback_start: date,
        fallback_end: Optional[date],
    ) -> None:
        child_measure_id = self.process_heading(child_heading)
        if not child_measure_id:
            LOGGER.info(
                "Child heading %s could not be processed; skipping reference linkage to %s",
                child_heading,
                parent_heading,
            )
            return

        scope = ScopeRecord(
            key=child_heading,
            key_type="heading",
            country_iso2=country_iso2,
            source_label=note_label or child_heading,
            effective_start_date=fallback_start,
            effective_end_date=fallback_end,
        )
        scope_id = self.db.ensure_scope(scope)
        if not scope_id:
            return

        self.db.ensure_scope_measure_map(
            scope_id=scope_id,
            measure_id=parent_measure_id,
            relation=relation,
            note_label=scope.source_label,
            text_criteria=None,
        )
        LOGGER.info(
            "Recorded child heading reference %s → %s as %s",
            parent_heading,
            child_heading,
            relation,
        )

    def _expand_referenced_subdivisions(
        self, note_blocks: List[Tuple[str, str, str]]
    ) -> Tuple[List[Tuple[str, str, str]], List[str]]:
        expanded = list(note_blocks)
        supporting: List[str] = []
        existing: set[str] = {normalized for _, normalized, _ in expanded}
        idx = 0
        while idx < len(expanded):
            original, normalized, text = expanded[idx]
            idx += 1
            related_labels = self._detect_subdivision_references(normalized, text)
            for label in related_labels:
                if label in existing:
                    continue
                note_rows = self.db.fetch_note_rows(label)
                if not note_rows:
                    continue
                block_lines: List[str] = []
                for row in note_rows:
                    label_text = (row.get("label") or "").strip()
                    content_text = (row.get("content") or "").strip()
                    parts: List[str] = []
                    if label_text:
                        parts.append(label_text)
                    if content_text:
                        parts.append(content_text)
                    line = " ".join(parts).strip()
                    if line:
                        block_lines.append(line)
                combined_text = "\n".join(block_lines).strip()
                if not combined_text:
                    continue
                expanded.append((label, label, combined_text))
                supporting.append(label)
                existing.add(label)
        return expanded, supporting

    @staticmethod
    def _detect_subdivision_references(
        normalized_label: str, text: str
    ) -> List[str]:
        LOGGER.info(text)
        if "(" not in normalized_label:
            return []
        base = normalized_label.rsplit("(", 1)[0].strip()
        if not base:
            return []
        matches = set()
        for letter in re.findall(r"subdivision\s*\(\s*([a-z])\s*\)", text, re.IGNORECASE):
            label = f"{base}({letter.lower()})"
            if label != normalized_label:
                matches.add(label)
        return sorted(matches)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTS Section ieepa ingestion agent.")
    parser.add_argument(
        "--dsn",
        default=os.getenv("DATABASE_DSN"),
        help="PostgreSQL DSN (default: DATABASE_DSN env variable).",
    )
    parser.add_argument(
        "--headings",
        nargs="*",
        default=DEFAULT_HEADINGS,
        help="HTS headings to process (default: 9903.88.01/.02/.03/.04/.15).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional file path for writing logs in addition to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(level=log_level, handlers=handlers)

    if not args.dsn:
        raise SystemExit("Database DSN is required (set DATABASE_DSN or pass --dsn).")

    db = Section232Database(args.dsn)
    llm = Section232LLM()
    agent = Section232Agent(db=db, llm=llm)

    try:
        agent.run(args.headings)
        db.commit()
    finally:
        db.close()


if __name__ == "__main__":  # pragma: no cover
    main()
