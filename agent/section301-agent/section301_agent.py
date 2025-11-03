#!/usr/bin/env python3
"""Section 301 processing agent.

This module implements the workflow described in ``agent.txt``:

1. Read the 301 HTS headings (e.g. 9903.88.01) from ``hts_codes``.
2. Insert or update records in ``s301_measures`` for those headings.
3. Use an LLM to structure the descriptions and notes into include / exclude
   scopes with their effective periods.
4. Persist the scopes in ``s301_scope`` and link them to measures via
   ``s301_scope_measure_map``.

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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

try:
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor
except ImportError as exc:  # pragma: no cover - allows linting without psycopg2
    psycopg2 = None  # type: ignore
    Json = None  # type: ignore
    RealDictCursor = None  # type: ignore
    _PSYCOPG2_IMPORT_ERROR = exc
else:
    _PSYCOPG2_IMPORT_ERROR = None


LOGGER = logging.getLogger("section301_agent")

DEFAULT_HEADINGS = [
    "9903.88.01",
    "9903.88.02",
    "9903.88.03",
    "9903.88.04",
    "9903.88.15",
]

LLM_MEASURE_PROMPT = """You are a legal text structure analyzer. From the following text, extract:
1. **Except sections**: all heading numbers listed after the phrase “Except as provided in headings”.
2. **Include sections**: all notes or subheadings mentioned after phrases like “as provided for in” or “as provided for in the subheadings enumerated in”.
3. **Effective period**: any date range describing when the rule is in effect, typically following phrases like “Effective with respect to entries on or after” or “through”.

Input text:
\"\"\"{description}\"\"\"

Return JSON only using this structure:
{{
  "except": [],
  "include": ["note20(vvv)"],
  "effective_period": {{
    "start_date": "June 15, 2024",
    "end_date": "November 29, 2025"
  }}
}}
"""


LLM_NOTE_PROMPT = """You are a structured extractor for HTSUS Section 301 notes.
Output JSON only. No inference. No paraphrasing. Include only codes that are explicitly printed in the input text.

Objective:
From the input legal note text, produce three outputs:
1. **input_htscode** – the heading(s) that the note applies to (from phrases like “For the purposes of heading …”).
2. **scope** – all headings or HTS8 codes that fall under the effective scope of those input headings.
3. **except** – all headings or HTS8 codes explicitly excluded (“except … provided for in …”).

Each object in scope or except must match the `s301_scope` table schema:

- key                  // exact heading or HTS8 code, e.g. "9903.88.05" or "8501.10.40"
- key_type             // "heading" or "hts8" (never "note")
- country_iso2         // "CN" if “products of China” appears, else null
- source_label         // e.g. "note20(a)" or "note20(b)-exclusion"
- effective_start_date // ISO date; use context.fallback_start_date if not stated
- effective_end_date   // ISO date or null

Extraction rules:
A. Identify **input_htscode** from the leading clause “For the purposes of heading …”.
B. Identify all **scope** items:
   - For headings: include every 9903.xx.xx printed under “applies to…” or similar.
   - For HTS8: include every 8-digit subheading printed in the list.
   - Do NOT include `input_htscode` again.
C. Identify all **except** items:
   - Items explicitly listed after “except … provided for in …”.
   - Mark their source_label with “-exclusion”.
D. Use “CN” as country_iso2 if “products of China” is mentioned.
E. Use context.fallback_start_date for start_date if no effective period is found.
F. Remove duplicates.
G. When multiple note blocks are provided, treat each block independently and set `source_label` using that note label (e.g. "note20(a)" or "note20(b)-exclusion"). Use the provided `context.note_labels` array (same order as the blocks) if you need the human-readable label.

Context (provided by caller):
{context_json}

Input (one or more note blocks, each beginning with its note label on the first line):
\"\"\"{note_text}\"\"\"

Output JSON format:
{{
  "input_htscode": ["9903.88.01"],
  "scope": [
    {{ "key": "2845.20.00", "key_type": "hts8", "country_iso2": "CN", "source_label": "note20(b)", "effective_start_date": "1900-01-01", "effective_end_date": null }},
    {{ "key": "2845.30.00", "key_type": "hts8", "country_iso2": "CN", "source_label": "note20(b)", "effective_start_date": "1900-01-01", "effective_end_date": null }}
  ],
  "except": [
    {{ "key": "9903.88.05", "key_type": "heading", "country_iso2": "CN", "source_label": "note20(a)-exclusion", "effective_start_date": "1900-01-01", "effective_end_date": null }}
  ]
}}
"""


NOTE_TOKEN_RE = re.compile(r"\(\s*([^)]+?)\s*\)")


@dataclass
class MeasureAnalysis:
    include: List[str]
    exclude: List[str]
    effective_start: Optional[date]
    effective_end: Optional[date]


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


class Section301Database:
    """Minimal database helper around psycopg2."""

    def __init__(self, dsn: str):
        if psycopg2 is None:
            raise RuntimeError(
                "psycopg2 is required to use Section301Database"
            ) from _PSYCOPG2_IMPORT_ERROR
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False

    def close(self) -> None:
        self._conn.close()

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

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
        country_iso2: str,
        ad_valorem_rate: Decimal,
        value_basis: str,
        notes: Optional[Dict[str, Any]],
        start_date: date,
        end_date: Optional[date],
    ) -> int:
        query_select = """
            SELECT id FROM s301_measures
            WHERE heading = %s
              AND country_iso2 = %s
              AND ad_valorem_rate = %s
              AND value_basis = %s
              AND effective_start_date = %s
              AND COALESCE(effective_end_date, DATE '9999-12-31') =
                  COALESCE(%s, DATE '9999-12-31')
            LIMIT 1
        """
        with self._conn.cursor() as cur:
            cur.execute(
                query_select,
                (heading, country_iso2, ad_valorem_rate, value_basis, start_date, end_date),
            )
            row = cur.fetchone()
            if row:
                return row[0]

        query_insert = """
            INSERT INTO s301_measures
            (heading, country_iso2, ad_valorem_rate, value_basis, notes,
             effective_start_date, effective_end_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        payload_notes = Json(notes) if notes is not None else None
        with self._conn.cursor() as cur:
            cur.execute(
                query_insert,
                (
                    heading,
                    country_iso2,
                    ad_valorem_rate,
                    value_basis,
                    payload_notes,
                    start_date,
                    end_date,
                ),
            )
            measure_id = cur.fetchone()[0]
        LOGGER.info("Created measure %s for heading %s", measure_id, heading)
        return measure_id

    def ensure_scope(self, record: ScopeRecord) -> int:
        query_select = """
            SELECT id FROM s301_scope
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
            INSERT INTO s301_scope
            (key, key_type, country_iso2, source_label,
             effective_start_date, effective_end_date)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        with self._conn.cursor() as cur:
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
        LOGGER.info("Created scope %s for key %s", scope_id, record.key)
        return scope_id

    def ensure_scope_measure_map(
        self,
        scope_id: int,
        measure_id: int,
        relation: str,
        note_label: Optional[str],
        text_criteria: Optional[str],
        start_date: date,
        end_date: Optional[date],
    ) -> int:
        query_select = """
            SELECT id FROM s301_scope_measure_map
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
            INSERT INTO s301_scope_measure_map
            (scope_id, measure_id, relation, note_label, text_criteria,
             effective_start_date, effective_end_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        with self._conn.cursor() as cur:
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
        LOGGER.info(
            "Created scope↔measure relation %s (scope=%s, measure=%s, %s)",
            map_id,
            scope_id,
            measure_id,
            relation,
        )
        return map_id

    def fetch_note_rows(self, label: str) -> List[Dict[str, Any]]:
        query = """
            SELECT subchapter, label, content, raw_html, path
            FROM hts_notes
            WHERE lower(label) = lower(%s)
            ORDER BY subchapter, array_length(path, 1), path
        """
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (label,))
            return cur.fetchall()


class Section301LLM:
    """Thin wrapper around an LLM endpoint (default: OpenAI chat completions)."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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
            "temperature": 0,
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
        raw = self._post(message)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to decode measure analysis JSON: {raw}") from exc

        include = [item.strip() for item in payload.get("include", []) if item and item.strip()]
        exclude = [item.strip() for item in payload.get("except", []) if item and item.strip()]
        eff_payload = payload.get("effective_period") or {}
        start = parse_date(eff_payload.get("start_date"))
        end = parse_date(eff_payload.get("end_date"))
        return MeasureAnalysis(include=include, exclude=exclude, effective_start=start, effective_end=end)

    def extract_note(self, note_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        context_json = json.dumps(context, ensure_ascii=False, sort_keys=True)
        message = LLM_NOTE_PROMPT.format(
            note_text=note_text.strip(),
            context_json=context_json,
        )
        raw = self._post(message)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to decode note analysis JSON: {raw}") from exc
        return payload


class Section301Agent:
    """Coordinates DB + LLM workflow for Section 301 measures."""

    def __init__(
        self,
        db: Section301Database,
        llm: Section301LLM,
        *,
        country_iso2: str = "CN",
    ):
        self.db = db
        self.llm = llm
        self.country_iso2 = country_iso2
        self._processed_headings: set[str] = set()
        self._note_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    def run(self, headings: Sequence[str]) -> None:
        for heading in headings:
            self.process_heading(heading)

    def process_heading(self, heading: str) -> None:
        normalized = heading.strip()
        if not normalized:
            return
        if normalized in self._processed_headings:
            LOGGER.debug("Heading %s already processed; skipping", normalized)
            return
        LOGGER.info("Processing heading %s", normalized)

        row = self.db.fetch_hts_code(normalized)
        if not row:
            LOGGER.warning("Heading %s not found in hts_codes; skipping", normalized)
            return

        status = (row.get("status") or "").strip().lower()
        if status == "expired":
            LOGGER.info("Heading %s is marked expired; skipping", normalized)
            return

        description = (row.get("description") or "").strip()
        if not description:
            LOGGER.warning("Heading %s has empty description; skipping", normalized)
            return

        analysis = self.llm.extract_measure(description)
        LOGGER.info("analysis : %s", analysis)
        start_date = analysis.effective_start or date(1900, 1, 1)
        end_date = analysis.effective_end
        rate = self._derive_rate(description)

        measure_id = self.db.ensure_measure(
            heading=normalized,
            country_iso2=self.country_iso2,
            ad_valorem_rate=rate,
            value_basis="customs_value",
            notes=None,
            start_date=start_date,
            end_date=end_date,
        )

        self._processed_headings.add(normalized)
        try:
            self._apply_scope_links(
                heading=normalized,
                measure_id=measure_id,
                start_date=start_date,
                end_date=end_date,
                includes=analysis.include,
                excludes=analysis.exclude,
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            self._processed_headings.discard(normalized)
            raise

    def _derive_rate(self, description: str) -> Decimal:
        match = re.search(r"(\d+(?:\.\d+)?)\s*percent", description, re.IGNORECASE)
        if match:
            return Decimal(match.group(1)).quantize(Decimal("0.001"))
        return Decimal("25.000")

    def _apply_scope_links(
        self,
        *,
        heading: str,
        measure_id: int,
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
                fallback_start=start_date,
                fallback_end=end_date,
            )
        for code in include_others:
            self._handle_non_note_reference(
                reference=code,
                relation="include",
                heading=heading,
                measure_id=measure_id,
                start_date=start_date,
                end_date=end_date,
            )

        if exclude_notes:
            self._process_note_references(
                note_refs=exclude_notes,
                relation="exclude",
                heading=heading,
                measure_id=measure_id,
                fallback_start=start_date,
                fallback_end=end_date,
            )
        for code in exclude_others:
            self._handle_non_note_reference(
                reference=code,
                relation="exclude",
                heading=heading,
                measure_id=measure_id,
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
            country_iso2=self.country_iso2,
            source_label=f"{heading}-description",
            effective_start_date=start_date,
            effective_end_date=end_date,
        )
        scope_id = self.db.ensure_scope(scope)
        self.db.ensure_scope_measure_map(
            scope_id=scope_id,
            measure_id=measure_id,
            relation=relation,
            note_label=scope.source_label,
            text_criteria=None,
            start_date=start_date,
            end_date=end_date,
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
            combined_text = "\n".join(
                row["content"] for row in note_rows if row.get("content")
            ).strip()
            if not combined_text:
                continue
            note_blocks.append((original, normalized, combined_text))

        if not note_blocks:
            return

        context = {
            "country_iso2": self.country_iso2,
            "fallback_start_date": fallback_start.isoformat(),
            "source_label_prefix": ", ".join(block[1] for block in note_blocks),
            "note_labels": [block[0] for block in note_blocks],
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
            self._note_cache[cache_key] = self.llm.extract_note(note_input, context)
        payload = self._note_cache[cache_key]

        if len(note_blocks) == 1:
            default_scope_label = note_blocks[0][1]
            default_except_label = f"{note_blocks[0][1]}-exclusion"
        else:
            default_scope_label = ",".join(block[1] for block in note_blocks)
            default_except_label = ",".join(f"{block[1]}-exclusion" for block in note_blocks)

        scope_records = self._convert_note_entries(
            payload.get("scope", []),
            default_label=default_scope_label,
            fallback_start=fallback_start,
            fallback_end=fallback_end,
        )
        except_records = self._convert_note_entries(
            payload.get("except", []),
            default_label=default_except_label,
            fallback_start=fallback_start,
            fallback_end=fallback_end,
        )

        for scope in scope_records:
            scope_id = self.db.ensure_scope(scope)
            self.db.ensure_scope_measure_map(
                scope_id=scope_id,
                measure_id=measure_id,
                relation="include",
                note_label=scope.source_label,
                text_criteria=None,
                start_date=scope.effective_start_date,
                end_date=scope.effective_end_date,
            )
            if scope.key.startswith("99"):
                self.process_heading(scope.key)

        for scope in except_records:
            scope_id = self.db.ensure_scope(scope)
            self.db.ensure_scope_measure_map(
                scope_id=scope_id,
                measure_id=measure_id,
                relation="exclude",
                note_label=scope.source_label,
                text_criteria=None,
                start_date=scope.effective_start_date,
                end_date=scope.effective_end_date,
            )
            if scope.key.startswith("99"):
                self.process_heading(scope.key)

    def _convert_note_entries(
        self,
        entries: Iterable[Dict[str, Any]],
        *,
        default_label: str,
        fallback_start: date,
        fallback_end: Optional[date],
    ) -> List[ScopeRecord]:
        records: List[ScopeRecord] = []
        for entry in entries:
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
            record = ScopeRecord(
                key=key,
                key_type=key_type,
                country_iso2=country,
                source_label=label,
                effective_start_date=start,
                effective_end_date=end,
            )
            records.append(record)
        return records


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTS Section 301 ingestion agent.")
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if not args.dsn:
        raise SystemExit("Database DSN is required (set DATABASE_DSN or pass --dsn).")

    db = Section301Database(args.dsn)
    llm = Section301LLM()
    agent = Section301Agent(db=db, llm=llm)

    try:
        agent.run(args.headings)
        db.commit()
    finally:
        db.close()


if __name__ == "__main__":  # pragma: no cover
    main()
