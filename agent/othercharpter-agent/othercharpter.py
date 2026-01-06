#!/usr/bin/env python3
"""Other Chapter (notes 33/36/37/38) processing agent.

Reads note content from local text files and HTS descriptions from hts_codes,
uses an LLM to extract measures with scoped relations, and stores them into
tables that mirror sieepa_measures, sieepa_scope, and sieepa_scope_measure_map.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - allows linting without psycopg2 installed
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor, execute_values
except ImportError as exc:  # pragma: no cover
    psycopg2 = None  # type: ignore
    Json = None  # type: ignore
    RealDictCursor = None  # type: ignore
    _PSYCOPG2_IMPORT_ERROR = exc
else:
    _PSYCOPG2_IMPORT_ERROR = None

try:  # pragma: no cover - allows linting without requests installed
    import requests
except ImportError as exc:  # pragma: no cover
    requests = None  # type: ignore
    _REQUESTS_IMPORT_ERROR = exc
else:
    _REQUESTS_IMPORT_ERROR = None

LOGGER = logging.getLogger("otherchapter_agent")

DEFAULT_START_DATE = date(1900, 1, 1)

NOTE_LABELS = {
    33: "note(33)",
    36: "note(36)",
    37: "note(37)",
    38: "note(38)",
}
NOTE_HEADINGS = {
    33: [
        "9903.94.01",
        "9903.94.02",
        "9903.94.03",
        "9903.94.04",
        "9903.94.05",
        "9903.94.06",
        "9903.94.07",
        "9903.94.31",
        "9903.94.32",
        "9903.94.33",
        "9903.94.40",
        "9903.94.41",
        "9903.94.42",
        "9903.94.43",
        "9903.94.44",
        "9903.94.45",
        "9903.94.50",
        "9903.94.51",
        "9903.94.52",
        "9903.94.53",
        "9903.94.54",
        "9903.94.55",
        "9903.94.60",
        "9903.94.61",
        "9903.94.62",
        "9903.94.63",
        "9903.94.64",
        "9903.94.65",
    ],
    36: ["9903.78.01", "9903.78.02"],
    37: [
        "9903.76.01",
        "9903.76.02",
        "9903.76.03",
        "9903.76.04",
        "9903.76.20",
        "9903.76.21",
        "9903.76.22",
        "9903.76.23",
    ],
    38: [
        "9903.74.01",
        "9903.74.02",
        "9903.74.03",
        "9903.74.05",
        "9903.74.06",
        "9903.74.07",
        "9903.74.08",
        "9903.74.09",
        "9903.74.10",
        "9903.74.11",
    ],
}
NOTE_PDF_DIR = Path(__file__).resolve().parent / "pdf"
NOTE_PDF_FILES = {
    33: "note33.txt",
    36: "note36.txt",
    37: "note37.txt",
    38: "note38.txt",
}
NOTE38_SUBVISIONI_PATH = NOTE_PDF_DIR / "note38subvisioni.txt"
EXTRA_MEASURES_DIR = Path(__file__).resolve().parent / "output"

HTS_CODE_RE = re.compile(r"\b\d{4}\.\d{2}\.\d{2}(?:\d{2})?\b")
RANGE_SHORT_RE = re.compile(r"\b(\d{4}\.\d{2}\.)(\d{2})\s*[~\u2013-]\s*(\d{2})\b")
RANGE_FULL_RE = re.compile(r"\b(\d{4}\.\d{2}\.\d{2})\s*[~\u2013-]\s*(\d{4}\.\d{2}\.\d{2})\b")

LLM_PROMPT = '''You are a structured extractor for HTSUS Chapter 99 notes (note 33/36/37/38).
Return JSON only.

Output must be a JSON object with a top-level key "measures" that is an array.
Each element represents one Chapter 99 heading referenced in this note (including headings mentioned only as exclusions).

Required measure fields:
- heading
- country_iso2
- ad_valorem_rate
- value_basis
- is_potential
- notes (object)
- effective_start_date
- effective_end_date
- scopes (array)

Each scope item must include:
- key or keys (HTS heading/subheading; do not use note references here)
- key_type
- relation (include/exclude)
- country_iso2
- source_label
- note_label
- text_criteria
- effective_start_date
- effective_end_date

Rules:
- Output all Chapter 99 headings for this note as separate array elements (no omissions).
- Do not output duplicate heading entries; each heading appears once in measures.
- Use context.chapter99_headings as the required list of Chapter 99 headings to output.
- You MUST also output measures for any other Chapter 99 headings referenced in the note text (e.g., headings listed in "shall not be subject to" or "except as provided" clauses), even if they are not in context.chapter99_headings; do not invent headings.
- Do not output alt_heading or relationship tables. Substitution/conditional rules must set is_potential=true.
- Set ad_valorem_rate by extracting the numeric percent from context.hts_codes[].general_rate_of_duty (ignore note text). Examples: "The duty provided in the applicable subheading + 25%" => 25; "25 percent" => 25. If multiple numbers exist, use the explicit percent value. If general_rate_of_duty only says "the duty provided in the applicable subheading" (no percent), set ad_valorem_rate to 0. If no percent is present and it is not that phrase, or the heading is missing from context.hts_codes, set ad_valorem_rate to null and value_basis to "customs_value".
- Use ad_valorem_rate=0 for exemptions.
- Use notes JSON to store pairing info, in_lieu_of vs in_addition_to, FTA/AD/CVD stacking, or other criteria.
- Include Chapter 99 headings as scope entries (key_type=heading) when they appear in exclusions.
- Use ISO-2 for country_iso2 (EU as "EU").
- If country_iso2 is not specified by the text, use null.
- Use ISO date strings when explicit dates are present; otherwise null.
- key_type must be one of: heading, hts8, hts10, note. Do not use "hts".
- You may group codes with identical attributes using "keys" (comma-separated) instead of "key".
- scopes can be an empty array when no explicit scope is stated.
- Only use HTS codes explicitly printed in the note text or listed in context.hts_codes. Do not invent codes.
- Chapter-level references (e.g., chapters 72/73/76) are scopes that represent heading2; emit them as key_type=heading using 2-digit chapter numbers (e.g., "72").
- If the note text references subdivision (i) of U.S. note 38, expand it using the "Note 38 subdivision (i) text" below and emit those HTS codes as scopes.
- If a measure notes exclusions like "HTS provisions of U.S. note 38(i)" or "articles classifiable in subdivision (i) of U.S. note 38", you MUST add those expanded HTS codes as exclude scopes for that measure.
- Do not emit key_type="note" in scopes. If the note text says "as provided in subdivision (b)", expand to the HTS codes listed in that subdivision (from context.hts_codes) and output those codes as scopes (hts8/hts10/heading).
- Interpret HTS description text in context.hts_codes:
  - "products of the United Kingdom" => country_iso2 = "UK" (similar for other countries).
  - "Except as provided in subheadings 9903.80.60 through 9903.80.62" => exclude scopes for 9903.80.60, 9903.80.61, 9903.80.62.
  - "Except for goods loaded ... before <date> ... entered ... before <date>" => store in notes as date_of_loading_before and entry_date_before (ISO dates).
  - "Effective with respect to entries on or after <date>" => store in notes as entry_date_on_or_after (ISO date).
- When multiple Chapter 99 headings are listed in one sentence, they share the same scope list unless the text limits a subset.
- Treat "and certain entries under 9903.94.31" (or similar phrasing) as part of the heading list that shares the subdivision scope list; do not omit 9903.94.31 from those shared HTS scopes.
- Exclude scopes indicate the scope key supersedes the current measure: if the scope key applies, the current measure does not.
- When the note says goods under heading A are "not subject to" duties under heading B, attach an exclude scope to heading B with key=A (key_type=heading). Do not attach B as an exclusion on heading A.

Context (JSON):
{context_json}

Note text:
<<<NOTE_TEXT
{note_text}
NOTE_TEXT>>>

Note 38 subdivision (i) text (authoritative for references in other notes):
<<<NOTE38_SUBVISION_I
{note38_subvision_i_text}
NOTE38_SUBVISION_I>>>
'''


def _normalize_code(value: str) -> str:
    return value.strip().strip(",;")


def _parse_date(value: Optional[Any]) -> Optional[date]:
    if not value:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in {"null", "none"}:
            return None
        if "/" in cleaned:
            cleaned = cleaned.replace("/", "-")
        try:
            return date.fromisoformat(cleaned)
        except ValueError:
            return None
    return None


def _parse_rate(value: Optional[Any]) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "")
        if not cleaned:
            return None
        try:
            return Decimal(cleaned)
        except Exception:
            return None
    return None


def _derive_rate(description: str) -> Decimal:
    desc = description or ""
    match_add = re.search(
        r"the duty provided in the applicable subheading\\s*(?:\\+|plus)\\s*(\\d+(?:\\.\\d+)?)\\s*(?:percent|%)",
        desc,
        re.IGNORECASE,
    )
    if match_add:
        return Decimal(match_add.group(1)).quantize(Decimal("0.001"))

    if re.search(r"the duty provided in the applicable subheading", desc, re.IGNORECASE):
        return Decimal("0.000")

    match = re.search(r"(\\d+(?:\\.\\d+)?)\\s*percent", desc, re.IGNORECASE)
    if match:
        return Decimal(match.group(1)).quantize(Decimal("0.001"))

    fallback_num = re.search(r"(\\d+(?:\\.\\d+)?)", desc)
    if fallback_num:
        value = Decimal(fallback_num.group(1)).quantize(Decimal("0.001"))
        if value <= Decimal("999.000"):
            return value
        logger.warning(
            "Derived fallback rate %s exceeds numeric precision; using default 25%%",
            value,
        )

    return Decimal("25.000")


def _normalize_iso(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = value.strip().upper()
    return cleaned or None


def _infer_key_type(code: str) -> str:
    digits = "".join(ch for ch in code if ch.isdigit())
    if len(digits) >= 10:
        return "hts10"
    if len(digits) == 8:
        return "hts8"
    return "heading"


def _normalize_key_type(raw: Optional[str], code: str) -> str:
    if not raw:
        return _infer_key_type(code)
    cleaned = raw.strip().lower()
    if cleaned == "hts":
        return _infer_key_type(code)
    if cleaned in {"heading", "hts8", "hts10", "note"}:
        return cleaned
    return _infer_key_type(code)


def _split_keys(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _expand_short_range(prefix: str, start: str, end: str) -> List[str]:
    start_int = int(start)
    end_int = int(end)
    if end_int < start_int:
        start_int, end_int = end_int, start_int
    return [f"{prefix}{i:02d}" for i in range(start_int, end_int + 1)]


def _expand_full_range(start: str, end: str) -> List[str]:
    start_prefix, start_suffix = start[:-2], start[-2:]
    end_prefix, end_suffix = end[:-2], end[-2:]
    if start_prefix != end_prefix:
        return [start, end]
    return _expand_short_range(start_prefix, start_suffix, end_suffix)


def _extract_codes(text: str) -> List[str]:
    codes: set[str] = set()
    for prefix, start, end in RANGE_SHORT_RE.findall(text):
        codes.update(_expand_short_range(prefix, start, end))
    for start, end in RANGE_FULL_RE.findall(text):
        codes.update(_expand_full_range(start, end))
    codes.update(HTS_CODE_RE.findall(text))
    return sorted({_normalize_code(code) for code in codes})


@dataclass
class MeasureRecord:
    heading: str
    country_iso2: Optional[str]
    ad_valorem_rate: Decimal
    value_basis: str
    melt_pour_origin_iso2: Optional[str]
    origin_exclude_iso2: Optional[List[str]]
    notes: Optional[Dict[str, Any]]
    effective_start_date: date
    effective_end_date: Optional[date]
    is_potential: Optional[bool]


@dataclass
class ScopeRecord:
    key: str
    key_type: str
    country_iso2: Optional[str]
    source_label: Optional[str]
    effective_start_date: date
    effective_end_date: Optional[date]


@dataclass
class ScopeMapEntry:
    scope: ScopeRecord
    relation: str
    note_label: Optional[str]
    text_criteria: Optional[str]
    map_start: date
    map_end: Optional[date]


class OtherChapterLLM:
    """Thin wrapper around an LLM endpoint (default: OpenAI chat completions)."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 36000,
    ) -> None:
        if requests is None:  # pragma: no cover - runtime dependency
            raise _REQUESTS_IMPORT_ERROR
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5")
        self.base_url = base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY (or explicit api_key) is required for LLM calls")

    def _chat(self, message: str) -> str:
        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a precise legal text parser. Respond with JSON only."},
                {"role": "user", "content": message},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"LLM HTTP error: {exc} -> {response.text}") from exc
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response structure: {data}") from exc

    def extract(
        self,
        note_number: int,
        note_text: str,
        context: Dict[str, Any],
        note38_subvision_i_text: str,
    ) -> Dict[str, Any]:
        message = LLM_PROMPT.format(
            note_text=note_text.strip(),
            context_json=json.dumps(context, ensure_ascii=True),
            note38_subvision_i_text=note38_subvision_i_text.strip(),
        )
        raw = self._chat(message)
        return _parse_json_payload(raw)


class OtherChapterDB:
    def __init__(self, dsn: str, table_prefix: str = "otherch") -> None:
        if psycopg2 is None:  # pragma: no cover
            raise _PSYCOPG2_IMPORT_ERROR
        self._dsn = dsn
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        self.measures_table = f"{table_prefix}_measures"
        self.scope_table = f"{table_prefix}_scope"
        self.map_table = f"{table_prefix}_scope_measure_map"

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    def fetch_note_rows(self, label: str) -> List[Dict[str, Any]]:
        base_query = (
            "SELECT chapter, subchapter, label, content, raw_html, path "
            "FROM hts_notes WHERE lower(label) = lower(%s) "
            "ORDER BY subchapter, array_length(path, 1), path"
        )
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
                (anchor["chapter"], anchor["subchapter"], len(anchor["path"]), anchor["path"]),
            )
            rows = cur.fetchall()
            return rows or [anchor]

    def fetch_hts_descriptions(self, codes: Sequence[str]) -> Dict[str, Dict[str, str]]:
        if not codes:
            return {}
        descriptions: Dict[str, Dict[str, str]] = {}
        chunk_size = 200
        for idx in range(0, len(codes), chunk_size):
            chunk = list(codes[idx : idx + chunk_size])
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT hts_number, description, general_rate_of_duty FROM hts_codes "
                    "WHERE hts_number = ANY(%s) AND COALESCE(status, '') != 'expired'",
                    (chunk,),
                )
                for hts_number, description, general_rate_of_duty in cur.fetchall():
                    code = _normalize_code(hts_number)
                    general_rate_text = (general_rate_of_duty or "").strip()
                    rate = _derive_rate(general_rate_text)
                    descriptions[code] = {
                        "description": description or "",
                        "general_rate_of_duty": general_rate_text,
                        "ad_valorem_rate": str(rate),
                    }
        return descriptions

    def insert_measures(self, records: Sequence[MeasureRecord]) -> List[int]:
        if not records:
            return []
        values = [
            (
                record.heading,
                record.country_iso2,
                record.ad_valorem_rate,
                record.value_basis,
                record.melt_pour_origin_iso2,
                record.origin_exclude_iso2,
                Json(record.notes) if record.notes is not None else None,
                record.effective_start_date,
                record.effective_end_date,
                record.is_potential,
            )
            for record in records
        ]
        insert_query = (
            f"INSERT INTO {self.measures_table} "
            "(heading, country_iso2, ad_valorem_rate, value_basis, melt_pour_origin_iso2, "
            "origin_exclude_iso2, notes, effective_start_date, effective_end_date, is_potential) "
            "VALUES %s ON CONFLICT DO NOTHING"
        )
        with self._conn:
            with self._conn.cursor() as cur:
                execute_values(cur, insert_query, values, page_size=len(values))
                lookup_values = [
                    (
                        idx,
                        record.heading,
                        record.country_iso2,
                        record.effective_start_date,
                        record.effective_end_date,
                    )
                    for idx, record in enumerate(records)
                ]
                lookup_query = (
                    f"SELECT v.idx, m.id "
                    f"FROM (VALUES %s) AS v(idx, heading, country_iso2, start_date, end_date) "
                    f"JOIN {self.measures_table} AS m "
                    "ON m.heading = v.heading "
                    "AND COALESCE(m.country_iso2, '') = COALESCE(v.country_iso2, '') "
                    "AND m.effective_start_date = v.start_date "
                    "AND COALESCE(m.effective_end_date, DATE '9999-12-31') = "
                    "COALESCE(v.end_date, DATE '9999-12-31') "
                    "ORDER BY v.idx"
                )
                execute_values(
                    cur,
                    lookup_query,
                    lookup_values,
                    template="(%s::int,%s::text,%s::text,%s::date,%s::date)",
                    page_size=len(lookup_values),
                )
                rows = cur.fetchall()
                return [row[1] for row in rows]

    def insert_scopes(self, records: Sequence[ScopeRecord]) -> List[int]:
        if not records:
            return []
        values = [
            (
                record.key,
                record.key_type,
                record.country_iso2,
                record.source_label,
                record.effective_start_date,
                record.effective_end_date,
            )
            for record in records
        ]
        insert_query = (
            f"INSERT INTO {self.scope_table} "
            "(key, key_type, country_iso2, source_label, effective_start_date, effective_end_date) "
            "VALUES %s ON CONFLICT DO NOTHING"
        )
        with self._conn:
            with self._conn.cursor() as cur:
                execute_values(cur, insert_query, values, page_size=len(values))
                lookup_values = [
                    (
                        idx,
                        record.key,
                        record.key_type,
                        record.country_iso2,
                        record.effective_start_date,
                        record.effective_end_date,
                    )
                    for idx, record in enumerate(records)
                ]
                lookup_query = (
                    f"SELECT v.idx, s.id "
                    f"FROM (VALUES %s) AS v(idx, key, key_type, country_iso2, start_date, end_date) "
                    f"JOIN {self.scope_table} AS s "
                    "ON s.key = v.key "
                    "AND s.key_type = v.key_type "
                    "AND COALESCE(s.country_iso2, '') = COALESCE(v.country_iso2, '') "
                    "AND s.effective_start_date = v.start_date "
                    "AND COALESCE(s.effective_end_date, DATE '9999-12-31') = "
                    "COALESCE(v.end_date, DATE '9999-12-31') "
                    "ORDER BY v.idx"
                )
                execute_values(
                    cur,
                    lookup_query,
                    lookup_values,
                    template="(%s::int,%s::text,%s::text,%s::text,%s::date,%s::date)",
                    page_size=len(lookup_values),
                )
                rows = cur.fetchall()
                return [row[1] for row in rows]

    def insert_scope_measure_map(self, entries: Sequence[Tuple[Any, ...]]) -> None:
        if not entries:
            return
        query = (
            f"INSERT INTO {self.map_table} "
            "(scope_id, measure_id, relation, note_label, text_criteria, effective_start_date, effective_end_date) "
            "VALUES %s ON CONFLICT DO NOTHING"
        )
        with self._conn:
            with self._conn.cursor() as cur:
                execute_values(cur, query, list(entries), page_size=len(entries))


class OtherChapterProcessor:
    def __init__(self, db: OtherChapterDB, llm: OtherChapterLLM) -> None:
        self.db = db
        self.llm = llm

    def deal_note33(self) -> None:
        self.process_note(33)

    def deal_note36(self) -> None:
        self.process_note(36)

    def deal_note37(self) -> None:
        self.process_note(37)

    def deal_note38(self) -> None:
        self.process_note(38)

    def process_note(self, note_number: int) -> None:
        label = NOTE_LABELS[note_number]
        try:
            note_text = self._load_note_text(note_number)
        except FileNotFoundError as exc:
            LOGGER.warning("Note %s PDF missing: %s", label, exc)
            return
        except Exception as exc:
            LOGGER.warning("Failed to load note %s PDF: %s", label, exc)
            return
        print(note_text)
        note_headings = NOTE_HEADINGS.get(note_number, [])
        # extracted_codes = _extract_codes(note_text)
        codes = sorted({*note_headings})
        descriptions = self.db.fetch_hts_descriptions(codes)
        context = {
            "note_number": note_number,
            "note_label": label,
            "chapter99_headings": note_headings,
            "hts_codes": [
                {
                    "code": code,
                    "description": descriptions.get(code, {}).get("description", ""),
                    "general_rate_of_duty": descriptions.get(code, {}).get("general_rate_of_duty", ""),
                    "ad_valorem_rate": descriptions.get(code, {}).get("ad_valorem_rate", ""),
                }
                for code in codes
            ],
        }
        note38_subvision_i_text = self._load_note38_subvision_i_text()
        analysis = self.llm.extract(note_number, note_text, context, note38_subvision_i_text)
        print("=======================")
        print(analysis)
        print("=======================")
        allowed_headings = {str(code) for code in note_headings}
        primary_measures, extra_measures = self._split_measures_by_heading(analysis, allowed_headings)
        self._write_extra_measures(note_number, label, extra_measures)
        self._persist(note_number, primary_measures)

    def _build_note_text(self, rows: Sequence[Dict[str, Any]]) -> str:
        chunks = []
        for row in rows:
            label = row.get("label") or ""
            content = row.get("content") or ""
            chunks.append(f"{label}\n{content}")
        return "\n\n".join(chunks)

    def _load_note_text(self, note_number: int) -> str:
        filename = NOTE_PDF_FILES.get(note_number)
        if not filename:
            raise FileNotFoundError(f"no note file mapping for note {note_number}")
        note_path = NOTE_PDF_DIR / filename
        if not note_path.exists():
            raise FileNotFoundError(str(note_path))
        return note_path.read_text(encoding="utf-8").strip()

    def _load_note38_subvision_i_text(self) -> str:
        if not NOTE38_SUBVISIONI_PATH.exists():
            LOGGER.warning("Note 38 subdivision (i) file missing: %s", NOTE38_SUBVISIONI_PATH)
            return ""
        return NOTE38_SUBVISIONI_PATH.read_text(encoding="utf-8").strip()

    def _split_measures_by_heading(
        self, payload: Any, allowed_headings: set[str]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        measures = _coerce_measure_payload(payload)
        primary: List[Dict[str, Any]] = []
        extra: List[Dict[str, Any]] = []
        for measure in measures:
            heading = _normalize_code(str(measure.get("heading") or ""))
            if heading and heading not in allowed_headings:
                extra.append(measure)
            else:
                primary.append(measure)
        return primary, extra

    def _write_extra_measures(
        self, note_number: int, note_label: str, extra_measures: Sequence[Dict[str, Any]]
    ) -> None:
        EXTRA_MEASURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = EXTRA_MEASURES_DIR / f"note{note_number}_extra_measures.json"
        payload = {
            "note_number": note_number,
            "note_label": note_label,
            "measures": list(extra_measures),
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Wrote extra measures to %s", output_path)

    def _persist(self, note_number: int, payload: Any) -> None:
        measure_entries = _coerce_measure_payload(payload)
        if not measure_entries:
            return
        measure_records: List[MeasureRecord] = []
        scopes_by_measure: List[List[ScopeMapEntry]] = []

        for entry in measure_entries:
            heading = _normalize_code(entry.get("heading", ""))
            if not heading:
                continue
            start = _parse_date(entry.get("effective_start_date")) or DEFAULT_START_DATE
            end = _parse_date(entry.get("effective_end_date"))
            notes = _coerce_notes(entry.get("notes"))
            notes.setdefault("note_number", note_number)
            record = MeasureRecord(
                heading=heading,
                country_iso2=_normalize_iso(entry.get("country_iso2")),
                ad_valorem_rate=_parse_rate(entry.get("ad_valorem_rate")) or Decimal("0"),
                value_basis=entry.get("value_basis") or "total_value",
                melt_pour_origin_iso2=_normalize_iso(entry.get("melt_pour_origin_iso2")),
                origin_exclude_iso2=_normalize_iso_list(entry.get("origin_exclude_iso2")),
                notes=notes,
                effective_start_date=start,
                effective_end_date=end,
                is_potential=_coerce_bool(entry.get("is_potential")),
            )
            measure_records.append(record)
            scopes_by_measure.append(_expand_scope_entries(entry.get("scopes") or []))

        measure_ids = self.db.insert_measures(measure_records)
        unique_scopes: List[ScopeRecord] = []
        scope_index: Dict[Tuple[str, str, Optional[str], date, Optional[date]], int] = {}
        pending_maps: List[Tuple[Tuple[str, str, Optional[str], date, Optional[date]], int, ScopeMapEntry]] = []

        for measure_id, scope_entries in zip(measure_ids, scopes_by_measure):
            for entry in scope_entries:
                signature = _scope_signature(entry.scope)
                if signature not in scope_index:
                    scope_index[signature] = len(unique_scopes)
                    unique_scopes.append(entry.scope)
                pending_maps.append((signature, measure_id, entry))

        scope_ids = self.db.insert_scopes(unique_scopes)
        signature_to_id = {
            signature: scope_ids[index] for signature, index in scope_index.items()
        }
        map_entries: List[Tuple[Any, ...]] = []
        seen_maps: set[Tuple[int, int, str, date, Optional[date]]] = set()
        for signature, measure_id, entry in pending_maps:
            scope_id = signature_to_id[signature]
            map_sig = (scope_id, measure_id, entry.relation, entry.map_start, entry.map_end)
            if map_sig in seen_maps:
                continue
            seen_maps.add(map_sig)
            map_entries.append(
                (
                    scope_id,
                    measure_id,
                    entry.relation,
                    entry.note_label,
                    entry.text_criteria,
                    entry.map_start,
                    entry.map_end,
                )
            )
        self.db.insert_scope_measure_map(map_entries)


def _coerce_measure_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        measures = payload.get("measures")
        if isinstance(measures, list):
            return measures
    return []


def _scope_signature(scope: ScopeRecord) -> Tuple[str, str, Optional[str], date, Optional[date]]:
    return (
        scope.key,
        scope.key_type,
        scope.country_iso2,
        scope.effective_start_date,
        scope.effective_end_date,
    )


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "yes", "1"}:
            return True
        if cleaned in {"false", "no", "0"}:
            return False
    return None


def _coerce_notes(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        return {"text": value}
    return {"value": value}


def _normalize_iso_list(value: Any) -> Optional[List[str]]:
    if not value:
        return None
    if isinstance(value, list):
        items = [_normalize_iso(item) for item in value]
        return [item for item in items if item]
    if isinstance(value, str):
        return [_normalize_iso(item) for item in value.split(",") if item.strip()]
    return None


def _expand_scope_entry(entry: Dict[str, Any]) -> List[ScopeRecord]:
    key = entry.get("key") or ""
    keys = entry.get("keys")
    key_type = entry.get("key_type")
    country_iso2 = _normalize_iso(entry.get("country_iso2"))
    source_label = entry.get("source_label")
    start = _parse_date(entry.get("effective_start_date")) or DEFAULT_START_DATE
    end = _parse_date(entry.get("effective_end_date"))

    scopes: List[ScopeRecord] = []
    if keys:
        for item in _split_keys(keys):
            scopes.append(
                ScopeRecord(
                    key=_normalize_code(item),
                    key_type=_normalize_key_type(key_type, item),
                    country_iso2=country_iso2,
                    source_label=source_label,
                    effective_start_date=start,
                    effective_end_date=end,
                )
            )
        return scopes

    if not key:
        return scopes
    scopes.append(
        ScopeRecord(
            key=_normalize_code(key),
            key_type=_normalize_key_type(key_type, key),
            country_iso2=country_iso2,
            source_label=source_label,
            effective_start_date=start,
            effective_end_date=end,
        )
    )
    return scopes


def _expand_scope_entries(entries: Sequence[Dict[str, Any]]) -> List[ScopeMapEntry]:
    expanded: List[ScopeMapEntry] = []
    for entry in entries:
        relation = entry.get("relation") or "include"
        note_label = entry.get("note_label")
        text_criteria = entry.get("text_criteria")
        map_start = _parse_date(entry.get("effective_start_date")) or DEFAULT_START_DATE
        map_end = _parse_date(entry.get("effective_end_date"))
        for scope in _expand_scope_entry(entry):
            expanded.append(
                ScopeMapEntry(
                    scope=scope,
                    relation=relation,
                    note_label=note_label,
                    text_criteria=text_criteria,
                    map_start=map_start,
                    map_end=map_end,
                )
            )
    return expanded


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text


def _parse_json_payload(raw: str) -> Dict[str, Any]:
    cleaned = _strip_json_fence(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        LOGGER.error("Failed to decode JSON from LLM: %s", cleaned)
        raise RuntimeError("LLM JSON decode failed") from exc


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process Chapter 99 notes 33/36/37/38.")
    parser.add_argument("--dsn", default=os.getenv("DATABASE_URL"), help="PostgreSQL DSN")
    parser.add_argument(
        "--note",
        choices=["33", "36", "37", "38", "all"],
        default="all",
        help="Which note to process",
    )
    parser.add_argument("--table-prefix", default="otherch", help="Table prefix for measures/scope/map")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_API_BASE"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=args.log_level)
    if not args.dsn:
        raise RuntimeError("DSN is required via --dsn or DATABASE_URL")
    db = OtherChapterDB(args.dsn, table_prefix=args.table_prefix)
    llm = OtherChapterLLM(model=args.model, base_url=args.base_url, api_key=args.api_key)
    processor = OtherChapterProcessor(db, llm)
    try:
        if args.note == "all":
            for note in sorted(NOTE_LABELS):
                processor.process_note(note)
        else:
            processor.process_note(int(args.note))
    finally:
        db.close()


if __name__ == "__main__":
    main()
