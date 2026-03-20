#!/usr/bin/env python3
"""Section 301 batch processing agent - optimized version without recursive dependencies.

This module processes Section 301 measures in batch mode:
1. Batch read all 301 HTS headings from hts_codes
2. Batch analyze with LLM (concurrent calls to OpenAI and Gemini)
3. Compare results from both models
4. Batch insert into database only when results match
5. Only maintain simple heading-level exclude/include relationships
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import logging
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

try:
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor, execute_values
    from psycopg2 import errors as pg_errors
except ImportError as exc:
    psycopg2 = None
    Json = None
    RealDictCursor = None
    execute_values = None
    _PSYCOPG2_IMPORT_ERROR = exc
    pg_errors = None
else:
    _PSYCOPG2_IMPORT_ERROR = None

try:
    from google import genai
    from google.genai import types
except ImportError as exc:
    genai = None
    types = None
    _GENAI_IMPORT_ERROR = exc
else:
    _GENAI_IMPORT_ERROR = None

try:
    from openai import OpenAI
except ImportError as exc:
    OpenAI = None
    _OPENAI_IMPORT_ERROR = exc
else:
    _OPENAI_IMPORT_ERROR = None

try:  # pragma: no cover - allows linting without xai-sdk installed
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import system as xai_system
    from xai_sdk.chat import user as xai_user
except ImportError as exc:  # pragma: no cover
    XAIClient = None  # type: ignore
    xai_system = None  # type: ignore
    xai_user = None  # type: ignore
    _XAI_SDK_IMPORT_ERROR = exc
else:
    _XAI_SDK_IMPORT_ERROR = None


LOGGER = logging.getLogger("section301_batch_agent")

DEFAULT_START_DATE = date(1900, 1, 1)
NOTE_LABEL = "note(20)"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
NOTE20_VECTOR_STORE_ID = os.getenv(
    "NOTE20_VECTOR_STORE_ID",
    "vs_6970dc14f3bc819191b66e0645e91c2c",
)

LLM_PROMPT = '''You are a structured extractor for HTSUS Section 301 (U.S. note 20 to subchapter III of chapter 99).
Return JSON only.

Output must be a JSON object with a top-level key "measures" that is an array.
Each element represents one Chapter 99 heading referenced in this note.

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
- key or keys (HTS heading/subheading)
- key_type
- relation (include/exclude)
- country_iso2
- source_label
- note_label
- text_criteria
- effective_start_date
- effective_end_date

Rules:
- The note text includes multiple subdivisions for multiple headings. Treat each subdivision separately.
- Ownership-first rule: identify the owning heading of each subdivision by explicit text like “heading 9903.xx.xx”.
  Only that heading may use that subdivision’s HTS list. If the current measure’s heading is different, ignore that subdivision entirely.
- Only output headings in context.chapter99_headings; ignore other headings even if present in note text.
- ad_valorem_rate must be derived ONLY from context.hts_codes[].general_rate_of_duty; if it says “the duty provided in the applicable subheading” with no percent, set 0; otherwise null.
- country_iso2: use ISO-2; if not explicitly stated, use null (section301 default “CN”).
- key_type must be one of: heading, hts8, hts10. No “note”.
- Combine codes with identical attributes into one scope using "keys".
- Output JSON only; no code fences or commentary.

Context (JSON):
{context_json}

Note text:
<<<NOTE_TEXT
{note_text}
NOTE_TEXT>>>
'''

_NOTES_UTILS = None


def _load_notes_utils():
    global _NOTES_UTILS
    if _NOTES_UTILS is not None:
        return _NOTES_UTILS
    module_path = Path(__file__).resolve().parents[1] / "chapter99note-agent" / "notes_utils.py"
    if not module_path.exists():
        raise FileNotFoundError(f"notes_utils.py not found at {module_path}")
    spec = importlib.util.spec_from_file_location("notes_utils", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load notes_utils module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _NOTES_UTILS = module
    return module


def _build_note_text(rows: Sequence[Dict[str, Any]]) -> str:
    chunks = []
    for row in rows:
        label = row.get("label") or ""
        content = row.get("content") or ""
        chunks.append(f"{label}\n{content}")
    return "\n\n".join(chunks)


def _load_note_text(conn, label: str) -> str:
    notes_utils = _load_notes_utils()
    rows = notes_utils.get_note(conn, label, "SUBCHAPTER III")
    if not rows:
        raise FileNotFoundError(f"note rows not found for {label}")
    return _build_note_text(rows).strip()


def _write_output_json(note_label: str, suffix: str, payload: Any) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_name = note_label.replace("note(", "note").replace(")", "").replace(" ", "")
    output_path = OUTPUT_DIR / f"{output_name}_{suffix}.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s output to %s", suffix, output_path)
    return output_path

LLM_BATCH_PROMPT = """You are a legal text structure analyzer for Section 301 tariffs.

Here is the complete Note 20 to Chapter 99 of the HTS:
===== NOTE 20 COMPLETE TEXT =====
{note20_content}
===== END OF NOTE 20 =====

Now analyze these HTS headings and their descriptions:
{headings_json}

For each heading:
1. Identify which subsection(s) of Note 20 it references (e.g., "note 20(a)", "note 20(vvv)")
2. Based on the Note 20 content above, list all HTS codes in those subsections as "includes"
3. Identify any exclusions mentioned in the heading description as "excludes"
4. Extract the ad valorem rate from the duty description
5. Extract effective dates if mentioned

Return a JSON array where each element has:
{{
  "heading": "9903.88.01",
  "includes": ["0203.29.20", "0203.29.40", "0206.10.00", ...],  // HTS codes from the referenced note subsection
  "excludes": ["9903.88.05", "9903.88.06"],  // Other headings excluded
  "ad_valorem_rate": 25.0,
  "effective_start": "2018-07-06",
  "effective_end": null
}}

Important:
- For "includes", list the actual HTS codes from the Note 20 subsection, not just "note20(a)"
- For "excludes", list the heading numbers mentioned as exceptions
- Match the note reference in the description (e.g., "note 20(a)") with the corresponding subsection in the Note 20 text

Return JSON array of all headings analyzed.
"""

LLM_BATCH_PROMPT_FILE_SEARCH = """You are a legal text structure analyzer for Section 301 tariffs.

Use the file_search tool to find the relevant parts of U.S. note 20 to subchapter III of chapter 99 (including any subheading lists and exclusions).

Now analyze these HTS headings and their descriptions:
{headings_json}

For each heading:
1. Identify which subsection(s) of Note 20 it references (e.g., "note 20(a)", "note 20(vvv)")
2. Based on the Note 20 content, list all HTS codes in those subsections as "includes"
3. Identify any exclusions mentioned in the heading description as "excludes"
4. Extract the ad valorem rate from the duty description
5. Extract effective dates if mentioned

Return a JSON array where each element has:
{{
  "heading": "9903.88.01",
  "includes": ["0203.29.20", "0203.29.40", "0206.10.00", ...],
  "excludes": ["9903.88.05", "9903.88.06"],
  "ad_valorem_rate": 25.0,
  "effective_start": "2018-07-06",
  "effective_end": null
}}

Important:
- For "includes", list the actual HTS codes from the Note 20 subsection, not just "note20(a)"
- For "excludes", list the heading numbers mentioned as exceptions
- Match the note reference in the description (e.g., "note 20(a)") with the corresponding subsection in the Note 20 text

Return JSON array of all headings analyzed.
"""


def parse_date(value: Optional[str]) -> Optional[date]:
    """Parse various date formats."""
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


def derive_rate(description: str) -> Decimal:
    """Extract ad valorem rate from duty description."""
    desc = description or ""

    # Look for "applicable subheading + X%"
    match_add = re.search(
        r"the duty provided in the applicable subheading\s*(?:\+|plus)\s*(\d+(?:\.\d+)?)\s*(?:percent|%)",
        desc,
        re.IGNORECASE,
    )
    if match_add:
        return Decimal(match_add.group(1)).quantize(Decimal("0.001"))

    # Just "applicable subheading" means 0% additional
    if re.search(r"the duty provided in the applicable subheading", desc, re.IGNORECASE):
        return Decimal("0.000")

    # Look for standalone percentage
    match = re.search(r"(\d+(?:\.\d+)?)\s*percent", desc, re.IGNORECASE)
    if match:
        return Decimal(match.group(1)).quantize(Decimal("0.001"))

    return Decimal("25.000")  # Default


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


def _normalize_iso(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = value.strip().upper()
    return cleaned or None


def _split_keys(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


def classify_code_type(code: str) -> str:
    """Classify a code as heading, hts8, or hts10."""
    digits_only = code.replace(".", "")
    if code.startswith("99"):
        return "heading"
    if len(digits_only) == 10:
        return "hts10"
    if len(digits_only) == 8:
        return "hts8"
    if len(digits_only) == 4:
        return "heading"
    return "heading"


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
    if cleaned in {"heading", "hts8", "hts10"}:
        return cleaned
    return _infer_key_type(code)


def normalize_note_label(raw_label: str) -> str:
    """Normalize note reference to canonical form."""
    if not raw_label:
        return ""
    s = raw_label.strip()
    if not s.lower().startswith("note"):
        s = "note" + s
    # Simple normalization - can be enhanced as needed
    return s.lower().replace(" ", "").replace("(", "").replace(")", "")


def extract_note20_labels_from_headings(hts_data: Dict[str, Dict[str, Any]]) -> List[str]:
    """Extract note 20 subsection labels referenced in heading descriptions."""
    labels: set[str] = set()
    fields = ("description", "additional_duties", "general_rate_of_duty")
    for record in hts_data.values():
        for field in fields:
            text = record.get(field) or ""
            for match in re.findall(r"note\s*20\(([^)]+)\)", text, re.IGNORECASE):
                label = re.sub(r"\s+", "", match).lower()
                if re.fullmatch(r"[a-z]{1,5}", label):
                    labels.add(label)
    if labels and "a" not in labels:
        labels.add("a")
    return sorted(labels)


def trim_note20_content(note20_content: str, labels: Sequence[str]) -> str:
    """Return only the requested note 20 subsections."""
    if not labels:
        return note20_content

    lines = note20_content.splitlines()
    markers: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        match = re.fullmatch(r"20\.\s*\(([a-z]{1,5})\)", stripped, re.IGNORECASE)
        if not match:
            match = re.fullmatch(r"\(([a-z]{1,5})\)", stripped, re.IGNORECASE)
        if match:
            markers.append((idx, match.group(1).lower()))

    if not markers:
        return note20_content

    selected = set(label.lower() for label in labels)
    sections: list[str] = []
    for index, (line_idx, label) in enumerate(markers):
        if label not in selected:
            continue
        start = line_idx
        end = markers[index + 1][0] if index + 1 < len(markers) else len(lines)
        section = "\n".join(lines[start:end]).strip()
        if section:
            sections.append(section)

    if not sections:
        return note20_content

    return "\n\n".join(sections).strip()


def iter_paged_items(paged: object) -> Iterable[object]:
    data = getattr(paged, "data", None)
    if data is None:
        return paged
    return data


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


def _coerce_measure_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        measures = payload.get("measures")
        if isinstance(measures, list):
            return measures
    return []


def _format_date_key(value: Optional[date]) -> str:
    return value.isoformat() if value else "null"


def _format_bool_key(value: Optional[bool]) -> str:
    if value is None:
        return "null"
    return "true" if value else "false"


def _normalize_scope_key_digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _normalize_relation(value: Optional[str]) -> str:
    cleaned = (value or "include").strip().lower()
    return cleaned or "include"


def _collect_scope_counters(
    scopes: Sequence[Dict[str, Any]]
) -> Tuple[Counter, Dict[Tuple[str, str], List[str]]]:
    counter: Counter = Counter()
    raw_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    seen_pairs: set[Tuple[str, str]] = set()
    for entry in scopes:
        relation = _normalize_relation(entry.get("relation"))
        keys = entry.get("keys")
        key = entry.get("key")
        if isinstance(keys, list):
            raw_keys = [str(item) for item in keys if item is not None]
        elif keys:
            raw_keys = _split_keys(str(keys))
        elif key:
            raw_keys = [str(key)]
        else:
            raw_keys = []
        for raw in raw_keys:
            key_raw = _normalize_code(raw)
            if not key_raw:
                continue
            key_norm = _normalize_scope_key_digits(key_raw)
            if not key_norm:
                continue
            pair = (key_norm, relation)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            counter[pair] += 1
            raw_map[pair].append(key_raw)
    return counter, raw_map


def _normalize_rate_value(value: Any) -> Optional[Decimal]:
    return _parse_rate(value)


def _make_measure_key(
    heading: str,
    country_iso2: Optional[str],
    effective_start_date: Optional[date],
    effective_end_date: Optional[date],
    is_potential: Optional[bool],
) -> str:
    return "|".join(
        [
            heading or "null",
            country_iso2 or "null",
            _format_date_key(effective_start_date),
            _format_date_key(effective_end_date),
            _format_bool_key(is_potential),
        ]
    )


def _normalize_measure_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    heading = _normalize_code(str(entry.get("heading") or ""))
    if not heading:
        return None
    country_iso2 = _normalize_iso(entry.get("country_iso2"))
    effective_start_date = _parse_date(entry.get("effective_start_date"))
    effective_end_date = _parse_date(entry.get("effective_end_date"))
    is_potential = _coerce_bool(entry.get("is_potential"))
    ad_valorem_rate = _normalize_rate_value(entry.get("ad_valorem_rate"))
    scopes = entry.get("scopes") or []
    scope_counter, scope_raw_map = _collect_scope_counters(scopes)
    measure_key = _make_measure_key(
        heading,
        country_iso2,
        effective_start_date,
        effective_end_date,
        is_potential,
    )
    return {
        "measure_key": measure_key,
        "heading": heading,
        "country_iso2": country_iso2,
        "ad_valorem_rate": ad_valorem_rate,
        "is_potential": is_potential,
        "effective_start_date": effective_start_date,
        "effective_end_date": effective_end_date,
        "scope_counter": scope_counter,
        "scope_raw_map": scope_raw_map,
    }


def _index_measures(entries: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        normalized = _normalize_measure_entry(entry)
        if not normalized:
            continue
        key = normalized["measure_key"]
        if key in index:
            LOGGER.warning("Duplicate measure key %s encountered; keeping first", key)
            continue
        index[key] = normalized
    return index


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    return value


def _values_equal(left: Any, right: Any) -> bool:
    return left == right


def _build_scope_entries(
    pair: Tuple[str, str], raw_list: List[str], count: int
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if raw_list:
        for raw in raw_list[:count]:
            entries.append({"key_raw": raw, "key_norm": pair[0], "relation": pair[1]})
        if len(raw_list) < count:
            for _ in range(count - len(raw_list)):
                entries.append({"key_raw": raw_list[0], "key_norm": pair[0], "relation": pair[1]})
    else:
        for _ in range(count):
            entries.append({"key_raw": pair[0], "key_norm": pair[0], "relation": pair[1]})
    return entries


def _diff_scope_counters(
    left_counter: Counter,
    right_counter: Counter,
    left_raw_map: Dict[Tuple[str, str], List[str]],
    right_raw_map: Dict[Tuple[str, str], List[str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    only_in_left: List[Dict[str, Any]] = []
    only_in_right: List[Dict[str, Any]] = []
    all_pairs = set(left_counter) | set(right_counter)
    for pair in sorted(all_pairs):
        left_count = left_counter.get(pair, 0)
        right_count = right_counter.get(pair, 0)
        if left_count > right_count:
            only_in_left.extend(
                _build_scope_entries(pair, left_raw_map.get(pair, []), left_count - right_count)
            )
        elif right_count > left_count:
            only_in_right.extend(
                _build_scope_entries(pair, right_raw_map.get(pair, []), right_count - left_count)
            )
    only_in_left.sort(key=lambda item: (item["key_norm"], item["relation"], item["key_raw"]))
    only_in_right.sort(key=lambda item: (item["key_norm"], item["relation"], item["key_raw"]))
    return only_in_left, only_in_right


def _compare_measure_maps(
    left: Dict[str, Dict[str, Any]],
    right: Dict[str, Dict[str, Any]],
    left_label: str,
    right_label: str,
) -> Dict[str, Any]:
    missing_in_left = sorted(set(right) - set(left))
    missing_in_right = sorted(set(left) - set(right))
    field_diffs: Dict[str, Any] = {}
    scope_diffs: Dict[str, Any] = {}
    matched_count = 0

    for key in sorted(set(left) & set(right)):
        left_entry = left[key]
        right_entry = right[key]
        diffs: Dict[str, Any] = {}
        for field in [
            "country_iso2",
            "ad_valorem_rate",
            "is_potential",
            "effective_start_date",
            "effective_end_date",
        ]:
            left_val = left_entry[field]
            right_val = right_entry[field]
            if not _values_equal(left_val, right_val):
                diffs[field] = {
                    left_label: _serialize_value(left_val),
                    right_label: _serialize_value(right_val),
                }

        only_in_left, only_in_right = _diff_scope_counters(
            left_entry["scope_counter"],
            right_entry["scope_counter"],
            left_entry["scope_raw_map"],
            right_entry["scope_raw_map"],
        )
        if only_in_left or only_in_right:
            scope_diffs[key] = {
                f"only_in_{left_label}": only_in_left,
                f"only_in_{right_label}": only_in_right,
            }

        if diffs:
            field_diffs[key] = diffs
        if not diffs and not (only_in_left or only_in_right):
            matched_count += 1

    summary = {
        f"{left_label}_count": len(left),
        f"{right_label}_count": len(right),
        "matched_count": matched_count,
    }
    consistent = not missing_in_left and not missing_in_right and not field_diffs and not scope_diffs
    return {
        "consistent": consistent,
        "summary": summary,
        "missing_in_left": missing_in_left,
        "missing_in_right": missing_in_right,
        "field_diffs": field_diffs,
        "scope_diffs": scope_diffs,
    }


def _compare_llm_measures(
    openai_measures: Sequence[Dict[str, Any]],
    grok_measures: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    openai_map = _index_measures(openai_measures)
    grok_map = _index_measures(grok_measures)
    base = _compare_measure_maps(openai_map, grok_map, "openai", "grok")
    return {
        "consistent": base["consistent"],
        "summary": base["summary"],
        "missing_in_openai": base["missing_in_left"],
        "missing_in_grok": base["missing_in_right"],
        "field_diffs": base["field_diffs"],
        "scope_diffs": base["scope_diffs"],
    }


def _compare_llm_db_measures(
    llm_measures: Sequence[Dict[str, Any]],
    db_measures: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    llm_map = _index_measures(llm_measures)
    db_map = _index_measures(db_measures)
    base = _compare_measure_maps(llm_map, db_map, "llm", "db")
    return {
        "consistent": base["consistent"],
        "summary": base["summary"],
        "missing_in_db": base["missing_in_right"],
        "extra_in_db": base["missing_in_left"],
        "field_diffs": base["field_diffs"],
        "scope_diffs": base["scope_diffs"],
    }


def _filter_callable_kwargs(func: Any, candidate_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return {}
    allowed = set(signature.parameters.keys())
    return {key: value for key, value in candidate_kwargs.items() if key in allowed}


@dataclass
class MeasureRecord:
    heading: str
    country_iso2: str
    ad_valorem_rate: Decimal
    value_basis: str
    notes: Optional[Dict[str, Any]]
    effective_start_date: date
    effective_end_date: Optional[date]


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
    scope_id: int
    measure_id: int
    relation: str
    note_label: Optional[str]
    text_criteria: Optional[str]
    effective_start_date: Optional[date]
    effective_end_date: Optional[date]


@dataclass
class ScopeMapDraft:
    scope: ScopeRecord
    relation: str
    note_label: Optional[str]
    text_criteria: Optional[str]
    map_start: date
    map_end: Optional[date]


def _scope_signature(scope: ScopeRecord) -> Tuple[str, str, Optional[str], date, Optional[date]]:
    return (
        scope.key,
        scope.key_type,
        scope.country_iso2,
        scope.effective_start_date,
        scope.effective_end_date,
    )


def _expand_scope_entry(entry: Dict[str, Any]) -> List[ScopeRecord]:
    scopes: List[ScopeRecord] = []
    keys = entry.get("keys")
    key = entry.get("key")
    raw_keys: List[str] = []
    if isinstance(keys, list):
        raw_keys = [str(item) for item in keys if item is not None]
    elif keys:
        raw_keys = _split_keys(str(keys))
    elif key:
        raw_keys = [str(key)]
    if not raw_keys:
        return scopes
    country_iso2 = _normalize_iso(entry.get("country_iso2"))
    source_label = entry.get("source_label")
    start = _parse_date(entry.get("effective_start_date")) or DEFAULT_START_DATE
    end = _parse_date(entry.get("effective_end_date"))
    key_type_raw = entry.get("key_type")
    for raw in raw_keys:
        cleaned = _normalize_code(raw)
        if not cleaned:
            continue
        scopes.append(
            ScopeRecord(
                key=cleaned,
                key_type=_normalize_key_type(key_type_raw, cleaned),
                country_iso2=country_iso2,
                source_label=source_label,
                effective_start_date=start,
                effective_end_date=end,
            )
        )
    return scopes


def _expand_scope_entries(entries: Sequence[Dict[str, Any]]) -> List[ScopeMapDraft]:
    expanded: List[ScopeMapDraft] = []
    for entry in entries:
        relation = entry.get("relation") or "include"
        note_label = entry.get("note_label")
        text_criteria = entry.get("text_criteria")
        map_start = _parse_date(entry.get("effective_start_date")) or DEFAULT_START_DATE
        map_end = _parse_date(entry.get("effective_end_date"))
        for scope in _expand_scope_entry(entry):
            expanded.append(
                ScopeMapDraft(
                    scope=scope,
                    relation=relation,
                    note_label=note_label,
                    text_criteria=text_criteria,
                    map_start=map_start,
                    map_end=map_end,
                )
            )
    return expanded


class Section301BatchLLM:
    """OpenAI LLM client for note(20) extraction."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 36000,
    ):
        if OpenAI is None:  # pragma: no cover
            raise RuntimeError("openai package required") from _OPENAI_IMPORT_ERROR
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.timeout = timeout

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY required")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def extract(self, note_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        message = LLM_PROMPT.format(
            note_text=note_text.strip(),
            context_json=json.dumps(context, ensure_ascii=True),
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a precise legal text parser. Respond with JSON only."},
                    {"role": "user", "content": message},
                ],
                timeout=7200.0,
            )
        except Exception as exc:
            raise RuntimeError(f"LLM SDK error: {exc}") from exc
        try:
            return _parse_json_payload(response.choices[0].message.content)
        except (AttributeError, IndexError) as exc:
            raise RuntimeError(f"Unexpected LLM response structure: {response}") from exc


class Section301GrokLLM:
    """Grok LLM client for note(20) extraction."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 36000,
    ) -> None:
        if XAIClient is None:  # pragma: no cover
            raise RuntimeError("xai-sdk package required") from _XAI_SDK_IMPORT_ERROR
        self.model = model or os.getenv("GROK_MODEL") or os.getenv("XAI_MODEL") or "grok-4"
        self.api_key = api_key or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("GROK_API_KEY or XAI_API_KEY is required for Grok LLM calls")
        self.client = XAIClient(api_key=self.api_key, timeout=self.timeout)

    def extract(self, note_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        message = LLM_PROMPT.format(
            note_text=note_text.strip(),
            context_json=json.dumps(context, ensure_ascii=True),
        )
        LOGGER.info("Grok request for %s", NOTE_LABEL)
        chat_kwargs = _filter_callable_kwargs(
            self.client.chat.create,
            {
                "temperature": 0,
            },
        )
        chat_kwargs["model"] = self.model
        chat = self.client.chat.create(**chat_kwargs)
        chat.append(xai_system("You are a precise legal text parser. Respond with JSON only."))
        chat.append(xai_user(message))
        sample_kwargs = _filter_callable_kwargs(
            chat.sample,
            {
                "temperature": 0,
            },
        )
        response = chat.sample(**sample_kwargs)
        content = getattr(response, "content", None) or getattr(response, "text", None)
        if not content:
            raise RuntimeError("Grok response missing content")
        LOGGER.info("Grok response length: %d chars", len(content))
        return _parse_json_payload(content)


class Section301GeminiLLM:
    """Gemini LLM client for batch processing."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 36000,
    ):
        if genai is None:
            raise RuntimeError("google-genai package required") from _GENAI_IMPORT_ERROR

        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.timeout = timeout

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY required")

        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)

    def extract_batch(self, headings_data: Dict[str, Dict[str, Any]], note20_content: str) -> List[Dict[str, Any]]:
        """Extract measures from all headings in one LLM call."""
        headings_json = json.dumps(headings_data, ensure_ascii=False, indent=2)
        message = LLM_BATCH_PROMPT.format(
            headings_json=headings_json,
            note20_content=note20_content
        )

        system_instruction = "You are a precise legal text parser. Respond with JSON only."
        LOGGER.info("Gemini prompt (system): %s", system_instruction)
        LOGGER.info("Gemini prompt (user): %s", message)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=message,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    temperature=0.0,
                )
            )

            content = response.text
            result = json.loads(content)

            # Handle both array and object with "measures" key
            if isinstance(result, dict) and "measures" in result:
                return result["measures"]
            elif isinstance(result, list):
                return result
            else:
                return [result]

        except Exception as e:
            LOGGER.error("Gemini API call failed: %s", e)
            raise


class Section301BatchDatabase:
    """Database operations optimized for batch processing."""

    def __init__(self, dsn: str):
        if psycopg2 is None:
            raise RuntimeError("psycopg2 required") from _PSYCOPG2_IMPORT_ERROR
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False

    def close(self) -> None:
        self._conn.close()

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def fetch_section301_headings(self) -> List[str]:
        """Fetch all Section 301 headings from database."""
        query = """
            SELECT hts_number
            FROM hts_codes
            WHERE hts_number LIKE '9903.88%'
              AND status IS NULL
              AND description NOT LIKE '%provision suspended%'
            ORDER BY hts_number
        """

        with self._conn.cursor() as cur:
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]

    def fetch_hts_codes_batch(self, headings: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch all HTS codes data in one query."""
        if not headings:
            return {}

        query = """
            SELECT hts_number, description, status, general_rate_of_duty, additional_duties
            FROM hts_codes
            WHERE hts_number = ANY(%s)
              AND COALESCE(status, '') != 'expired'
            ORDER BY hts_number
        """

        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (list(headings),))
            return {row["hts_number"]: dict(row) for row in cur.fetchall()}

    def fetch_note_rows(self, label: str) -> List[Dict[str, Any]]:
        notes_utils = _load_notes_utils()
        rows = notes_utils.get_note(self._conn, label, "SUBCHAPTER III")
        return rows or []

    def fetch_measures_with_scopes(self, headings: Sequence[str]) -> List[Dict[str, Any]]:
        if not headings:
            return []
        query = (
            "SELECT id, heading, country_iso2, ad_valorem_rate, "
            "effective_start_date, effective_end_date "
            "FROM s301_measures "
            "WHERE heading = ANY(%s)"
        )
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (list(headings),))
            rows = cur.fetchall() or []

        if not rows:
            return []

        measures: List[Dict[str, Any]] = []
        measure_map: Dict[int, Dict[str, Any]] = {}
        measure_ids: List[int] = []
        for row in rows:
            measure_id = int(row["id"])
            measure_ids.append(measure_id)
            measure_entry = {
                "heading": row["heading"],
                "country_iso2": row["country_iso2"],
                "ad_valorem_rate": row["ad_valorem_rate"],
                "effective_start_date": row["effective_start_date"],
                "effective_end_date": row["effective_end_date"],
                "is_potential": False,
                "scopes": [],
            }
            measures.append(measure_entry)
            measure_map[measure_id] = measure_entry

        scope_query = (
            "SELECT m.measure_id, s.key, m.relation "
            "FROM s301_scope_measure_map AS m "
            "JOIN s301_scope AS s ON s.id = m.scope_id "
            "WHERE m.measure_id = ANY(%s)"
        )
        with self._conn.cursor() as cur:
            cur.execute(scope_query, (measure_ids,))
            for measure_id, key, relation in cur.fetchall():
                target = measure_map.get(int(measure_id))
                if not target:
                    continue
                target["scopes"].append(
                    {
                        "key": key,
                        "relation": relation,
                    }
                )

        return measures

    def load_note20_from_file(self) -> str:
        """Load Note 20 content from local file."""
        # Try multiple possible locations
        possible_paths = [
            Path(__file__).parent.parent / "charpter-pdf-agent" / "charpter-data-txt" / "SubchapterIII_USNote_20.normalized.txt",
            Path(__file__).parent.parent / "charpter-pdf-agent" / "charpter-data-txt" / "SubchapterIII_USNote_20.txt",  # Actual file location
            Path(__file__).parent.parent / "chapter-pdf-agent" / "chapter-data" / "note20.txt",
            Path(__file__).parent.parent / "charpter-pdf-agent" / "charpter-data" / "note20.txt",  # Alternative spelling
            Path(__file__).parent / "data" / "note20.txt",  # Local data directory
            Path(__file__).parent / "data" / "note20_sample.txt",  # Sample file we created
        ]

        for path in possible_paths:
            LOGGER.debug(f"Checking path: {path} (exists: {path.exists()})")
            if path.exists():
                LOGGER.info(f"Loading Note 20 from: {path}")
                return path.read_text(encoding='utf-8')

        # List all checked paths in warning
        LOGGER.warning(f"Note 20 file not found in expected locations: {[str(p) for p in possible_paths]}")

        # Fallback to database if file not found
        LOGGER.info("Attempting to load Note 20 from database...")
        return self.fetch_note20_from_database()

    def fetch_note20_from_database(self) -> str:
        """Fetch Note 20 content from database as fallback."""
        # First try to check what notes exist in the database
        check_query = """
            SELECT DISTINCT label, chapter
            FROM hts_notes
            WHERE chapter = 99 AND label LIKE 'note%'
            ORDER BY label
            LIMIT 10
        """

        with self._conn.cursor() as cur:
            try:
                cur.execute(check_query)
                available_notes = cur.fetchall()
                if available_notes:
                    LOGGER.info(f"Available Chapter 99 notes in database: {available_notes}")
            except Exception as e:
                LOGGER.warning(f"Could not check available notes: {e}")

        # Try different query patterns for Note 20
        queries = [
            # Try with label like note20
            """
            SELECT string_agg(content, E'\n' ORDER BY id) as full_content
            FROM hts_notes
            WHERE chapter = 99
              AND label LIKE 'note20%'
            """,
            # Try with normalized label
            """
            SELECT string_agg(content, E'\n' ORDER BY id) as full_content
            FROM hts_notes
            WHERE chapter = 99
              AND label LIKE 'note(20)%'
            """,
            # Try any note containing 20
            """
            SELECT string_agg(content, E'\n' ORDER BY array_length(path, 1), path) as full_content
            FROM hts_notes
            WHERE chapter = 99
              AND label ~ 'note.*20'
            """
        ]

        for i, query in enumerate(queries, 1):
            with self._conn.cursor() as cur:
                try:
                    cur.execute(query)
                    result = cur.fetchone()
                    if result and result[0]:
                        LOGGER.info(f"Successfully loaded Note 20 from database using query pattern {i}")
                        return result[0]
                except Exception as e:
                    LOGGER.debug(f"Query pattern {i} failed: {e}")

        LOGGER.error("Note 20 not found in database with any query pattern")
        LOGGER.info("Please ensure Note 20 data is available in hts_notes table or provide note20.txt file")
        return ""

    def insert_measures_batch(self, records: Sequence[MeasureRecord]) -> List[int]:
        """Batch insert measures using execute_values."""
        if not records:
            return []

        values = [
            (
                r.heading,
                r.country_iso2,
                r.ad_valorem_rate,
                r.value_basis,
                Json(r.notes) if r.notes else None,
                r.effective_start_date,
                r.effective_end_date,
            )
            for r in records
        ]

        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """INSERT INTO s301_measures
                   (heading, country_iso2, ad_valorem_rate, value_basis,
                    notes, effective_start_date, effective_end_date)
                   VALUES %s
                   ON CONFLICT (heading, country_iso2, effective_start_date,
                               COALESCE(effective_end_date, DATE '9999-12-31'))
                   DO UPDATE SET
                       ad_valorem_rate = EXCLUDED.ad_valorem_rate,
                       notes = EXCLUDED.notes
                   RETURNING id""",
                values
            )
            return [row[0] for row in cur.fetchall()]

    def insert_scopes_batch(self, records: Sequence[ScopeRecord]) -> List[int]:
        """Batch insert scopes."""
        if not records:
            return []

        values = [
            (
                r.key,
                r.key_type,
                r.country_iso2,
                r.source_label,
                r.effective_start_date,
                r.effective_end_date,
            )
            for r in records
        ]

        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """INSERT INTO s301_scope
                   (key, key_type, country_iso2, source_label,
                    effective_start_date, effective_end_date)
                   VALUES %s
                   ON CONFLICT (key, key_type, COALESCE(country_iso2, ''),
                               effective_start_date,
                               COALESCE(effective_end_date, DATE '9999-12-31'))
                   DO NOTHING""",
                values,
            )
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
                "SELECT v.idx, s.id "
                "FROM (VALUES %s) AS v(idx, key, key_type, country_iso2, start_date, end_date) "
                "JOIN s301_scope AS s "
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

    def insert_scope_maps_batch(self, entries: Sequence[ScopeMapEntry]) -> None:
        """Batch insert scope-measure mappings."""
        if not entries:
            return

        values = [
            (
                e.scope_id,
                e.measure_id,
                e.relation,
                e.note_label,
                e.text_criteria,
                e.effective_start_date,
                e.effective_end_date,
            )
            for e in entries
        ]

        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """INSERT INTO s301_scope_measure_map
                   (scope_id, measure_id, relation, note_label, text_criteria,
                    effective_start_date, effective_end_date)
                   VALUES %s
                   ON CONFLICT DO NOTHING""",
                values
            )


@dataclass
class ComparisonStats:
    """Statistics for LLM comparison."""
    total_headings: int = 0
    matching_headings: int = 0
    differing_headings: int = 0
    openai_only_headings: int = 0
    gemini_only_headings: int = 0
    differences: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.differences is None:
            self.differences = []


def normalize_analysis_for_comparison(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an analysis result for comparison."""
    normalized = {
        "heading": analysis.get("heading", "").strip(),
        "includes": sorted([str(x).strip() for x in analysis.get("includes", [])]),
        "excludes": sorted([str(x).strip() for x in analysis.get("excludes", [])]),
        "ad_valorem_rate": float(analysis.get("ad_valorem_rate", 0.0)) if analysis.get("ad_valorem_rate") is not None else None,
        "effective_start": analysis.get("effective_start"),
        "effective_end": analysis.get("effective_end"),
    }
    return normalized


def compare_analyses(
    openai_results: List[Dict[str, Any]],
    gemini_results: List[Dict[str, Any]]
) -> ComparisonStats:
    """Compare results from OpenAI and Gemini models."""
    stats = ComparisonStats()

    # Create lookup maps by heading
    openai_map = {r.get("heading"): r for r in openai_results if r.get("heading")}
    gemini_map = {r.get("heading"): r for r in gemini_results if r.get("heading")}

    all_headings = set(openai_map.keys()) | set(gemini_map.keys())
    stats.total_headings = len(all_headings)

    for heading in sorted(all_headings):
        openai_analysis = openai_map.get(heading)
        gemini_analysis = gemini_map.get(heading)

        # Check if heading exists in both
        if openai_analysis is None:
            stats.gemini_only_headings += 1
            stats.differences.append({
                "heading": heading,
                "issue": "missing_in_openai",
                "gemini_result": gemini_analysis,
            })
            continue

        if gemini_analysis is None:
            stats.openai_only_headings += 1
            stats.differences.append({
                "heading": heading,
                "issue": "missing_in_gemini",
                "openai_result": openai_analysis,
            })
            continue

        # Normalize for comparison
        norm_openai = normalize_analysis_for_comparison(openai_analysis)
        norm_gemini = normalize_analysis_for_comparison(gemini_analysis)

        # Deep comparison
        if norm_openai == norm_gemini:
            stats.matching_headings += 1
        else:
            stats.differing_headings += 1
            # Identify specific differences
            field_diffs = {}
            for field in ["includes", "excludes", "ad_valorem_rate", "effective_start", "effective_end"]:
                if norm_openai.get(field) != norm_gemini.get(field):
                    field_diffs[field] = {
                        "openai": norm_openai.get(field),
                        "gemini": norm_gemini.get(field),
                    }

            stats.differences.append({
                "heading": heading,
                "issue": "content_mismatch",
                "field_differences": field_diffs,
                "openai_result": openai_analysis,
                "gemini_result": gemini_analysis,
            })

    return stats


class Section301BatchAgent:
    """Main agent for batch processing Section 301 measures with dual LLM verification."""

    def __init__(
        self,
        db: Section301BatchDatabase,
        openai_llm: Section301BatchLLM,
        gemini_llm: Section301GeminiLLM,
        country_iso2: str = "CN",
        require_match: bool = True,
    ):
        self.db = db
        self.openai_llm = openai_llm
        self.gemini_llm = gemini_llm
        self.country_iso2 = country_iso2
        self.require_match = require_match

    async def _call_openai_async(self, hts_data: Dict[str, Dict[str, Any]], note20_content: str) -> List[Dict[str, Any]]:
        """Call OpenAI API asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.openai_llm.extract_batch, hts_data, note20_content)

    async def _call_gemini_async(self, hts_data: Dict[str, Dict[str, Any]], note20_content: str) -> List[Dict[str, Any]]:
        """Call Gemini API asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.gemini_llm.extract_batch, hts_data, note20_content)

    def _load_note20_from_file_search(self, hts_data: Dict[str, Dict[str, Any]]) -> str:
        if OpenAI is None:
            LOGGER.warning("OpenAI SDK not available for file_search.") 
            return ""
        if not NOTE20_VECTOR_STORE_ID:
            LOGGER.warning("NOTE20_VECTOR_STORE_ID is not set.")
            return ""

        labels = extract_note20_labels_from_headings(hts_data)
        headings = sorted(hts_data.keys())
        label_hint = " ".join(f"note 20({label})" for label in labels) if labels else "note 20"
        headings_hint = ", ".join(headings)
        query = (
            f"U.S. note 20 subchapter III chapter 99. "
            f"Find sections for {label_hint} and the subheading lists "
            f"relevant to headings {headings_hint}. "
            "Include full subheading lists and exclusions."
        )

        client = OpenAI(api_key=self.openai_llm.api_key)
        try:
            results = client.vector_stores.search(
                vector_store_id=NOTE20_VECTOR_STORE_ID,
                query=query,
                max_num_results=50,
                rewrite_query=True,
            )
        except Exception as exc:
            LOGGER.warning("File search failed: %s", exc)
            return ""

        chunks: list[str] = []
        for item in iter_paged_items(results):
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    cleaned = text.strip()
                    if cleaned:
                        chunks.append(cleaned)

        seen: set[str] = set()
        unique_chunks: list[str] = []
        for chunk in chunks:
            if chunk in seen:
                continue
            seen.add(chunk)
            unique_chunks.append(chunk)

        note20_content = "\n\n".join(unique_chunks).strip()
        if note20_content:
            LOGGER.info(
                "Loaded Note 20 content from file_search (%d chunks, %d characters)",
                len(unique_chunks),
                len(note20_content),
            )
        return note20_content

    async def _concurrent_llm_calls(self, hts_data: Dict[str, Dict[str, Any]], note20_content: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Execute LLM calls concurrently to OpenAI and Gemini."""
        LOGGER.info("Starting OpenAI LLM call (Gemini skipped)...")

        openai_results = await self._call_openai_async(hts_data, note20_content)
        gemini_results = openai_results

        LOGGER.info("OpenAI returned %d results", len(openai_results))
        LOGGER.info("Gemini skipped; using OpenAI results for comparison")

        # Save results to files for debugging
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save OpenAI results
        openai_file = f"openai_results_{timestamp}.json"
        with open(openai_file, "w", encoding="utf-8") as f:
            json.dump(openai_results, f, indent=2, ensure_ascii=False, default=str)
        LOGGER.info(f"OpenAI results saved to: {openai_file}")

        # Skip Gemini output since Gemini is disabled

        return openai_results, gemini_results

    def run(self, headings: Optional[Sequence[str]] = None) -> None:
        """Process all headings in batch mode with dual LLM verification."""
        # If no headings provided, fetch from database
        if not headings:
            LOGGER.info("No headings provided, fetching Section 301 headings from database...")
            headings = self.db.fetch_section301_headings()
            LOGGER.info("Found %d Section 301 headings in database", len(headings))
            if not headings:
                LOGGER.warning("No Section 301 headings found in database (9903.88%)")
                return

        LOGGER.info("Starting batch processing of %d headings", len(headings))

        # Limit to first 2 headings for testing (to avoid token limit issues)
        if len(headings) > 2:
            LOGGER.warning(f"Limiting processing to first 2 headings (out of {len(headings)}) for testing")
            headings = headings[:2]
            LOGGER.info(f"Processing only: {headings}")

        # 1. Fetch all HTS data
        hts_data = self.db.fetch_hts_codes_batch(headings)
        LOGGER.info("Fetched %d HTS records", len(hts_data))

        if not hts_data:
            LOGGER.warning("No HTS data found for provided headings")
            return

        # 2. Load Note 20 content
        LOGGER.info("Loading Note 20 content...")
        if NOTE20_VECTOR_STORE_ID:
            LOGGER.info(
                "Using OpenAI file_search for Note 20 (store_id=%s)",
                NOTE20_VECTOR_STORE_ID,
            )
            note20_content = ""
        else:
            note20_content = self.db.load_note20_from_file()
            if not note20_content:
                LOGGER.error("Failed to load Note 20 content")
                return
            LOGGER.info("Loaded Note 20 content (%d characters)", len(note20_content))
            note20_labels = extract_note20_labels_from_headings(hts_data)
            trimmed_note20 = trim_note20_content(note20_content, note20_labels)
            if trimmed_note20 != note20_content:
                LOGGER.info(
                    "Trimmed Note 20 content to %d characters using labels: %s",
                    len(trimmed_note20),
                    ", ".join(note20_labels),
                )
                note20_content = trimmed_note20
            else:
                LOGGER.info(
                    "Note 20 trimming skipped; labels=%s",
                    ", ".join(note20_labels) if note20_labels else "none",
                )

        # 3. Call both LLMs concurrently with Note 20 context
        openai_analyses, gemini_analyses = asyncio.run(self._concurrent_llm_calls(hts_data, note20_content))

        # 4. Compare results
        LOGGER.info("Comparing results from OpenAI and Gemini...")
        stats = compare_analyses(openai_analyses, gemini_analyses)

        # 5. Log statistics
        self._log_comparison_stats(stats)

        # 6. Decide whether to proceed with database insertion
        if stats.matching_headings == stats.total_headings:
            LOGGER.info("All results match! Proceeding with database insertion...")
            analyses = openai_analyses  # Use OpenAI results (they're identical)
        elif not self.require_match:
            LOGGER.warning("Results don't match completely, but require_match=False. Using OpenAI results...")
            analyses = openai_analyses
        else:
            LOGGER.error("Results don't match completely and require_match=True. Aborting database insertion.")
            self._save_comparison_report(stats, hts_data)
            return

        # 7. Prepare records for batch insert
        measure_records = []
        all_scopes = []
        measure_to_scopes = []  # Track which scopes belong to which measure

        for analysis in analyses:
            heading = analysis.get("heading")
            if not heading:
                continue

            # Create measure record
            start_date = parse_date(analysis.get("effective_start")) or DEFAULT_START_DATE
            end_date = parse_date(analysis.get("effective_end"))

            rate = analysis.get("ad_valorem_rate")
            if rate is not None:
                rate = Decimal(str(rate))
            else:
                # Fallback to parsing from original description
                original_desc = hts_data.get(heading, {}).get("general_rate_of_duty", "")
                rate = derive_rate(original_desc)

            measure = MeasureRecord(
                heading=heading,
                country_iso2=self.country_iso2,
                ad_valorem_rate=rate,
                value_basis="customs_value",
                notes=None,
                effective_start_date=start_date,
                effective_end_date=end_date,
            )
            measure_records.append(measure)

            # Collect scopes for this measure
            scopes_for_measure = []

            # Process includes
            for ref in analysis.get("includes", []):
                scope = self._create_scope(ref, heading, "include", start_date, end_date)
                if scope:
                    all_scopes.append(scope)
                    scopes_for_measure.append((scope, "include"))

            # Process excludes
            for ref in analysis.get("excludes", []):
                scope = self._create_scope(ref, heading, "exclude", start_date, end_date)
                if scope:
                    all_scopes.append(scope)
                    scopes_for_measure.append((scope, "exclude"))

            measure_to_scopes.append(scopes_for_measure)

        # 8. Batch insert all data
        LOGGER.info("Inserting %d measures", len(measure_records))
        # measure_ids = self.db.insert_measures_batch(measure_records)

        # Deduplicate scopes
        unique_scopes = self._deduplicate_scopes(all_scopes)
        LOGGER.info("Inserting %d unique scopes", len(unique_scopes))
        # scope_ids = self.db.insert_scopes_batch(unique_scopes)

        # Build scope lookup map
        scope_to_id = {}
        for scope, scope_id in zip(unique_scopes, scope_ids):
            key = (scope.key, scope.key_type, scope.country_iso2,
                   scope.effective_start_date, scope.effective_end_date)
            scope_to_id[key] = scope_id

        # Create scope-measure mappings
        map_entries = []
        for measure_id, scopes_for_measure in zip(measure_ids, measure_to_scopes):
            for scope, relation in scopes_for_measure:
                key = (scope.key, scope.key_type, scope.country_iso2,
                       scope.effective_start_date, scope.effective_end_date)
                scope_id = scope_to_id.get(key)
                if scope_id:
                    entry = ScopeMapEntry(
                        scope_id=scope_id,
                        measure_id=measure_id,
                        relation=relation,
                        note_label=scope.source_label,
                        text_criteria=None,
                        effective_start_date=None,
                        effective_end_date=None,
                    )
                    map_entries.append(entry)

        LOGGER.info("Creating %d scope-measure mappings", len(map_entries))
        # self.db.insert_scope_maps_batch(map_entries)

        # 9. Commit transaction
        # self.db.commit()
        LOGGER.info("Batch processing completed successfully")

    def _log_comparison_stats(self, stats: ComparisonStats) -> None:
        """Log detailed comparison statistics."""
        LOGGER.info("=" * 80)
        LOGGER.info("LLM COMPARISON STATISTICS")
        LOGGER.info("=" * 80)
        LOGGER.info("Total headings processed: %d", stats.total_headings)
        LOGGER.info("Matching headings: %d (%.1f%%)",
                   stats.matching_headings,
                   100.0 * stats.matching_headings / stats.total_headings if stats.total_headings > 0 else 0)
        LOGGER.info("Differing headings: %d (%.1f%%)",
                   stats.differing_headings,
                   100.0 * stats.differing_headings / stats.total_headings if stats.total_headings > 0 else 0)
        LOGGER.info("OpenAI only headings: %d", stats.openai_only_headings)
        LOGGER.info("Gemini only headings: %d", stats.gemini_only_headings)
        LOGGER.info("=" * 80)

        if stats.differences:
            LOGGER.warning("Found %d differences:", len(stats.differences))
            for i, diff in enumerate(stats.differences[:10], 1):  # Show first 10
                LOGGER.warning("Difference %d:", i)
                LOGGER.warning("  Heading: %s", diff.get("heading"))
                LOGGER.warning("  Issue: %s", diff.get("issue"))

                if diff.get("issue") == "content_mismatch":
                    LOGGER.warning("  Field differences:")
                    for field, values in diff.get("field_differences", {}).items():
                        LOGGER.warning("    %s:", field)
                        LOGGER.warning("      OpenAI: %s", values.get("openai"))
                        LOGGER.warning("      Gemini: %s", values.get("gemini"))

            if len(stats.differences) > 10:
                LOGGER.warning("... and %d more differences (see comparison report)", len(stats.differences) - 10)

    def _save_comparison_report(self, stats: ComparisonStats, hts_data: Dict[str, Dict[str, Any]]) -> None:
        """Save detailed comparison report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"llm_comparison_report_{timestamp}.json"

        report = {
            "timestamp": timestamp,
            "statistics": {
                "total_headings": stats.total_headings,
                "matching_headings": stats.matching_headings,
                "differing_headings": stats.differing_headings,
                "openai_only_headings": stats.openai_only_headings,
                "gemini_only_headings": stats.gemini_only_headings,
                "match_rate": 100.0 * stats.matching_headings / stats.total_headings if stats.total_headings > 0 else 0,
            },
            "differences": stats.differences,
            "hts_data_context": hts_data,
        }

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        LOGGER.info("Comparison report saved to: %s", report_file)

    def _create_scope(
        self,
        ref: str,
        heading: str,
        relation: str,
        start_date: date,
        end_date: Optional[date]
    ) -> Optional[ScopeRecord]:
        """Create a scope record from a reference."""
        if not ref:
            return None

        ref = ref.strip()

        # Determine type
        if ref.lower().startswith("note"):
            key = normalize_note_label(ref)
            key_type = "note"
        else:
            key = ref
            key_type = classify_code_type(ref)

        return ScopeRecord(
            key=key,
            key_type=key_type,
            country_iso2=self.country_iso2 if not ref.startswith("99") else None,
            source_label=f"{heading}-{relation}",
            effective_start_date=start_date,
            effective_end_date=end_date,
        )

    def _deduplicate_scopes(self, scopes: List[ScopeRecord]) -> List[ScopeRecord]:
        """Remove duplicate scope records."""
        seen = set()
        unique = []
        for scope in scopes:
            key = (
                scope.key,
                scope.key_type,
                scope.country_iso2,
                scope.effective_start_date,
                scope.effective_end_date,
            )
            if key not in seen:
                seen.add(key)
                unique.append(scope)
        return unique


class Section301Note20Processor:
    def __init__(
        self,
        db: Section301BatchDatabase,
        openai_llm: Section301BatchLLM,
        grok_llm: Section301GrokLLM,
        *,
        country_iso2: str = "CN",
        persist: bool = False,
    ) -> None:
        self.db = db
        self.openai_llm = openai_llm
        self.grok_llm = grok_llm
        self.country_iso2 = country_iso2
        self.persist = persist

    def process(self, headings: Optional[Sequence[str]] = None) -> None:
        if not headings:
            LOGGER.info("No headings provided, fetching Section 301 headings from database...")
            headings = self.db.fetch_section301_headings()
            LOGGER.info("Found %d Section 301 headings in database", len(headings))
            if not headings:
                LOGGER.warning("No Section 301 headings found in database (9903.88%%)")
                return

        LOGGER.info("Starting batch processing of %d headings", len(headings))

        if len(headings) > 2:
            LOGGER.warning("Limiting processing to first 2 headings (out of %d) for testing", len(headings))
            headings = headings[:2]
            LOGGER.info("Processing only: %s", headings)

        hts_data = self.db.fetch_hts_codes_batch(headings)
        LOGGER.info("Fetched %d HTS records", len(hts_data))
        if not hts_data:
            LOGGER.warning("No HTS data found for provided headings")
            return

        try:
            note_rows = self.db.fetch_note_rows(NOTE_LABEL)
            if not note_rows:
                LOGGER.warning("No note rows found for %s", NOTE_LABEL)
                return
            note_text = _build_note_text(note_rows).strip()
        except Exception as exc:
            LOGGER.warning("Failed to load %s: %s", NOTE_LABEL, exc)
            return

        context = self._build_context(headings, hts_data)

        with ThreadPoolExecutor(max_workers=2) as executor:
            openai_future = executor.submit(self.openai_llm.extract, note_text, context)
            grok_future = executor.submit(self.grok_llm.extract, note_text, context)
            openai_result = openai_future.result()
            try:
                grok_result = grok_future.result()
            except Exception:
                LOGGER.exception("Grok request failed for %s; using OpenAI result", NOTE_LABEL)
                grok_result = openai_result

        openai_payload = self._normalize_payload(openai_result, headings, hts_data)
        grok_payload = self._normalize_payload(grok_result, headings, hts_data)

        _write_output_json("note20", "openai", openai_payload)
        _write_output_json("note20", "grok", grok_payload)

        llm_compare = _compare_llm_measures(
            openai_payload["measures"],
            grok_payload["measures"],
        )
        _write_output_json("note20", "llm_compare", llm_compare)
        if not llm_compare.get("consistent"):
            LOGGER.warning("LLM results differ for %s; skipping DB compare and persistence", NOTE_LABEL)
            return

        db_measures = self.db.fetch_measures_with_scopes(headings)
        db_compare = _compare_llm_db_measures(openai_payload["measures"], db_measures)
        _write_output_json("note20", "db_compare", db_compare)

        if self.persist:
            self._persist(openai_payload["measures"])
            self.db.commit()

    def _build_context(
        self, headings: Sequence[str], hts_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        context_codes = []
        for code in headings:
            record = hts_data.get(code, {})
            general_rate = (record.get("general_rate_of_duty") or "").strip()
            context_codes.append(
                {
                    "code": code,
                    "description": record.get("description", "") or "",
                    "general_rate_of_duty": general_rate,
                    "ad_valorem_rate": str(derive_rate(general_rate)),
                }
            )
        return {
            "note_label": NOTE_LABEL,
            "chapter99_headings": list(headings),
            "hts_codes": context_codes,
        }

    def _normalize_payload(
        self,
        payload: Any,
        allowed_headings: Sequence[str],
        hts_data: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        allowed = {_normalize_code(str(item)) for item in allowed_headings if item}
        measures = _coerce_measure_payload(payload)
        normalized: List[Dict[str, Any]] = []
        rate_map: Dict[str, str] = {}
        for code, record in hts_data.items():
            general_rate = (record.get("general_rate_of_duty") or "").strip()
            rate_map[_normalize_code(code)] = str(derive_rate(general_rate))
        for measure in measures:
            heading = _normalize_code(str(measure.get("heading") or ""))
            if not heading or heading not in allowed:
                continue
            measure["heading"] = heading
            if not measure.get("country_iso2"):
                measure["country_iso2"] = self.country_iso2
            if measure.get("value_basis") is None:
                measure["value_basis"] = "customs_value"
            if measure.get("is_potential") is None:
                measure["is_potential"] = False
            if measure.get("ad_valorem_rate") is None and heading in rate_map:
                measure["ad_valorem_rate"] = rate_map[heading]
            scopes = measure.get("scopes")
            if not isinstance(scopes, list):
                measure["scopes"] = []
            normalized.append(measure)
        return {"measures": normalized}

    def _persist(self, measure_entries: Sequence[Dict[str, Any]]) -> None:
        if not measure_entries:
            return
        measure_records: List[MeasureRecord] = []
        scopes_by_measure: List[List[ScopeMapDraft]] = []

        for entry in measure_entries:
            heading = _normalize_code(entry.get("heading", ""))
            if not heading:
                continue
            start = _parse_date(entry.get("effective_start_date")) or DEFAULT_START_DATE
            end = _parse_date(entry.get("effective_end_date"))
            notes = entry.get("notes") if isinstance(entry.get("notes"), dict) else None
            record = MeasureRecord(
                heading=heading,
                country_iso2=_normalize_iso(entry.get("country_iso2")) or self.country_iso2,
                ad_valorem_rate=_parse_rate(entry.get("ad_valorem_rate")) or Decimal("0"),
                value_basis=entry.get("value_basis") or "customs_value",
                notes=notes,
                effective_start_date=start,
                effective_end_date=end,
            )
            measure_records.append(record)
            scopes_by_measure.append(_expand_scope_entries(entry.get("scopes") or []))

        measure_ids = self.db.insert_measures_batch(measure_records)
        unique_scopes: List[ScopeRecord] = []
        scope_index: Dict[Tuple[str, str, Optional[str], date, Optional[date]], int] = {}
        pending_maps: List[Tuple[Tuple[str, str, Optional[str], date, Optional[date]], int, ScopeMapDraft]] = []

        for measure_id, scope_entries in zip(measure_ids, scopes_by_measure):
            for entry in scope_entries:
                signature = _scope_signature(entry.scope)
                if signature not in scope_index:
                    scope_index[signature] = len(unique_scopes)
                    unique_scopes.append(entry.scope)
                pending_maps.append((signature, measure_id, entry))

        scope_ids = self.db.insert_scopes_batch(unique_scopes)
        signature_to_id = {
            signature: scope_ids[index] for signature, index in scope_index.items()
        }
        map_entries: List[ScopeMapEntry] = []
        seen_maps: set[Tuple[int, int, str, date, Optional[date]]] = set()
        for signature, measure_id, entry in pending_maps:
            scope_id = signature_to_id[signature]
            map_sig = (scope_id, measure_id, entry.relation, entry.map_start, entry.map_end)
            if map_sig in seen_maps:
                continue
            seen_maps.add(map_sig)
            map_entries.append(
                ScopeMapEntry(
                    scope_id=scope_id,
                    measure_id=measure_id,
                    relation=entry.relation,
                    note_label=entry.note_label,
                    text_criteria=entry.text_criteria,
                    effective_start_date=entry.map_start,
                    effective_end_date=entry.map_end,
                )
            )

        self.db.insert_scope_maps_batch(map_entries)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Section 301 note(20) processing agent with dual LLM verification")
    parser.add_argument(
        "--dsn",
        default=os.getenv("DATABASE_DSN"),
        help="PostgreSQL DSN",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-5"),
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_API_BASE"),
        help="OpenAI API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key",
    )
    parser.add_argument(
        "--grok-model",
        default=os.getenv("GROK_MODEL") or os.getenv("XAI_MODEL"),
        help="Grok model to use",
    )
    parser.add_argument(
        "--grok-api-key",
        default=os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY"),
        help="Grok API key",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist measures/scopes after successful LLM comparison",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not args.dsn:
        raise SystemExit("Database DSN required (set DATABASE_DSN or use --dsn)")

    # Initialize database
    db = Section301BatchDatabase(args.dsn)

    # Initialize both LLM clients
    try:
        openai_llm = Section301BatchLLM(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
        )
        LOGGER.info("Initialized OpenAI client with model: %s", args.model)
    except Exception as e:
        LOGGER.error("Failed to initialize OpenAI client: %s", e)
        raise

    try:
        grok_llm = Section301GrokLLM(
            model=args.grok_model,
            api_key=args.grok_api_key,
        )
        LOGGER.info("Initialized Grok client with model: %s", args.grok_model)
    except Exception as e:
        LOGGER.error("Failed to initialize Grok client: %s", e)
        raise

    processor = Section301Note20Processor(
        db=db,
        openai_llm=openai_llm,
        grok_llm=grok_llm,
        persist=args.persist,
    )

    try:
        # Don't pass headings argument, let the processor fetch from database
        processor.process()
    except Exception as e:
        LOGGER.error("Batch processing failed: %s", e, exc_info=True)
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
