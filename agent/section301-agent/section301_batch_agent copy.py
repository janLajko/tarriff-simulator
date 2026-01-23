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
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


LOGGER = logging.getLogger("section301_batch_agent")

DEFAULT_START_DATE = date(1900, 1, 1)

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


class Section301BatchLLM:
    """OpenAI LLM client for batch processing."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 36000,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "GPT-5.2")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.timeout = timeout

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY required")

    def extract_batch(self, headings_data: Dict[str, Dict[str, Any]], note20_content: str) -> List[Dict[str, Any]]:
        """Extract measures from all headings in one LLM call."""
        headings_json = json.dumps(headings_data, ensure_ascii=False, indent=2)
        message = LLM_BATCH_PROMPT.format(
            headings_json=headings_json,
            note20_content=note20_content
        )
        system_prompt = "You are a precise legal text parser. Respond with JSON only."
        LOGGER.info("OpenAI prompt (system): %s", system_prompt)
        LOGGER.info("OpenAI prompt (user): %s", message)

        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
        }

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        result = json.loads(content)
        # Handle both array and object with "measures" key
        if isinstance(result, dict) and "measures" in result:
            return result["measures"]
        elif isinstance(result, list):
            return result
        else:
            return [result]


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
                   DO NOTHING
                   RETURNING id""",
                values
            )
            return [row[0] for row in cur.fetchall()]

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

    async def _concurrent_llm_calls(self, hts_data: Dict[str, Dict[str, Any]], note20_content: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Execute LLM calls concurrently to OpenAI and Gemini."""
        LOGGER.info("Starting concurrent LLM calls to OpenAI and Gemini...")

        openai_task = self._call_openai_async(hts_data, note20_content)
        gemini_task = self._call_gemini_async(hts_data, note20_content)

        openai_results, gemini_results = await asyncio.gather(openai_task, gemini_task)

        LOGGER.info("OpenAI returned %d results", len(openai_results))
        LOGGER.info("Gemini returned %d results", len(gemini_results))

        # Save results to files for debugging
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save OpenAI results
        openai_file = f"openai_results_{timestamp}.json"
        with open(openai_file, "w", encoding="utf-8") as f:
            json.dump(openai_results, f, indent=2, ensure_ascii=False, default=str)
        LOGGER.info(f"OpenAI results saved to: {openai_file}")

        # Save Gemini results
        gemini_file = f"gemini_results_{timestamp}.json"
        with open(gemini_file, "w", encoding="utf-8") as f:
            json.dump(gemini_results, f, indent=2, ensure_ascii=False, default=str)
        LOGGER.info(f"Gemini results saved to: {gemini_file}")

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

        # Limit to first 2 headings for testing (to avoid token limit issues) 9903.88.01
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Section 301 batch processing agent with dual LLM verification")
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
        "--no-require-match",
        action="store_true",
        help="Don't require OpenAI and Gemini results to match before inserting (default: require match)",
    )
    parser.add_argument(
        "--openai-model",
        default=os.getenv("OPENAI_MODEL", "gpt-5.2"),
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--gemini-model",
        default=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
        help="Gemini model to use",
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
        openai_llm = Section301BatchLLM(model=args.openai_model)
        LOGGER.info("Initialized OpenAI client with model: %s", args.openai_model)
    except Exception as e:
        LOGGER.error("Failed to initialize OpenAI client: %s", e)
        raise

    try:
        gemini_llm = Section301GeminiLLM(model=args.gemini_model)
        LOGGER.info("Initialized Gemini client with model: %s", args.gemini_model)
    except Exception as e:
        LOGGER.error("Failed to initialize Gemini client: %s", e)
        raise

    # Initialize agent with dual LLM support
    agent = Section301BatchAgent(
        db=db,
        openai_llm=openai_llm,
        gemini_llm=gemini_llm,
        require_match=not args.no_require_match
    )

    try:
        # Don't pass headings argument, let the agent fetch from database
        agent.run()
    except Exception as e:
        LOGGER.error("Batch processing failed: %s", e, exc_info=True)
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
