"""
Generate IEEPA test cases for specific headings by reading note2(v) with an LLM
and sampling HTS10 codes from the database (hts_codes).

Environment:
    DATABASE_DSN      Postgres connection string (required)
    OPENAI_API_KEY    API key for the LLM call (required)
    OPENAI_MODEL      Optional override of the model name (default: gpt-5)
    OPENAI_API_BASE   Optional API base (default: https://api.openai.com/v1)

Output:
    simulator/test/ieepa/ieepa_test_data.csv
    Header: ieepahts,inputhts,entrydate,country,steel_percentage,
            aluminum_percentage,A=预测结果,duty
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import psycopg2
import requests

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

OUTPUT_PATH = Path(__file__).with_suffix(".csv")
NOTE_PATH = Path(__file__).with_name("note2v.txt")
TARGET_HEADINGS = ["9903.01.25", "9903.02.02"]

# Local, simplified version of the note prompt to avoid importing the agent module.
# We only need input_htscode and explicit scope/except extraction.
LLM_NOTE_PROMPT = """
You are a structured extractor for HTSUS Section ieepa notes.
Output JSON only with fields: input_htscode (copy from context), scope, except.
Each item in scope or except should have key and key_type ("heading" | "hts8" | "hts10").
Expand printed ranges (e.g., 9903.02.01-9903.02.03).

Context: {context_json}
Input:
\"\"\"{note_text}\"\"\"
"""


class LLMClient:
    """Minimal OpenAI-compatible chat client."""

    def __init__(self) -> None:
        self.model = os.getenv("OPENAI_MODEL", "gpt-5")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required to call the LLM")

    def chat(self, message: str) -> str:
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
        response = requests.post(url, headers=headers, json=payload, timeout=36000)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


def load_note_text() -> str:
    return NOTE_PATH.read_text(encoding="utf-8")


def build_llm_message(note_text: str, heading: str) -> str:
    context = {
        "input_htscode": [heading],
        "note_labels": ["note(2)(v)"],
        "primary_note_labels": ["note(2)(v)"],
        "supporting_note_labels": [],
    }
    return LLM_NOTE_PROMPT.format(
        context_json=json.dumps(context, ensure_ascii=False),
        note_text=note_text.strip(),
    )


def parse_llm_payload(raw: str) -> Dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse LLM JSON: {raw}") from exc


def extract_note_exclusion_codes(note_text: str, limit: int = 10) -> List[str]:
    """Pull HTS references from the 9903.01.32 exclusion list in note2(v)."""
    import re

    # Grab all HTS-like patterns (e.g., 0201.10.05.10 or 0201.10.05)
    pattern = re.compile(r"\b\d{4}\.\d{2}\.\d{2}(?:\.\d{2})?\b")
    codes = []
    for match in pattern.finditer(note_text):
        code = match.group(0)
        if code.startswith("99") or code.startswith("98"):  # ignore chapter 99/98 references
            continue
        codes.append(code)
    # Deduplicate while preserving order, then take a slice to keep test set small.
    seen = set()
    unique = []
    for code in codes:
        if code in seen:
            continue
        seen.add(code)
        unique.append(code)
    return unique[:limit]


def connect_db():
    dsn = os.getenv("DATABASE_DSN")
    if not dsn:
        raise RuntimeError("DATABASE_DSN is required")
    return psycopg2.connect(dsn)


def fetch_rate(conn, heading: str) -> Optional[float]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ad_valorem_rate FROM sieepa_measures WHERE heading=%s LIMIT 1",
            (heading,),
        )
        row = cur.fetchone()
    return float(row[0]) if row else None


def sample_hts10(conn, prefix: str) -> Optional[str]:
    """Pick the first HTS10 (>=4 segments) under the given prefix from hts_codes."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT hts_number
            FROM hts_codes
            WHERE hts_number LIKE %s || '%%'
              AND length(hts_number) >= 12
              AND hts_number NOT LIKE '99%%'
              AND hts_number NOT LIKE '98%%'
            ORDER BY hts_number
            LIMIT 1
            """,
            (prefix,),
        )
        row = cur.fetchone()
    return row[0] if row else None


def first_non_excluded(conn, excluded_prefixes: Sequence[str]) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT hts_number
            FROM hts_codes
            WHERE length(hts_number) >= 12
              AND hts_number NOT LIKE '99%%'
              AND hts_number NOT LIKE '98%%'
            ORDER BY hts_number
            """,
        )
        for (code,) in cur.fetchall():
            if not any(code.startswith(p) for p in excluded_prefixes):
                return code
    return None


def collect_excluded_prefixes(llm_payload: Dict) -> List[str]:
    excluded: List[str] = []
    for item in llm_payload.get("except", []):
        key = item.get("key") or item.get("keys")
        if not key:
            continue
        if isinstance(key, str):
            candidates = [p.strip() for p in key.split(",")]
        else:
            continue
        excluded.extend([c for c in candidates if c])
    return excluded


def generate_rows(conn, heading: str, llm_payload: Dict, exclusion_codes: List[str]) -> List[List]:
    rows: List[List] = []
    excluded_prefixes = collect_excluded_prefixes(llm_payload)

    # Positive case: generic good not excluded → heading rate
    generic_hts = first_non_excluded(conn, excluded_prefixes)
    rate = fetch_rate(conn, heading)
    if rate is None:
        # skip if heading not found
        return rows
    if generic_hts:
        rows.append(
            [
                heading,
                generic_hts,
                date.today().strftime("%Y/%m/%d"),
                "CN",
                0,
                0,
                f"【正向】note2(v)(i) 普通商品无豁免 ⇒ {heading} 加征{rate}%",
                rate,
            ]
        )

    # Negative samples from note-listed exclusions (9903.01.32 zero-rate mapping)
    for code in exclusion_codes:
        prefix = code.strip()
        if not prefix:
            continue
        hts10 = sample_hts10(conn, prefix) or prefix
        rows.append(
            [
                "9903.01.32",
                hts10,
                date.today().strftime("%Y/%m/%d"),
                "CN",
                0,
                0,
                f"【负向】note2(v)(iii) 排除 {prefix} ⇒ 9903.01.32 零税",
                0.0,
            ]
        )

    return rows


def write_csv(rows: Iterable[Sequence]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ieepahts",
                "inputhts",
                "entrydate",
                "country",
                "steel_percentage",
                "aluminum_percentage",
                "A=预测结果",
                "duty",
            ]
        )
        writer.writerows(rows)
    print(f"Wrote {OUTPUT_PATH}")


def main() -> None:
    note_text = load_note_text()
    exclusion_codes = extract_note_exclusion_codes(note_text, limit=6)
    llm = LLMClient()
    all_rows: List[List] = []

    with connect_db() as conn:
        for heading in TARGET_HEADINGS:
            message = build_llm_message(note_text, heading)
            payload = parse_llm_payload(llm.chat(message))
            all_rows.extend(generate_rows(conn, heading, payload, exclusion_codes))

    write_csv(all_rows)


if __name__ == "__main__":
    main()
