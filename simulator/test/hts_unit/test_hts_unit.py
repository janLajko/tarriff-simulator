import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from simulator import section232_rate
except ModuleNotFoundError as exc:  # pragma: no cover - protective guard
    if exc.name == "psycopg2":
        pytest.skip(
            "psycopg2 must be installed to run HTS unit integration tests.",
            allow_module_level=True,
        )
    raise


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

DIGIT_RE = re.compile(r"[^0-9]")
JSON_CONFIG_PATH = Path(__file__).parent / "hts_basic_rate.parsed_with232_v6.json"


@pytest.fixture(scope="module")
def db_connection():
    dsn = os.environ["DATABASE_DSN"]
    conn = section232_rate.psycopg2.connect(dsn)
    try:
        yield conn
    finally:
        conn.close()


def _normalize_digits(value: str) -> str:
    return DIGIT_RE.sub("", value or "")


def _build_hts_prefix_index(conn) -> Dict[int, Dict[str, Set[str]]]:
    index: Dict[int, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    query = """
        SELECT hts_number,
               regexp_replace(hts_number, '[^0-9]', '', 'g') AS digits
        FROM hts_codes
        WHERE hts_number IS NOT NULL
          AND regexp_replace(hts_number, '[^0-9]', '', 'g') ~ '^[0-9]{10}$'
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    for hts_number, digits in rows:
        normalized_digits = (digits or "").strip()
        normalized_hts = (hts_number or "").strip()
        if len(normalized_digits) != 10 or not normalized_hts:
            continue
        for length in range(2, 11):
            prefix = normalized_digits[:length]
            index[length][prefix].add(normalized_hts)
    return index


def _resolve_prefix_length(key_type: str, digits: str) -> int:
    normalized = (key_type or "").strip().lower()
    if normalized.startswith("hts"):
        suffix = DIGIT_RE.sub("", normalized)
        if suffix:
            return max(2, min(int(suffix), 10))
    if normalized == "heading":
        return max(2, min(len(digits), 10))
    return max(2, min(len(digits), 10))


def _fetch_scope_rows(conn, prefixes: Sequence[str]) -> List[Tuple[str, str, str]]:
    if not prefixes:
        return []
    conditions = " OR ".join("m.heading LIKE %s" for _ in prefixes)
    sql = f"""
        SELECT m.heading, scope.key, scope.key_type
        FROM s232_measures AS m
        JOIN s232_scope_measure_map AS map ON map.measure_id = m.id
        JOIN s232_scope AS scope ON scope.id = map.scope_id
        WHERE map.relation = 'include' AND ({conditions})
    """
    params = tuple(f"{prefix}%" for prefix in prefixes)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    seen: Set[Tuple[str, str, str]] = set()
    results: List[Tuple[str, str, str]] = []
    for measure_heading, key, key_type in rows:
        normalized_key = (key or "").strip()
        normalized_type = (key_type or "").strip()
        normalized_measure = (measure_heading or "").strip()
        if not normalized_key:
            continue
        entry = (normalized_measure, normalized_key, normalized_type)
        if entry not in seen:
            seen.add(entry)
            results.append(entry)
    return results


def _expand_scope_key(
    key: str,
    key_type: str,
    prefix_index: Dict[int, Dict[str, Set[str]]],
) -> Set[str]:
    digits = _normalize_digits(key)
    if not digits:
        return set()
    prefix_length = _resolve_prefix_length(key_type, digits)
    prefix_digits = digits[:prefix_length]
    bucket = prefix_index.get(prefix_length, {})
    matched = bucket.get(prefix_digits)
    if matched:
        return set(matched)
    return set()


def _collect_component_hts(
    conn,
    prefix_index: Dict[int, Dict[str, Set[str]]],
    prefixes: Sequence[str],
    requirement_fn: Callable[[str], Optional[str]],
) -> Set[str]:
    rows = _fetch_scope_rows(conn, prefixes)
    matches: Set[str] = set()
    for measure_heading, key, key_type in rows:
        required_chapter = requirement_fn(measure_heading)
        if required_chapter is None:
            continue
        row_matches = _expand_scope_key(key, key_type, prefix_index)
        if not row_matches:
            continue
        if required_chapter:
            filtered = {
                code
                for code in row_matches
                if _normalize_digits(code).startswith(required_chapter)
            }
        else:
            filtered = row_matches
        matches.update(filtered)
    return matches


def _steel_requirement(heading: str) -> Optional[str]:
    normalized = (heading or "").strip()
    if normalized in section232_rate.STEEL_CH73_HEADINGS:
        return "73"
    if normalized in section232_rate.STEEL_ALWAYS_HEADINGS:
        return ""
    return None


def _aluminum_requirement(heading: str) -> Optional[str]:
    normalized = (heading or "").strip()
    if normalized in section232_rate.ALUMINUM_CH76_HEADINGS:
        return "76"
    if normalized in section232_rate.ALUMINUM_ALWAYS_HEADINGS:
        return ""
    return None


def _describe(label: str, codes: Set[str]) -> None:
    sorted_codes = sorted(codes)
    preview = ", ".join(sorted_codes[:10])
    print(f"{label}: {len(sorted_codes)} HTS10 codes")
    if preview:
        print(f"  Sample: {preview}")


def _write_codes(path: Path, codes: Set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_codes = sorted(codes)
    with path.open("w", encoding="utf-8") as handle:
        for code in sorted_codes:
            handle.write(f"{code}\n")


def test_measurement_requirement_statistics(db_connection):
    steel_only, aluminum_only, both = _compute_measurement_sets(db_connection)

    _describe("Requires steel_percentage", steel_only)
    _describe("Requires aluminum_percentage", aluminum_only)
    _describe("Requires both steel and aluminum percentages", both)

    output_dir = Path(__file__).parent
    _write_codes(output_dir / "steel_hts.txt", steel_only)
    _write_codes(output_dir / "aluminum_hts.txt", aluminum_only)
    _write_codes(output_dir / "steel_and_aluminum_hts.txt", both)

    assert steel_only, "Expected at least one HTS requiring steel percentage input"
    assert aluminum_only, "Expected at least one HTS requiring aluminum percentage input"


def _compute_measurement_sets(
    db_connection,
) -> Tuple[Set[str], Set[str], Set[str]]:
    prefix_index = _build_hts_prefix_index(db_connection)
    steel_hts = _collect_component_hts(
        db_connection, prefix_index, section232_rate.STEEL_PREFIXES, _steel_requirement
    )
    aluminum_hts = _collect_component_hts(
        db_connection, prefix_index, section232_rate.ALUMINUM_PREFIXES, _aluminum_requirement
    )
    both = steel_hts & aluminum_hts
    steel_only = steel_hts - both
    aluminum_only = aluminum_hts - both
    return steel_only, aluminum_only, both


def _load_section232_sets() -> Tuple[Set[str], Set[str], Set[str]]:
    if not JSON_CONFIG_PATH.exists():
        raise FileNotFoundError(f"HTS config JSON not found: {JSON_CONFIG_PATH}")
    payload = json.loads(JSON_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("HTS config JSON must be an object mapping HTS → metadata")

    steel_only: Set[str] = set()
    aluminum_only: Set[str] = set()
    both: Set[str] = set()

    for hts_code, entry in payload.items():
        section_data = entry.get("section232") if isinstance(entry, dict) else None
        if not isinstance(section_data, dict):
            continue
        steel_required = bool(
            section_data.get("steel_percentage")
            or section_data.get("steel_pour_country")
        )
        aluminum_required = bool(
            section_data.get("aluminum_percentage")
            or section_data.get("aluminum_pour_country")
        )
        normalized_code = str(hts_code or "").strip()
        if not normalized_code:
            continue
        if steel_required and aluminum_required:
            both.add(normalized_code)
        elif steel_required:
            steel_only.add(normalized_code)
        elif aluminum_required:
            aluminum_only.add(normalized_code)
    return steel_only, aluminum_only, both


def _print_diff(label: str, missing: Set[str], extra: Set[str]) -> None:
    if missing:
        print(f"{label} missing in JSON ({len(missing)}): {', '.join(sorted(list(missing)[:10]))}")
    if extra:
        print(f"{label} extra in JSON ({len(extra)}): {', '.join(sorted(list(extra)[:10]))}")


def test_section232_config_matches_dataset(db_connection):
    steel_only_db, aluminum_only_db, both_db = _compute_measurement_sets(db_connection)
    steel_only_json, aluminum_only_json, both_json = _load_section232_sets()

    steel_missing = steel_only_db - steel_only_json
    steel_extra = steel_only_json - steel_only_db
    aluminum_missing = aluminum_only_db - aluminum_only_json
    aluminum_extra = aluminum_only_json - aluminum_only_db
    both_missing = both_db - both_json
    both_extra = both_json - both_db

    _print_diff("Steel-only HTS", steel_missing, steel_extra)
    _print_diff("Aluminum-only HTS", aluminum_missing, aluminum_extra)
    _print_diff("Steel+Aluminum HTS", both_missing, both_extra)

    assert not steel_missing and not steel_extra, "Steel-only HTS mismatch with JSON config"
    assert not aluminum_missing and not aluminum_extra, "Aluminum-only HTS mismatch with JSON config"
    assert not both_missing and not both_extra, "Steel+Aluminum HTS mismatch with JSON config"
