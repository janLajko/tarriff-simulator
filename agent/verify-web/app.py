#!/usr/bin/env python3
"""简化版Web审核平台后端"""

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import json
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover - runtime dependency
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor, execute_values
except ImportError:
    psycopg2 = None  # type: ignore
    Json = None  # type: ignore
    RealDictCursor = None  # type: ignore
    execute_values = None  # type: ignore

app = Flask(__name__)
CORS(app)

# 数据目录
OTHERCH_OUTPUT_DIR = Path(__file__).parent.parent / "othercharpter-agent" / "output"
SECTION301_OUTPUT_DIR = Path(__file__).parent.parent / "section301-agent" / "output"
SECTION232_OUTPUT_DIR = Path(__file__).parent.parent / "section232-agent" / "output"
SECTIONIEEPA_OUTPUT_DIR = Path(__file__).parent.parent / "sectionieepa" / "output"
SESSIONS = {}  # 简单的内存存储
HISTORY_DIR = Path(__file__).parent / "history"

DB_DSN = os.getenv("DATABASE_URL")
TABLE_PREFIX = os.getenv("OTHERCH_TABLE_PREFIX", "otherch")
DEFAULT_START_DATE = date(1900, 1, 1)

NOTE_IDS = [33, 36, 37, 38, 20, 16, 2]


def _get_module_config(note_id: int) -> Dict[str, Any]:
    if note_id in {33, 36, 37, 38}:
        return {
            "module": "otherchapter",
            "output_dir": OTHERCH_OUTPUT_DIR,
            "file_prefix": f"note{note_id}",
            "measures_table": f"{TABLE_PREFIX}_measures",
            "scope_table": f"{TABLE_PREFIX}_scope",
            "map_table": f"{TABLE_PREFIX}_scope_measure_map",
            "query_by": "note_number",
        }
    if note_id == 20:
        return {
            "module": "section301",
            "output_dir": SECTION301_OUTPUT_DIR,
            "file_prefix": "note20",
            "measures_table": "s301_measures",
            "scope_table": "s301_scope",
            "map_table": "s301_scope_measure_map",
            "query_by": "headings",
        }
    if note_id == 16:
        return {
            "module": "section232",
            "output_dir": SECTION232_OUTPUT_DIR,
            "file_prefix": "note16",
            "measures_table": "s232_measures",
            "scope_table": "s232_scope",
            "map_table": "s232_scope_measure_map",
            "query_by": "headings",
        }
    if note_id == 2:
        return {
            "module": "sectionieepa",
            "output_dir": SECTIONIEEPA_OUTPUT_DIR,
            "file_prefix": "note2",
            "measures_table": "sieepa_measures",
            "scope_table": "sieepa_scope",
            "map_table": "sieepa_scope_measure_map",
            "query_by": "headings",
        }
    raise ValueError(f"Unsupported note id: {note_id}")

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/notes', methods=['GET'])
def get_notes():
    """获取所有可审核的Note列表"""
    notes = []
    for note_num in NOTE_IDS:
        config = _get_module_config(note_num)
        compare_file = config["output_dir"] / f"{config['file_prefix']}_llm_compare.json"
        status = "not_processed"
        has_conflicts = False

        if compare_file.exists():
            with open(compare_file) as f:
                data = json.load(f)
                has_conflicts = not data.get('consistent', True)
                status = "has_conflicts" if has_conflicts else "consistent"

        notes.append({
            'note_number': note_num,
            'module': config["module"],
            'status': status,
            'has_conflicts': has_conflicts
        })

    return jsonify(notes)

@app.route('/api/notes/<int:note_id>/start-review', methods=['POST'])
def start_review(note_id):
    """启动审核会话"""
    config = _get_module_config(note_id)
    # 加载文件
    openai_file = config["output_dir"] / f"{config['file_prefix']}_openai.json"
    grok_file = config["output_dir"] / f"{config['file_prefix']}_grok.json"
    compare_file = config["output_dir"] / f"{config['file_prefix']}_llm_compare.json"

    if not all([openai_file.exists(), grok_file.exists(), compare_file.exists()]):
        return jsonify({'error': 'Required files not found'}), 404

    with open(openai_file) as f:
        openai_data = json.load(f)
    with open(grok_file) as f:
        grok_data = json.load(f)
    with open(compare_file) as f:
        compare_data = json.load(f)

    # 创建会话
    session_id = f"sess_{note_id}_{int(datetime.now().timestamp())}"
    SESSIONS[session_id] = {
        'session_id': session_id,
        'note_id': note_id,
        'module': config["module"],
        'output_dir': str(config["output_dir"]),
        'file_prefix': config["file_prefix"],
        'measures_table': config["measures_table"],
        'scope_table': config["scope_table"],
        'map_table': config["map_table"],
        'query_by': config["query_by"],
        'status': 'conflict_review',
        'openai_data': openai_data,
        'grok_data': grok_data,
        'compare_data': compare_data,
        'resolutions': {},
        'start_time': datetime.now().isoformat()
    }

    # 解析冲突，传入session以获取原始数据
    conflicts = parse_conflicts(compare_data, openai_data, grok_data)

    return jsonify({
        'session_id': session_id,
        'conflicts': conflicts,
        'total_conflicts': len(conflicts['measure_conflicts']) + len(conflicts['scope_conflicts'])
    })

@app.route('/api/sessions/<session_id>/conflicts', methods=['GET'])
def get_conflicts(session_id):
    """获取冲突列表"""
    session = SESSIONS.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    conflicts = parse_conflicts(session['compare_data'], session['openai_data'], session['grok_data'])
    return jsonify(conflicts)

@app.route('/api/sessions/<session_id>/resolve', methods=['POST'])
def resolve_conflict(session_id):
    """解决冲突"""
    session = SESSIONS.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    data = request.json
    conflict_id = data['conflict_id']
    resolution = data['resolution']  # 'openai', 'grok', or 'manual'
    value = data.get('value')

    # 记录解决方案
    if 'resolutions' not in session:
        session['resolutions'] = {}

    session['resolutions'][conflict_id] = {
        'resolution': resolution,
        'value': value,
        'resolved_at': datetime.now().isoformat()
    }

    return jsonify({'status': 'resolved', 'conflict_id': conflict_id})

@app.route('/api/sessions/<session_id>/generate-unified', methods=['POST'])
def generate_unified(session_id):
    """生成统一JSON"""
    session = SESSIONS.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    # 构建统一数据
    unified_data = build_unified_json(session)

    # 保存文件
    output_dir = Path(session.get("output_dir") or OTHERCH_OUTPUT_DIR)
    output_file = output_dir / f"{_session_file_prefix(session)}_unified.json"
    with open(output_file, 'w') as f:
        json.dump(unified_data, f, indent=2)

    session['unified_json'] = unified_data
    session['status'] = 'unified_generated'

    return jsonify({
        'status': 'success',
        'output_file': str(output_file),
        'measure_count': len(unified_data.get('measures', []))
    })


@app.route('/api/sessions/<session_id>/next-step', methods=['POST'])
def next_step(session_id):
    """生成统一JSON并执行数据库比对"""
    session = SESSIONS.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    compare_data = session.get('compare_data', {})
    conflicts = parse_conflicts(compare_data, session['openai_data'], session['grok_data'])
    total_conflicts = len(conflicts['measure_conflicts']) + len(conflicts['scope_conflicts'])
    if len(session.get('resolutions', {})) < total_conflicts:
        return jsonify({'error': 'Not all conflicts resolved'}), 400

    unified_data = build_unified_json(session)
    output_dir = Path(session.get("output_dir") or OTHERCH_OUTPUT_DIR)
    output_file = output_dir / f"{_session_file_prefix(session)}_unified.json"
    with open(output_file, 'w') as f:
        json.dump(unified_data, f, indent=2)

    session['unified_json'] = unified_data
    session['status'] = 'unified_generated'

    db_compare_file = output_dir / f"{_session_file_prefix(session)}_db_compare.json"
    use_file_compare = compare_data.get('consistent') and db_compare_file.exists()

    if use_file_compare:
        with open(db_compare_file) as f:
            db_compare = json.load(f)
        headings = [m.get("heading") for m in unified_data.get("measures", []) if m.get("heading")]
        db_data = fetch_db_measures(session['note_id'], session, headings)
    else:
        headings = [m.get("heading") for m in unified_data.get("measures", []) if m.get("heading")]
        db_data = fetch_db_measures(session['note_id'], session, headings)
        db_compare = compare_unified_with_db(unified_data.get('measures', []), db_data)

    session['db_compare'] = db_compare
    session['db_data'] = db_data
    session['status'] = 'db_comparison'

    if not db_compare.get('consistent', True):
        db_conflicts = parse_db_conflicts(db_compare, unified_data, db_data)
        session['db_conflicts'] = db_conflicts
        session['status'] = 'db_conflict_review'
        return jsonify({
            'status': 'db_conflicts',
            'conflicts': db_conflicts,
            'total_conflicts': len(db_conflicts['measure_conflicts']) + len(db_conflicts['scope_conflicts']),
            'output_file': str(output_file)
        })

    session['status'] = 'ready_to_commit'
    return jsonify({
        'status': 'ready_to_commit',
        'output_file': str(output_file),
        'measure_count': len(unified_data.get('measures', []))
    })


@app.route('/api/sessions/<session_id>/resolve-db', methods=['POST'])
def resolve_db_conflict(session_id):
    """解决数据库差异"""
    session = SESSIONS.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    data = request.json
    conflict_id = data['conflict_id']
    resolution = data['resolution']
    value = data.get('value')

    if 'db_resolutions' not in session:
        session['db_resolutions'] = {}

    session['db_resolutions'][conflict_id] = {
        'resolution': resolution,
        'value': value,
        'resolved_at': datetime.now().isoformat()
    }

    return jsonify({'status': 'resolved', 'conflict_id': conflict_id})


@app.route('/api/sessions/<session_id>/update-db', methods=['POST'])
def update_db(session_id):
    """更新数据库"""
    session = SESSIONS.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    unified_data = session.get('unified_json')
    db_compare = session.get('db_compare')
    db_data = session.get('db_data', [])
    if not unified_data or db_compare is None:
        return jsonify({'error': 'Unified data or DB compare missing'}), 400

    db_conflicts = session.get('db_conflicts') or parse_db_conflicts(db_compare, unified_data, db_data)
    total_conflicts = len(db_conflicts['measure_conflicts']) + len(db_conflicts['scope_conflicts'])
    if len(session.get('db_resolutions', {})) < total_conflicts:
        return jsonify({'error': 'Not all DB conflicts resolved'}), 400

    final_data = apply_db_resolutions(unified_data, db_data, session.get('db_resolutions', {}))
    result = persist_measures_to_db(session['note_id'], final_data.get('measures', []), session)

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history_subdir = HISTORY_DIR / f"{session['note_id']}_{session_id}"
    history_subdir.mkdir(parents=True, exist_ok=True)
    for suffix in ("openai", "grok", "llm_compare", "unified"):
        output_dir = Path(session.get("output_dir") or OTHERCH_OUTPUT_DIR)
        source = output_dir / f"{_session_file_prefix(session)}_{suffix}.json"
        if source.exists():
            shutil.move(str(source), history_subdir / source.name)

    session['status'] = 'committed'
    return jsonify({
        'status': 'success',
        'inserted_measures': result.get('inserted_measures', 0),
        'inserted_scopes': result.get('inserted_scopes', 0)
    })

@app.route('/api/sessions/<session_id>/status', methods=['GET'])
def get_session_status(session_id):
    """获取会话状态"""
    session = SESSIONS.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    return jsonify({
        'session_id': session_id,
        'status': session['status'],
        'note_id': session['note_id'],
        'resolved_count': len(session.get('resolutions', {}))
    })

def _session_file_prefix(session: Dict[str, Any]) -> str:
    prefix = session.get("file_prefix")
    if prefix:
        return prefix
    return f"note{session['note_id']}"

def _normalize_code(value: str) -> str:
    return (value or "").strip().strip(",;")


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


def _normalize_effective_date(value: Optional[Any]) -> Optional[date]:
    parsed = _parse_date(value)
    if parsed == DEFAULT_START_DATE:
        return None
    return parsed


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


def _normalize_relation(value: Optional[str]) -> str:
    cleaned = (value or "include").strip().lower()
    return cleaned or "include"


def _normalize_scope_key_digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


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
            effective_start_date.isoformat() if effective_start_date else "null",
            effective_end_date.isoformat() if effective_end_date else "null",
            "true" if is_potential else "false" if is_potential is not None else "null",
        ]
    )


def _expand_scope_entries(scopes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for scope in scopes:
        keys = scope.get("keys")
        key = scope.get("key")
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
            entry = dict(scope)
            entry.pop("keys", None)
            entry["key"] = key_raw
            expanded.append(entry)
    return expanded


def _scope_union(openai_scopes: List[Dict[str, Any]], grok_scopes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for scope in _expand_scope_entries(openai_scopes):
        relation = _normalize_relation(scope.get("relation"))
        key_norm = _normalize_scope_key_digits(scope.get("key", ""))
        if not key_norm:
            continue
        merged[(key_norm, relation)] = scope
    for scope in _expand_scope_entries(grok_scopes):
        relation = _normalize_relation(scope.get("relation"))
        key_norm = _normalize_scope_key_digits(scope.get("key", ""))
        if not key_norm:
            continue
        if (key_norm, relation) not in merged:
            merged[(key_norm, relation)] = scope
    return list(merged.values())

def _build_manual_scope_selection(value: Optional[Any]) -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        selected = value.get("selected", [])
    elif isinstance(value, list):
        selected = value
    else:
        selected = []

    manual_entries: List[Dict[str, Any]] = []
    for item in selected:
        if not isinstance(item, dict):
            continue
        key_raw = _normalize_code(item.get("key") or item.get("key_raw") or "")
        relation = _normalize_relation(item.get("relation"))
        side = (item.get("side") or "").strip().lower() or None
        if not key_raw:
            continue
        manual_entries.append({"key": key_raw, "relation": relation, "side": side})
    return manual_entries


def _scope_manual_merge(
    openai_scopes: List[Dict[str, Any]],
    grok_scopes: List[Dict[str, Any]],
    manual_entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    openai_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    grok_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for scope in _expand_scope_entries(openai_scopes):
        relation = _normalize_relation(scope.get("relation"))
        key_norm = _normalize_scope_key_digits(scope.get("key", ""))
        if not key_norm:
            continue
        signature = (key_norm, relation)
        if signature not in openai_map:
            openai_map[signature] = scope

    for scope in _expand_scope_entries(grok_scopes):
        relation = _normalize_relation(scope.get("relation"))
        key_norm = _normalize_scope_key_digits(scope.get("key", ""))
        if not key_norm:
            continue
        signature = (key_norm, relation)
        if signature not in grok_map:
            grok_map[signature] = scope

    for signature, scope in openai_map.items():
        if signature in grok_map:
            merged[signature] = scope

    for entry in manual_entries:
        relation = _normalize_relation(entry.get("relation"))
        key_norm = _normalize_scope_key_digits(entry.get("key", ""))
        if not key_norm:
            continue
        signature = (key_norm, relation)
        if signature not in merged:
            side = entry.get("side")
            if side == "grok" and signature in grok_map:
                merged[signature] = grok_map[signature]
            elif side == "openai" and signature in openai_map:
                merged[signature] = openai_map[signature]
            else:
                merged[signature] = {"key": entry.get("key"), "relation": relation}

    return list(merged.values())

def _scope_manual_merge_db(
    unified_scopes: List[Dict[str, Any]],
    db_scopes: List[Dict[str, Any]],
    manual_entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    unified_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    db_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for scope in _expand_scope_entries(unified_scopes):
        relation = _normalize_relation(scope.get("relation"))
        key_norm = _normalize_scope_key_digits(scope.get("key", ""))
        if not key_norm:
            continue
        signature = (key_norm, relation)
        if signature not in unified_map:
            unified_map[signature] = scope

    for scope in _expand_scope_entries(db_scopes):
        relation = _normalize_relation(scope.get("relation"))
        key_norm = _normalize_scope_key_digits(scope.get("key", ""))
        if not key_norm:
            continue
        signature = (key_norm, relation)
        if signature not in db_map:
            db_map[signature] = scope

    for signature, scope in unified_map.items():
        if signature in db_map:
            merged[signature] = scope

    for entry in manual_entries:
        relation = _normalize_relation(entry.get("relation"))
        key_norm = _normalize_scope_key_digits(entry.get("key", ""))
        if not key_norm:
            continue
        signature = (key_norm, relation)
        if signature not in merged:
            side = entry.get("side")
            if side == "db" and signature in db_map:
                merged[signature] = db_map[signature]
            elif side == "unified" and signature in unified_map:
                merged[signature] = unified_map[signature]
            else:
                merged[signature] = {"key": entry.get("key"), "relation": relation}

    return list(merged.values())


def _collect_scope_counters(
    scopes: List[Dict[str, Any]]
) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, str], List[str]]]:
    counts: Dict[Tuple[str, str], int] = {}
    raw_map: Dict[Tuple[str, str], List[str]] = {}
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
            counts[pair] = counts.get(pair, 0) + 1
            raw_map.setdefault(pair, []).append(key_raw)
    return counts, raw_map


def _normalize_measure(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    heading = _normalize_code(str(entry.get("heading") or ""))
    if not heading:
        return None
    country_iso2 = _normalize_iso(entry.get("country_iso2"))
    effective_start_date = _normalize_effective_date(entry.get("effective_start_date"))
    effective_end_date = _normalize_effective_date(entry.get("effective_end_date"))
    is_potential = _coerce_bool(entry.get("is_potential"))
    ad_valorem_rate = _parse_rate(entry.get("ad_valorem_rate"))
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


def _index_measures_by_key(measures: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for measure in measures:
        normalized = _normalize_measure(measure)
        if not normalized:
            continue
        key = normalized["measure_key"]
        if key not in index:
            index[key] = normalized
    return index


def _build_measure_key_index(measures: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for measure in measures:
        heading = _normalize_code(str(measure.get("heading") or ""))
        if not heading:
            continue
        key = _make_measure_key(
            heading,
            _normalize_iso(measure.get("country_iso2")),
            _normalize_effective_date(measure.get("effective_start_date")),
            _normalize_effective_date(measure.get("effective_end_date")),
            _coerce_bool(measure.get("is_potential")),
        )
        if key not in index:
            index[key] = measure
    return index


def _diff_scope_counters(
    left_counter: Dict[Tuple[str, str], int],
    right_counter: Dict[Tuple[str, str], int],
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
            raw_list = left_raw_map.get(pair, [])
            for idx in range(left_count - right_count):
                key_raw = raw_list[idx] if idx < len(raw_list) else pair[0]
                only_in_left.append({"key_raw": key_raw, "key_norm": pair[0], "relation": pair[1]})
        elif right_count > left_count:
            raw_list = right_raw_map.get(pair, [])
            for idx in range(right_count - left_count):
                key_raw = raw_list[idx] if idx < len(raw_list) else pair[0]
                only_in_right.append({"key_raw": key_raw, "key_norm": pair[0], "relation": pair[1]})
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
            if left_val != right_val:
                diffs[field] = {
                    left_label: left_val,
                    right_label: right_val,
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


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


def _serialize_measure(measure: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _serialize_value(val) for key, val in measure.items()}

def parse_conflicts(compare_data, openai_data, grok_data):
    """解析冲突为前端友好格式"""
    conflicts = {
        'measure_conflicts': [],
        'scope_conflicts': []
    }

    openai_index = {k: v for k, v in _build_measure_key_index(openai_data.get('measures', [])).items()}
    grok_index = {k: v for k, v in _build_measure_key_index(grok_data.get('measures', [])).items()}

    # 字段差异（原始measure_key一致）
    for measure_key in compare_data.get('field_diffs', {}).keys():
        openai_m = _serialize_measure(openai_index.get(measure_key, {}))
        grok_m = _serialize_measure(grok_index.get(measure_key, {}))
        heading = measure_key.split('|')[0] if '|' in measure_key else measure_key
        conflicts['measure_conflicts'].append({
            'id': f"field_diff|{measure_key}",
            'type': 'field_diff',
            'measure_key': measure_key,
            'heading': heading,
            'openai_data': openai_m,
            'grok_data': grok_m,
            'differences': compare_measure_fields(openai_m, grok_m)
        })

    # 稳定键归并（stable_key = heading）
    openai_only = list(compare_data.get('missing_in_grok', []))
    grok_only = list(compare_data.get('missing_in_openai', []))

    openai_by_heading: Dict[str, List[str]] = {}
    grok_by_heading: Dict[str, List[str]] = {}
    for measure_key in openai_only:
        heading = measure_key.split('|')[0] if '|' in measure_key else measure_key
        openai_by_heading.setdefault(heading, []).append(measure_key)
    for measure_key in grok_only:
        heading = measure_key.split('|')[0] if '|' in measure_key else measure_key
        grok_by_heading.setdefault(heading, []).append(measure_key)

    for heading in sorted(set(openai_by_heading) & set(grok_by_heading)):
        openai_list = sorted(openai_by_heading.get(heading, []))
        grok_list = sorted(grok_by_heading.get(heading, []))
        while openai_list and grok_list:
            openai_key = openai_list.pop(0)
            grok_key = grok_list.pop(0)
            openai_m = _serialize_measure(openai_index.get(openai_key, {}))
            grok_m = _serialize_measure(grok_index.get(grok_key, {}))
            conflicts['measure_conflicts'].append({
                'id': f"field_diff::{openai_key}::{grok_key}",
                'type': 'field_diff',
                'stable_key': heading,
                'openai_measure_key': openai_key,
                'grok_measure_key': grok_key,
                'heading': heading,
                'openai_data': openai_m,
                'grok_data': grok_m,
                'differences': compare_measure_fields(openai_m, grok_m)
            })
        openai_by_heading[heading] = openai_list
        grok_by_heading[heading] = grok_list

    # 剩余未配对的 only_in
    for heading, keys in openai_by_heading.items():
        for measure_key in keys:
            openai_m = _serialize_measure(openai_index.get(measure_key, {}))
            conflicts['measure_conflicts'].append({
                'id': f"only_in_openai|{measure_key}",
                'type': 'only_in_openai',
                'measure_key': measure_key,
                'heading': heading,
                'openai_data': openai_m
            })

    for heading, keys in grok_by_heading.items():
        for measure_key in keys:
            grok_m = _serialize_measure(grok_index.get(measure_key, {}))
            conflicts['measure_conflicts'].append({
                'id': f"only_in_grok|{measure_key}",
                'type': 'only_in_grok',
                'measure_key': measure_key,
                'heading': heading,
                'grok_data': grok_m
            })

    # Scope冲突
    for measure_key, scope_diff in compare_data.get('scope_diffs', {}).items():
        conflicts['scope_conflicts'].append({
            'id': f"scope_diff|{measure_key}",
            'measure_key': measure_key,
            'heading': measure_key.split('|')[0] if '|' in measure_key else measure_key,
            'only_in_openai': len(scope_diff.get('only_in_openai', [])),
            'only_in_grok': len(scope_diff.get('only_in_grok', [])),
            'openai_scopes': scope_diff.get('only_in_openai', []),
            'grok_scopes': scope_diff.get('only_in_grok', [])
        })

    return conflicts

def compare_measure_fields(openai_m, grok_m):
    """比较两个措施的字段差异"""
    differences = []

    fields = [
        ('country_iso2', '国家'),
        ('effective_start_date', '生效日期'),
        ('effective_end_date', '结束日期'),
        ('is_potential', '是否潜在'),
        ('ad_valorem_rate', '税率')
    ]

    for field, label in fields:
        openai_val = openai_m.get(field)
        grok_val = grok_m.get(field)

        if openai_val != grok_val:
            differences.append({
                'field': field,
                'label': label,
                'openai_value': _serialize_value(openai_val),
                'grok_value': _serialize_value(grok_val)
            })

    return differences

def build_unified_json(session):
    """构建统一JSON"""
    unified = {
        'note_number': session['note_id'],
        'session_id': session.get('session_id'),
        'reviewed_at': datetime.now().isoformat(),
        'measures': []
    }

    openai_measures = session['openai_data'].get('measures', [])
    grok_measures = session['grok_data'].get('measures', [])
    compare_data = session.get('compare_data', {})
    resolutions = session.get('resolutions', {})

    openai_index = _build_measure_key_index(openai_measures)
    grok_index = _build_measure_key_index(grok_measures)

    missing_in_openai = set(compare_data.get('missing_in_openai', []))
    missing_in_grok = set(compare_data.get('missing_in_grok', []))
    field_diff_keys = set(compare_data.get('field_diffs', {}).keys())
    scope_diff_keys = set(compare_data.get('scope_diffs', {}).keys())

    all_keys = set(openai_index) | set(grok_index)
    unified_measures: Dict[str, Dict[str, Any]] = {}

    skipped_keys: set[str] = set()
    for conflict_id, resolution in resolutions.items():
        if not conflict_id.startswith("field_diff::"):
            continue
        parts = conflict_id.split("::", 2)
        if len(parts) != 3:
            continue
        openai_key = parts[1]
        grok_key = parts[2]
        if resolution.get("resolution") == "grok":
            skipped_keys.add(openai_key)
        elif resolution.get("resolution") == "openai":
            skipped_keys.add(grok_key)

    for measure_key in sorted(all_keys):
        if measure_key in skipped_keys:
            continue
        if measure_key in missing_in_grok:
            resolution = resolutions.get(f"only_in_openai|{measure_key}", {}).get("resolution")
            if resolution == "reject":
                continue
            base_measure = openai_index.get(measure_key)
        elif measure_key in missing_in_openai:
            resolution = resolutions.get(f"only_in_grok|{measure_key}", {}).get("resolution")
            if resolution == "reject":
                continue
            base_measure = grok_index.get(measure_key)
        else:
            resolution = resolutions.get(f"field_diff|{measure_key}", {}).get("resolution")
            if resolution == "grok":
                base_measure = grok_index.get(measure_key)
            else:
                base_measure = openai_index.get(measure_key)

        if not base_measure:
            continue

        merged_measure = dict(base_measure)

        if measure_key in scope_diff_keys:
            scope_entry = resolutions.get(f"scope_diff|{measure_key}", {})
            scope_resolution = scope_entry.get("resolution")
            openai_scopes = openai_index.get(measure_key, {}).get("scopes", [])
            grok_scopes = grok_index.get(measure_key, {}).get("scopes", [])
            if scope_resolution == "grok":
                merged_measure["scopes"] = grok_scopes
            elif scope_resolution == "union":
                merged_measure["scopes"] = _scope_union(openai_scopes, grok_scopes)
            elif scope_resolution == "manual":
                manual_entries = _build_manual_scope_selection(scope_entry.get("value"))
                merged_measure["scopes"] = _scope_manual_merge(openai_scopes, grok_scopes, manual_entries)
            else:
                merged_measure["scopes"] = openai_scopes
        else:
            if "scopes" not in merged_measure:
                merged_measure["scopes"] = base_measure.get("scopes", [])

        unified_measures[measure_key] = merged_measure

    unified['measures'] = list(unified_measures.values())

    return unified


def compare_unified_with_db(unified_measures: List[Dict[str, Any]], db_measures: List[Dict[str, Any]]) -> Dict[str, Any]:
    unified_index = _index_measures_by_key(unified_measures)
    db_index = _index_measures_by_key(db_measures)
    base = _compare_measure_maps(unified_index, db_index, "unified", "db")
    return {
        "consistent": base["consistent"],
        "summary": base["summary"],
        "missing_in_db": base["missing_in_right"],
        "extra_in_db": base["missing_in_left"],
        "field_diffs": base["field_diffs"],
        "scope_diffs": base["scope_diffs"],
    }


def fetch_db_measures(note_number: int, config: Dict[str, Any], headings: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is required for database access")
    if not DB_DSN:
        raise RuntimeError("DATABASE_URL is required for database access")
    conn = psycopg2.connect(DB_DSN)
    try:
        measures_table = config.get("measures_table")
        scope_table = config.get("scope_table")
        map_table = config.get("map_table")
        query_by = config.get("query_by")
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if query_by == "note_number":
                cur.execute(
                    f"SELECT id, heading, country_iso2, ad_valorem_rate, value_basis, notes, "
                    f"effective_start_date, effective_end_date, is_potential "
                    f"FROM {measures_table} WHERE notes->>'note_number' = %s",
                    (str(note_number),),
                )
            elif query_by == "headings":
                if not headings:
                    return []
                cur.execute(
                    f"SELECT id, heading, country_iso2, ad_valorem_rate, value_basis, notes, "
                    f"effective_start_date, effective_end_date, is_potential "
                    f"FROM {measures_table} WHERE heading = ANY(%s)",
                    (headings,),
                )
            else:
                return []
            rows = cur.fetchall() or []

        if not rows:
            return []

        measures: List[Dict[str, Any]] = []
        measure_map: Dict[int, Dict[str, Any]] = {}
        measure_ids: List[int] = []
        for row in rows:
            measure_id = int(row["id"])
            measure_ids.append(measure_id)
            entry = {
                "heading": row["heading"],
                "country_iso2": row["country_iso2"],
                "ad_valorem_rate": row["ad_valorem_rate"],
                "value_basis": row.get("value_basis"),
                "notes": row.get("notes"),
                "effective_start_date": row["effective_start_date"],
                "effective_end_date": row["effective_end_date"],
                "is_potential": row["is_potential"],
                "scopes": [],
            }
            measures.append(entry)
            measure_map[measure_id] = entry

        with conn.cursor() as cur:
            cur.execute(
                f"SELECT m.measure_id, s.key, m.relation "
                f"FROM {map_table} AS m "
                f"JOIN {scope_table} AS s ON s.id = m.scope_id "
                "WHERE m.measure_id = ANY(%s)",
                (measure_ids,),
            )
            for measure_id, key, relation in cur.fetchall():
                target = measure_map.get(int(measure_id))
                if not target:
                    continue
                target["scopes"].append({"key": key, "relation": relation})

        return measures
    finally:
        conn.close()


def parse_db_conflicts(db_compare: Dict[str, Any], unified_data: Dict[str, Any], db_data: List[Dict[str, Any]]):
    conflicts = {
        'measure_conflicts': [],
        'scope_conflicts': []
    }
    unified_index = _build_measure_key_index(unified_data.get('measures', []))
    db_index = _build_measure_key_index(db_data)

    for measure_key in db_compare.get('field_diffs', {}).keys():
        unified_m = _serialize_measure(unified_index.get(measure_key, {}))
        db_m = _serialize_measure(db_index.get(measure_key, {}))
        heading = measure_key.split('|')[0] if '|' in measure_key else measure_key
        conflicts['measure_conflicts'].append({
            'id': f"db_field_diff|{measure_key}",
            'type': 'db_field_diff',
            'measure_key': measure_key,
            'heading': heading,
            'unified_data': unified_m,
            'db_data': db_m,
            'differences': compare_measure_fields(unified_m, db_m)
        })

    missing_in_db = list(db_compare.get('missing_in_db', []))
    extra_in_db = list(db_compare.get('extra_in_db', []))

    unified_by_heading: Dict[str, List[str]] = {}
    db_by_heading: Dict[str, List[str]] = {}
    for measure_key in missing_in_db:
        heading = measure_key.split('|')[0] if '|' in measure_key else measure_key
        unified_by_heading.setdefault(heading, []).append(measure_key)
    for measure_key in extra_in_db:
        heading = measure_key.split('|')[0] if '|' in measure_key else measure_key
        db_by_heading.setdefault(heading, []).append(measure_key)

    for heading in sorted(set(unified_by_heading) & set(db_by_heading)):
        unified_list = sorted(unified_by_heading.get(heading, []))
        db_list = sorted(db_by_heading.get(heading, []))
        while unified_list and db_list:
            unified_key = unified_list.pop(0)
            db_key = db_list.pop(0)
            unified_m = _serialize_measure(unified_index.get(unified_key, {}))
            db_m = _serialize_measure(db_index.get(db_key, {}))
            conflicts['measure_conflicts'].append({
                'id': f"db_field_diff::{unified_key}::{db_key}",
                'type': 'db_field_diff',
                'stable_key': heading,
                'unified_measure_key': unified_key,
                'db_measure_key': db_key,
                'heading': heading,
                'unified_data': unified_m,
                'db_data': db_m,
                'differences': compare_measure_fields(unified_m, db_m)
            })
        unified_by_heading[heading] = unified_list
        db_by_heading[heading] = db_list

    for heading, keys in unified_by_heading.items():
        for measure_key in keys:
            unified_m = _serialize_measure(unified_index.get(measure_key, {}))
            conflicts['measure_conflicts'].append({
                'id': f"db_only_in_unified|{measure_key}",
                'type': 'only_in_unified',
                'measure_key': measure_key,
                'heading': heading,
                'unified_data': unified_m
            })

    for heading, keys in db_by_heading.items():
        for measure_key in keys:
            db_m = _serialize_measure(db_index.get(measure_key, {}))
            conflicts['measure_conflicts'].append({
                'id': f"db_only_in_db|{measure_key}",
                'type': 'only_in_db',
                'measure_key': measure_key,
                'heading': heading,
                'db_data': db_m
            })

    for measure_key, scope_diff in db_compare.get('scope_diffs', {}).items():
        conflicts['scope_conflicts'].append({
            'id': f"db_scope_diff|{measure_key}",
            'type': 'db_scope_diff',
            'measure_key': measure_key,
            'heading': measure_key.split('|')[0] if '|' in measure_key else measure_key,
            'only_in_unified': len(scope_diff.get('only_in_unified', [])),
            'only_in_db': len(scope_diff.get('only_in_db', [])),
            'unified_scopes': scope_diff.get('only_in_unified', []),
            'db_scopes': scope_diff.get('only_in_db', [])
        })

    return conflicts


def apply_db_resolutions(unified_data: Dict[str, Any], db_data: List[Dict[str, Any]], db_resolutions: Dict[str, Any]) -> Dict[str, Any]:
    unified_index = _build_measure_key_index(unified_data.get('measures', []))
    db_index = _build_measure_key_index(db_data)

    final_measures: Dict[str, Dict[str, Any]] = dict(unified_index)

    for conflict_id, resolution in db_resolutions.items():
        decision = resolution.get("resolution")
        if "||" in conflict_id:
            continue
        if conflict_id.startswith("db_field_diff::"):
            parts = conflict_id.split("::", 2)
            if len(parts) != 3:
                continue
            unified_key = parts[1]
            db_key = parts[2]
            if decision == "db":
                db_measure = db_index.get(db_key)
                if db_measure:
                    existing = final_measures.get(unified_key, {})
                    db_copy = dict(db_measure)
                    db_copy["scopes"] = existing.get("scopes", db_measure.get("scopes", []))
                    final_measures[unified_key] = db_copy
            continue

        if "|" not in conflict_id:
            continue
        _, measure_key = conflict_id.split("|", 1)

        if conflict_id.startswith("db_only_in_unified|"):
            if decision == "skip":
                final_measures.pop(measure_key, None)
        elif conflict_id.startswith("db_only_in_db|"):
            if decision == "keep":
                db_measure = db_index.get(measure_key)
                if db_measure:
                    final_measures[measure_key] = db_measure
            elif decision == "delete":
                final_measures.pop(measure_key, None)
        elif conflict_id.startswith("db_scope_diff|"):
            if decision == "db":
                db_measure = db_index.get(measure_key)
                if db_measure:
                    existing = final_measures.get(measure_key, {})
                    merged = dict(existing)
                    merged["scopes"] = db_measure.get("scopes", [])
                    final_measures[measure_key] = merged
            elif decision == "manual":
                db_measure = db_index.get(measure_key, {})
                existing = final_measures.get(measure_key, {})
                unified_scopes = existing.get("scopes", [])
                db_scopes = db_measure.get("scopes", [])
                manual_entries = _build_manual_scope_selection(resolution.get("value"))
                merged_scopes = _scope_manual_merge_db(unified_scopes, db_scopes, manual_entries)
                merged = dict(existing)
                merged["scopes"] = merged_scopes
                final_measures[measure_key] = merged

    return {
        "note_number": unified_data.get("note_number"),
        "session_id": unified_data.get("session_id"),
        "reviewed_at": datetime.now().isoformat(),
        "measures": list(final_measures.values()),
    }


def _expand_scope_entry_for_db(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    key = entry.get("key") or ""
    keys = entry.get("keys")
    key_type = entry.get("key_type")
    country_iso2 = _normalize_iso(entry.get("country_iso2"))
    source_label = entry.get("source_label")
    start = _parse_date(entry.get("effective_start_date")) or DEFAULT_START_DATE
    end = _parse_date(entry.get("effective_end_date"))

    scopes: List[Dict[str, Any]] = []
    if isinstance(keys, list):
        raw_keys = [str(item) for item in keys if item is not None]
    elif keys:
        raw_keys = _split_keys(str(keys))
    elif key:
        raw_keys = [str(key)]
    else:
        raw_keys = []

    for raw in raw_keys:
        normalized_key = _normalize_code(raw)
        if not normalized_key:
            continue
        scopes.append(
            {
                "key": normalized_key,
                "key_type": _normalize_key_type(key_type, normalized_key),
                "country_iso2": country_iso2,
                "source_label": source_label,
                "effective_start_date": start,
                "effective_end_date": end,
            }
        )
    return scopes


def _expand_scope_entries_for_db(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for entry in entries:
        relation = entry.get("relation") or "include"
        note_label = entry.get("note_label")
        text_criteria = entry.get("text_criteria")
        map_start = _parse_date(entry.get("effective_start_date")) or DEFAULT_START_DATE
        map_end = _parse_date(entry.get("effective_end_date"))
        for scope in _expand_scope_entry_for_db(entry):
            expanded.append(
                {
                    "scope": scope,
                    "relation": relation,
                    "note_label": note_label,
                    "text_criteria": text_criteria,
                    "map_start": map_start,
                    "map_end": map_end,
                }
            )
    return expanded


def persist_measures_to_db(note_number: int, measures: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is required for database access")
    if not DB_DSN:
        raise RuntimeError("DATABASE_URL is required for database access")
    if execute_values is None:
        raise RuntimeError("psycopg2 execute_values is required for database access")

    conn = psycopg2.connect(DB_DSN)
    try:
        measures_table = config.get("measures_table")
        scope_table = config.get("scope_table")
        map_table = config.get("map_table")
        query_by = config.get("query_by")
        with conn:
            with conn.cursor() as cur:
                if query_by == "note_number":
                    cur.execute(
                        f"SELECT id FROM {measures_table} WHERE notes->>'note_number' = %s",
                        (str(note_number),),
                    )
                    measure_ids = [row[0] for row in cur.fetchall()]
                elif query_by == "headings":
                    headings = [
                        _normalize_code(str(entry.get("heading", "")))
                        for entry in measures
                        if entry.get("heading")
                    ]
                    if headings:
                        cur.execute(
                            f"SELECT id FROM {measures_table} WHERE heading = ANY(%s)",
                            (headings,),
                        )
                        measure_ids = [row[0] for row in cur.fetchall()]
                    else:
                        measure_ids = []
                else:
                    measure_ids = []
                if measure_ids:
                    cur.execute(
                        f"DELETE FROM {map_table} WHERE measure_id = ANY(%s)",
                        (measure_ids,),
                    )
                    cur.execute(
                        f"DELETE FROM {measures_table} WHERE id = ANY(%s)",
                        (measure_ids,),
                    )

        measure_records: List[Dict[str, Any]] = []
        scopes_by_measure: List[List[Dict[str, Any]]] = []
        for entry in measures:
            heading = _normalize_code(str(entry.get("heading", "")))
            if not heading:
                continue
            start = _parse_date(entry.get("effective_start_date")) or DEFAULT_START_DATE
            end = _parse_date(entry.get("effective_end_date"))
            notes = entry.get("notes") or {}
            if not isinstance(notes, dict):
                notes = {"value": notes}
            if query_by == "note_number":
                notes.setdefault("note_number", note_number)
            record = {
                "heading": heading,
                "country_iso2": _normalize_iso(entry.get("country_iso2")),
                "ad_valorem_rate": _parse_rate(entry.get("ad_valorem_rate")) or Decimal("0"),
                "value_basis": entry.get("value_basis") or "total_value",
                "melt_pour_origin_iso2": _normalize_iso(entry.get("melt_pour_origin_iso2")),
                "origin_exclude_iso2": entry.get("origin_exclude_iso2"),
                "notes": notes,
                "effective_start_date": start,
                "effective_end_date": end,
                "is_potential": _coerce_bool(entry.get("is_potential")),
            }
            measure_records.append(record)
            scopes_by_measure.append(_expand_scope_entries_for_db(entry.get("scopes") or []))

        if not measure_records:
            return {"inserted_measures": 0, "inserted_scopes": 0}

        values = [
            (
                record["heading"],
                record["country_iso2"],
                record["ad_valorem_rate"],
                record["value_basis"],
                record["melt_pour_origin_iso2"],
                record["origin_exclude_iso2"],
                Json(record["notes"]),
                record["effective_start_date"],
                record["effective_end_date"],
                record["is_potential"],
            )
            for record in measure_records
        ]
        insert_query = (
            f"INSERT INTO {measures_table} "
            "(heading, country_iso2, ad_valorem_rate, value_basis, melt_pour_origin_iso2, "
            "origin_exclude_iso2, notes, effective_start_date, effective_end_date, is_potential) "
            "VALUES %s ON CONFLICT DO NOTHING"
        )
        with conn:
            with conn.cursor() as cur:
                execute_values(cur, insert_query, values, page_size=len(values))
                lookup_values = [
                    (
                        idx,
                        record["heading"],
                        record["country_iso2"],
                        record["effective_start_date"],
                        record["effective_end_date"],
                    )
                    for idx, record in enumerate(measure_records)
                ]
                lookup_query = (
                    f"SELECT v.idx, m.id "
                    f"FROM (VALUES %s) AS v(idx, heading, country_iso2, start_date, end_date) "
                    f"JOIN {measures_table} AS m "
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
                measure_ids = [row[1] for row in rows]

        unique_scopes: List[Dict[str, Any]] = []
        scope_index: Dict[Tuple[str, str, Optional[str], date, Optional[date]], int] = {}
        pending_maps: List[Tuple[Tuple[str, str, Optional[str], date, Optional[date]], int, Dict[str, Any]]] = []

        for measure_id, scope_entries in zip(measure_ids, scopes_by_measure):
            for entry in scope_entries:
                scope = entry["scope"]
                signature = (
                    scope["key"],
                    scope["key_type"],
                    scope["country_iso2"],
                    scope["effective_start_date"],
                    scope["effective_end_date"],
                )
                if signature not in scope_index:
                    scope_index[signature] = len(unique_scopes)
                    unique_scopes.append(scope)
                pending_maps.append((signature, measure_id, entry))

        if unique_scopes:
            scope_values = [
                (
                    scope["key"],
                    scope["key_type"],
                    scope["country_iso2"],
                    scope["source_label"],
                    scope["effective_start_date"],
                    scope["effective_end_date"],
                )
                for scope in unique_scopes
            ]
            scope_insert = (
                f"INSERT INTO {scope_table} "
                "(key, key_type, country_iso2, source_label, effective_start_date, effective_end_date) "
                "VALUES %s ON CONFLICT DO NOTHING"
            )
            with conn:
                with conn.cursor() as cur:
                    execute_values(cur, scope_insert, scope_values, page_size=len(scope_values))
                    lookup_values = [
                        (
                            idx,
                            scope["key"],
                            scope["key_type"],
                            scope["country_iso2"],
                            scope["effective_start_date"],
                            scope["effective_end_date"],
                        )
                        for idx, scope in enumerate(unique_scopes)
                    ]
                    lookup_query = (
                        f"SELECT v.idx, s.id "
                        f"FROM (VALUES %s) AS v(idx, key, key_type, country_iso2, start_date, end_date) "
                        f"JOIN {scope_table} AS s "
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
                    scope_rows = cur.fetchall()
                    scope_ids = [row[1] for row in scope_rows]
        else:
            scope_ids = []

        signature_to_id = {
            signature: scope_ids[index] for signature, index in scope_index.items()
        }
        map_entries: List[Tuple[Any, ...]] = []
        seen_maps: set[Tuple[int, int, str, date, Optional[date]]] = set()
        for signature, measure_id, entry in pending_maps:
            scope_id = signature_to_id[signature]
            map_sig = (scope_id, measure_id, entry["relation"], entry["map_start"], entry["map_end"])
            if map_sig in seen_maps:
                continue
            seen_maps.add(map_sig)
            map_entries.append(
                (
                    scope_id,
                    measure_id,
                    entry["relation"],
                    entry["note_label"],
                    entry["text_criteria"],
                    entry["map_start"],
                    entry["map_end"],
                )
            )

        if map_entries:
            map_insert = (
                f"INSERT INTO {map_table} "
                "(scope_id, measure_id, relation, note_label, text_criteria, effective_start_date, effective_end_date) "
                "VALUES %s ON CONFLICT DO NOTHING"
            )
            with conn:
                with conn.cursor() as cur:
                    execute_values(cur, map_insert, map_entries, page_size=len(map_entries))

        return {"inserted_measures": len(measure_records), "inserted_scopes": len(unique_scopes)}
    finally:
        conn.close()

if __name__ == '__main__':
    app.run(debug=True, port=8081)
