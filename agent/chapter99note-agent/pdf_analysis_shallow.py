#!/usr/bin/env python3
"""Parse Chapter 99 PDF notes into structured entries (max depth 2)."""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import requests


LOGGER = logging.getLogger(__name__)

CHAPTER_NUMBER = 99
MAX_DEPTH = 2

SUBCHAPTER_RE = re.compile(r"^SUBCHAPTER\s+([IVXLCDM]+)\s*$", re.IGNORECASE)

HEADER_PATTERNS = [
    re.compile(r"^Harmonized Tariff Schedule of the United States", re.IGNORECASE),
    re.compile(r"^Annotated for Statistical Reporting Purposes$", re.IGNORECASE),
    re.compile(r"^U\.S\. Notes( \(con\.\))?$", re.IGNORECASE),
    re.compile(r"^Statistical Notes$", re.IGNORECASE),
    re.compile(r"^XXII$", re.IGNORECASE),
    re.compile(r"^99\s*-\s*[IVX]+\s*-\s*\d+$", re.IGNORECASE),
]

US_NOTES_RE = re.compile(r"^U\.S\. Notes( \(con\.\))?$", re.IGNORECASE)
STATISTICAL_NOTES_RE = re.compile(r"^Statistical Notes$", re.IGNORECASE)

TABLE_HEADER_TOKENS = {
    "HEADING/",
    "SUBHEADING",
    "RATES OF DUTY",
    "STAT.",
    "UNIT",
    "ARTICLE DESCRIPTION",
}
TABLE_HEADER_WINDOW = 12

NOTE_DOT_RE = re.compile(r"^\s*(\d{1,2})\.(?!\d)\s*(.*)$")
NOTE_PAREN_RE = re.compile(r"^\s*\(\s*(\d{1,2})\s*\)\s*(.*)$")
PAREN_RE = re.compile(r"^\s*\(\s*([^)]+?)\s*\)\s*(.*)$")
DOT_RE = re.compile(r"^\s*(\d{1,3})\.(?:\s+|$)(.*)$")
ROMAN_RE = re.compile(
    r"^m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$",
    re.IGNORECASE,
)

LEADING_PAREN_RE = re.compile(r"^\s*\(\s*([^)]+?)\s*\)\s*")
LEADING_DOT_RE = re.compile(r"^\s*(\d{1,3})\.(?:\s+|$)")

DEFAULT_INDENT_SAME_THRESHOLD = 4.0
DEFAULT_NOTE_INDENT_THRESHOLD = 6.0
SAME_TYPE_NEST_INDENT = 12.0
INDENT_CLUSTER_THRESHOLD = 3.0


@dataclass
class TextLine:
    text: str
    norm_text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page: int


@dataclass
class Node:
    tokens: tuple[str, ...]
    label: str
    parent_label: Optional[str]
    marker_indent: float
    marker_type: Optional[str]
    subchapter: Optional[str]
    order_index: int
    creation_order: int
    content_lines: list[str] = field(default_factory=list)
    content_indent_min: Optional[float] = None
    parent_tokens: Optional[tuple[str, ...]] = None


@dataclass
class MarkerCandidate:
    line_index: int
    line: TextLine
    token: str
    kind: str
    remainder_raw: str
    cluster_id: int = -1
    marker_type: Optional[str] = None


@dataclass
class MarkerSpan:
    token: str
    kind: str
    start: int
    end: int


@dataclass
class ClusterStats:
    alpha_transitions_lower: int = 0
    alpha_transitions_upper: int = 0
    roman_transitions_lower: int = 0
    roman_transitions_upper: int = 0
    alpha_repeats_lower: int = 0
    alpha_repeats_upper: int = 0
    roman_tokens_lower: int = 0
    roman_tokens_upper: int = 0


def _normalize_for_match(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _iter_text_lines(layout_obj: object) -> Iterable[object]:
    try:
        from pdfminer.layout import LTTextLine
    except ModuleNotFoundError:
        return []

    if isinstance(layout_obj, LTTextLine):
        yield layout_obj
        return

    children = getattr(layout_obj, "_objs", None)
    if not children:
        return

    for child in children:
        yield from _iter_text_lines(child)


def _bbox_intersects(
    bbox: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
) -> bool:
    x0, y0, x1, y1 = bbox
    cx0, cy0, cx1, cy1 = crop
    return x1 > cx0 and x0 < cx1 and y1 > cy0 and y0 < cy1


def _load_crop_boxes(pdf_path: Path) -> list[tuple[float, float, float, float]]:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: install pypdf.") from exc

    reader = PdfReader(str(pdf_path), strict=False)
    boxes: list[tuple[float, float, float, float]] = []
    for page in reader.pages:
        crop = page.cropbox or page.mediabox
        x0, y0 = crop.lower_left
        x1, y1 = crop.upper_right
        boxes.append((float(x0), float(y0), float(x1), float(y1)))
    return boxes


def _extract_lines(pdf_path: Path, *, use_cropbox: bool = True) -> list[TextLine]:
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LAParams
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: install pdfminer.six.") from exc

    crop_boxes = _load_crop_boxes(pdf_path) if use_cropbox else []
    laparams = LAParams(
        line_margin=0.2,
        char_margin=2.0,
        word_margin=0.1,
        boxes_flow=None,
    )

    lines: list[TextLine] = []
    for page_index, page_layout in enumerate(
        extract_pages(str(pdf_path), laparams=laparams)
    ):
        crop_box = crop_boxes[page_index] if page_index < len(crop_boxes) else None
        for line in _iter_text_lines(page_layout):
            if crop_box is not None and not _bbox_intersects(line.bbox, crop_box):
                continue
            raw_text = line.get_text()
            if not raw_text.strip():
                continue
            raw_text = raw_text.rstrip("\n")
            norm_text = _normalize_for_match(raw_text)
            if not norm_text:
                continue
            x0, y0, x1, y1 = line.bbox
            lines.append(
                TextLine(
                    text=raw_text,
                    norm_text=norm_text,
                    x0=float(x0),
                    y0=float(y0),
                    x1=float(x1),
                    y1=float(y1),
                    page=page_index + 1,
                )
            )
    return lines


def _is_header(text: str) -> bool:
    return any(pattern.match(text) for pattern in HEADER_PATTERNS)


def _build_label(tokens: tuple[str, ...]) -> str:
    return "note(" + ")(".join(tokens) + ")"


def _format_path(tokens: tuple[str, ...]) -> str:
    return "{" + ",".join(("note",) + tokens) + "}"


def _is_roman(token: str) -> bool:
    return bool(token) and bool(ROMAN_RE.match(token))


def _roman_to_int(token: str) -> Optional[int]:
    if not token:
        return None
    token = token.lower()
    if not ROMAN_RE.match(token):
        return None
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    prev = 0
    for ch in reversed(token):
        value = values.get(ch)
        if value is None:
            return None
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    return total


def _alpha_index(token: str) -> Optional[int]:
    if not token.isalpha():
        return None
    value = 0
    for ch in token.lower():
        if ch < "a" or ch > "z":
            return None
        value = value * 26 + (ord(ch) - ord("a") + 1)
    return value


def _repeat_letter_key(token: str) -> Optional[tuple[int, int]]:
    if not token.isalpha() or len(token) < 2:
        return None
    lowered = token.lower()
    if len(set(lowered)) != 1:
        return None
    return len(lowered), ord(lowered[0]) - ord("a") + 1


def _alpha_sequence(prev_token: Optional[str], token: Optional[str]) -> bool:
    if not prev_token or not token:
        return False
    if not (prev_token.isalpha() and token.isalpha()):
        return False
    if prev_token.isupper() != token.isupper():
        return False
    prev_idx = _alpha_index(prev_token.lower())
    curr_idx = _alpha_index(token.lower())
    if prev_idx and curr_idx and curr_idx == prev_idx + 1:
        return True
    prev_repeat = _repeat_letter_key(prev_token)
    curr_repeat = _repeat_letter_key(token)
    if (
        prev_repeat
        and curr_repeat
        and prev_repeat[0] == curr_repeat[0]
        and curr_repeat[1] == prev_repeat[1] + 1
    ):
        return True
    return False


def _roman_sequence(prev_token: Optional[str], token: Optional[str]) -> bool:
    if not prev_token or not token:
        return False
    if not (prev_token.isalpha() and token.isalpha()):
        return False
    if prev_token.isupper() != token.isupper():
        return False
    prev_val = _roman_to_int(prev_token.lower())
    curr_val = _roman_to_int(token.lower())
    if prev_val and curr_val and curr_val == prev_val + 1:
        return True
    return False


def _is_structural_token(token: str) -> bool:
    token = token.strip()
    if not token:
        return False
    if not token.isalnum():
        return False
    if token.isdigit():
        return len(token) <= 3
    if not token.isalpha():
        return False
    if _is_roman(token):
        return True
    lowered = token.lower()
    if len(lowered) == 1:
        return True
    if len(lowered) <= 4 and len(set(lowered)) == 1:
        return True
    return False


def _cluster_indents(values: list[float], threshold: float) -> list[float]:
    if not values:
        return []
    values_sorted = sorted(values)
    clusters: list[list[float]] = [[values_sorted[0]]]
    for value in values_sorted[1:]:
        if abs(value - clusters[-1][-1]) <= threshold:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    centers = [sum(group) / len(group) for group in clusters]
    return centers


def _strip_prefix(raw: str, pattern: re.Pattern) -> str:
    match = pattern.match(raw)
    if not match:
        return raw
    return raw[match.end() :]


def _strip_marker_prefix(raw: str, token: str, kind: str) -> str:
    if kind == "paren":
        pattern = re.compile(
            rf"^\s*\(\s*{re.escape(token)}\s*\)\s*",
            re.IGNORECASE,
        )
    else:
        pattern = re.compile(rf"^\s*{re.escape(token)}\.\s*")
    return _strip_prefix(raw, pattern)


def _strip_note_prefix(raw: str, token: str) -> str:
    dot_pattern = re.compile(rf"^\s*{re.escape(token)}\.\s*")
    paren_pattern = re.compile(
        rf"^\s*\(\s*{re.escape(token)}\s*\)\s*",
        re.IGNORECASE,
    )
    stripped = _strip_prefix(raw, dot_pattern)
    if stripped != raw:
        return stripped
    return _strip_prefix(raw, paren_pattern)


def _note_start_remainder_ok(remainder: str) -> bool:
    remainder = remainder.strip()
    return not remainder or remainder.startswith("(")


class ShallowNoteParser:
    def __init__(
        self,
        lines: list[TextLine],
        note_numbers: set[str],
        *,
        max_depth: int = MAX_DEPTH,
        indent_same_threshold: float = DEFAULT_INDENT_SAME_THRESHOLD,
        note_indent_threshold: float = DEFAULT_NOTE_INDENT_THRESHOLD,
    ) -> None:
        self.lines = sorted(lines, key=lambda line: (line.page, -line.y1, line.x0))
        self.note_numbers = note_numbers
        self.max_depth = max_depth
        self.indent_same_threshold = indent_same_threshold
        self.note_indent_threshold = note_indent_threshold
        self.same_type_indent_threshold = SAME_TYPE_NEST_INDENT
        self.nodes: dict[tuple[str, tuple[str, ...]], Node] = {}
        self.node_order: list[Node] = []
        self.seen_notes: set[str] = set()
        self._creation_counter = 0

    def _get_or_create_node(
        self,
        subchapter: str,
        tokens: tuple[str, ...],
        marker_indent: float,
        marker_type: Optional[str],
        order_index: int,
    ) -> Node:
        key = (subchapter, tokens)
        node = self.nodes.get(key)
        if node:
            return node
        label = _build_label(tokens)
        parent_tokens = tokens[:-1] if len(tokens) > 1 else None
        parent_label = _build_label(parent_tokens) if parent_tokens else None
        self._creation_counter += 1
        node = Node(
            tokens=tokens,
            label=label,
            parent_label=parent_label,
            marker_indent=marker_indent,
            marker_type=marker_type,
            subchapter=subchapter,
            order_index=order_index,
            creation_order=self._creation_counter,
            parent_tokens=parent_tokens,
        )
        self.nodes[key] = node
        self.node_order.append(node)
        return node

    def _find_subchapters(self) -> list[tuple[str, int, int]]:
        matches: list[tuple[int, str]] = []
        for idx, line in enumerate(self.lines):
            match = SUBCHAPTER_RE.match(line.norm_text)
            if match:
                roman = match.group(1).upper()
                matches.append((idx, f"SUBCHAPTER {roman}"))
        ranges: list[tuple[str, int, int]] = []
        for i, (idx, label) in enumerate(matches):
            end = matches[i + 1][0] if i + 1 < len(matches) else len(self.lines)
            ranges.append((label, idx, end))
        return ranges

    def _find_note_start_indent(
        self,
        start_idx: int,
        end_idx: int,
    ) -> tuple[Optional[float], bool]:
        dot_x0s: list[float] = []
        paren_x0s: list[float] = []
        for idx in range(start_idx, end_idx):
            line = self.lines[idx]
            if _is_header(line.norm_text):
                continue
            dot_match = NOTE_DOT_RE.match(line.norm_text)
            if dot_match and _note_start_remainder_ok(dot_match.group(2)):
                dot_x0s.append(line.x0)
                continue
            paren_match = NOTE_PAREN_RE.match(line.norm_text)
            if paren_match and _note_start_remainder_ok(paren_match.group(2)):
                paren_x0s.append(line.x0)
        if dot_x0s:
            return min(dot_x0s), False
        if paren_x0s:
            return min(paren_x0s), True
        return None, False

    def _is_table_header_start(self, idx: int, end_idx: int) -> bool:
        token = self.lines[idx].norm_text.upper()
        if token not in TABLE_HEADER_TOKENS:
            return False
        look_end = min(end_idx, idx + TABLE_HEADER_WINDOW)
        hits: set[str] = set()
        for j in range(idx, look_end):
            candidate = self.lines[j].norm_text.upper()
            if candidate in TABLE_HEADER_TOKENS:
                hits.add(candidate)
        if "HEADING/" in hits and "SUBHEADING" in hits:
            if "RATES OF DUTY" in hits or ("STAT." in hits and "UNIT" in hits):
                return True
        return False

    def _find_us_notes_range(
        self,
        start_idx: int,
        end_idx: int,
    ) -> Optional[tuple[int, int]]:
        us_start = None
        for idx in range(start_idx, end_idx):
            if US_NOTES_RE.match(self.lines[idx].norm_text):
                us_start = idx + 1
                break
        if us_start is None:
            note_indent, allow_paren = self._find_note_start_indent(start_idx, end_idx)
            if note_indent is None:
                return None
            for idx in range(start_idx, end_idx):
                if self._match_note_start(
                    self.lines[idx],
                    left_margin=note_indent,
                    allow_paren=allow_paren,
                ):
                    us_start = idx
                    break
        if us_start is None:
            return None
        us_end = end_idx
        for idx in range(us_start, end_idx):
            if STATISTICAL_NOTES_RE.match(self.lines[idx].norm_text):
                us_end = idx
                break
            if self._is_table_header_start(idx, end_idx):
                us_end = idx
                break
        return (us_start, us_end)

    def _match_note_start(
        self,
        line: TextLine,
        *,
        left_margin: float,
        allow_paren: bool = False,
    ) -> Optional[tuple[str, str]]:
        if not line.norm_text:
            return None
        if line.x0 > left_margin + self.note_indent_threshold:
            return None
        dot_match = NOTE_DOT_RE.match(line.norm_text)
        if dot_match:
            token = dot_match.group(1)
            if not _note_start_remainder_ok(dot_match.group(2)):
                return None
            remainder_raw = _strip_note_prefix(line.text, token)
            return token, remainder_raw
        if allow_paren:
            paren_match = NOTE_PAREN_RE.match(line.norm_text)
            if paren_match:
                token = paren_match.group(1)
                if not _note_start_remainder_ok(paren_match.group(2)):
                    return None
                remainder_raw = _strip_note_prefix(line.text, token)
                return token, remainder_raw
        return None

    def _extract_marker_candidate(
        self, line_index: int, line: TextLine
    ) -> Optional[MarkerCandidate]:
        paren_match = PAREN_RE.match(line.norm_text)
        if paren_match:
            token = paren_match.group(1).strip()
            if _is_structural_token(token):
                remainder_raw = _strip_marker_prefix(line.text, token, "paren")
                return MarkerCandidate(
                    line_index=line_index,
                    line=line,
                    token=token,
                    kind="paren",
                    remainder_raw=remainder_raw,
                )
        dot_match = DOT_RE.match(line.norm_text)
        if dot_match:
            token = dot_match.group(1).strip()
            if _is_structural_token(token):
                remainder_raw = _strip_marker_prefix(line.text, token, "dot")
                return MarkerCandidate(
                    line_index=line_index,
                    line=line,
                    token=token,
                    kind="dot",
                    remainder_raw=remainder_raw,
                )
        return None

    def _line_starts_with_marker(self, line: TextLine) -> bool:
        if NOTE_DOT_RE.match(line.norm_text) or NOTE_PAREN_RE.match(line.norm_text):
            return False
        paren_match = PAREN_RE.match(line.norm_text)
        if paren_match:
            token = paren_match.group(1).strip()
            return _is_structural_token(token)
        dot_match = DOT_RE.match(line.norm_text)
        if dot_match:
            token = dot_match.group(1).strip()
            return _is_structural_token(token)
        return False

    def _is_inline_reference(
        self,
        candidate: MarkerCandidate,
        prev_line: Optional[TextLine],
    ) -> bool:
        remainder = candidate.remainder_raw.lstrip()
        if not remainder:
            return False
        first = remainder[0]
        if first in ",;":
            return True
        if not first.islower():
            return False
        if prev_line is None:
            return False
        if candidate.line.x0 + self.indent_same_threshold < prev_line.x0:
            return False
        if self._line_starts_with_marker(prev_line):
            return False
        prev_text = prev_line.text.strip()
        if prev_text.endswith(":"):
            return False
        return True

    def _collect_marker_candidates(
        self,
        start_idx: int,
        end_idx: int,
        note_start_indices: set[int],
    ) -> list[MarkerCandidate]:
        candidates: list[MarkerCandidate] = []
        prev_line: Optional[TextLine] = None
        for idx in range(start_idx, end_idx):
            if idx in note_start_indices:
                prev_line = self.lines[idx]
                continue
            line = self.lines[idx]
            if _is_header(line.norm_text):
                continue
            candidate = self._extract_marker_candidate(idx, line)
            if candidate and not self._is_inline_reference(candidate, prev_line):
                candidates.append(candidate)
            prev_line = line
        return candidates

    def _assign_clusters(self, candidates: list[MarkerCandidate]) -> list[float]:
        centers = _cluster_indents(
            [candidate.line.x0 for candidate in candidates],
            INDENT_CLUSTER_THRESHOLD,
        )
        if not centers:
            return []
        for candidate in candidates:
            candidate.cluster_id = min(
                range(len(centers)),
                key=lambda idx: abs(candidate.line.x0 - centers[idx]),
            )
        return centers

    def _build_cluster_stats(
        self, candidates: list[MarkerCandidate]
    ) -> tuple[dict[int, ClusterStats], dict[int, list[MarkerCandidate]]]:
        clusters: dict[int, list[MarkerCandidate]] = {}
        for candidate in candidates:
            clusters.setdefault(candidate.cluster_id, []).append(candidate)
        stats_map: dict[int, ClusterStats] = {}
        for cluster_id, items in clusters.items():
            items_sorted = sorted(items, key=lambda c: c.line_index)
            stats = ClusterStats()
            for item in items_sorted:
                token = item.token
                if not token.isalpha():
                    continue
                token_case = "upper" if token.isupper() else "lower"
                roman_val = _roman_to_int(token.lower())
                if roman_val and len(token) > 1:
                    if token_case == "upper":
                        stats.roman_tokens_upper += 1
                    else:
                        stats.roman_tokens_lower += 1
                if len(token) > 1 and len(set(token.lower())) == 1 and not _is_roman(token):
                    if token_case == "upper":
                        stats.alpha_repeats_upper += 1
                    else:
                        stats.alpha_repeats_lower += 1
            for prev, curr in zip(items_sorted, items_sorted[1:]):
                prev_token = prev.token
                curr_token = curr.token
                if not (prev_token.isalpha() and curr_token.isalpha()):
                    continue
                prev_case = "upper" if prev_token.isupper() else "lower"
                curr_case = "upper" if curr_token.isupper() else "lower"
                if prev_case != curr_case:
                    continue
                prev_alpha = _alpha_index(prev_token)
                curr_alpha = _alpha_index(curr_token)
                if prev_alpha and curr_alpha and curr_alpha == prev_alpha + 1:
                    if curr_case == "upper":
                        stats.alpha_transitions_upper += 1
                    else:
                        stats.alpha_transitions_lower += 1
                prev_roman = _roman_to_int(prev_token)
                curr_roman = _roman_to_int(curr_token)
                if prev_roman and curr_roman and curr_roman == prev_roman + 1:
                    if curr_case == "upper":
                        stats.roman_transitions_upper += 1
                    else:
                        stats.roman_transitions_lower += 1
            stats_map[cluster_id] = stats
        return stats_map, clusters

    def _classify_marker_type(
        self,
        token: str,
        stats: ClusterStats,
        prev_token: Optional[str],
        next_token: Optional[str],
    ) -> str:
        if token.isdigit():
            return "numeric"
        token_case = "upper" if token.isupper() else "lower"
        token_lower = token.lower()
        roman_val = _roman_to_int(token_lower)
        alpha_val = _alpha_index(token_lower)
        if roman_val is None:
            return f"alpha_{token_case}"
        if token_case == "upper":
            roman_score = stats.roman_transitions_upper + stats.roman_tokens_upper
            alpha_score = stats.alpha_transitions_upper + stats.alpha_repeats_upper
        else:
            roman_score = stats.roman_transitions_lower + stats.roman_tokens_lower
            alpha_score = stats.alpha_transitions_lower + stats.alpha_repeats_lower
        if _alpha_sequence(prev_token, token) or _alpha_sequence(token, next_token):
            return f"alpha_{token_case}"
        if _roman_sequence(prev_token, token) or _roman_sequence(token, next_token):
            return f"roman_{token_case}"
        if alpha_score >= roman_score + 1:
            return f"alpha_{token_case}"
        if roman_score >= alpha_score + 1:
            return f"roman_{token_case}"
        if alpha_val:
            return f"alpha_{token_case}"
        return f"roman_{token_case}"

    def _apply_marker_types(
        self,
        candidates: list[MarkerCandidate],
        stats_map: dict[int, ClusterStats],
        cluster_map: dict[int, list[MarkerCandidate]],
    ) -> None:
        for cluster_id, items in cluster_map.items():
            items_sorted = sorted(items, key=lambda c: c.line_index)
            stats = stats_map.get(cluster_id, ClusterStats())
            for idx, candidate in enumerate(items_sorted):
                prev_token = items_sorted[idx - 1].token if idx > 0 else None
                next_token = (
                    items_sorted[idx + 1].token
                    if idx + 1 < len(items_sorted)
                    else None
                )
                candidate.marker_type = self._classify_marker_type(
                    candidate.token,
                    stats,
                    prev_token,
                    next_token,
                )

    def _refine_marker_type(
        self,
        marker_type: str,
        candidate: MarkerCandidate,
        stack: list[Node],
    ) -> str:
        if stack and abs(candidate.line.x0 - stack[-1].marker_indent) <= self.indent_same_threshold:
            prev_token = stack[-1].tokens[-1]
            prev_type = stack[-1].marker_type or ""
            if marker_type.startswith("roman") and prev_type.startswith("alpha"):
                if _alpha_sequence(prev_token, candidate.token):
                    case = "upper" if candidate.token.isupper() else "lower"
                    return f"alpha_{case}"
            if marker_type.startswith("alpha") and prev_type.startswith("roman"):
                if _roman_sequence(prev_token, candidate.token):
                    case = "upper" if candidate.token.isupper() else "lower"
                    return f"roman_{case}"
        if not marker_type.startswith("alpha"):
            return marker_type
        if len(candidate.token) != 1 or not _is_roman(candidate.token):
            return marker_type
        if len(stack) <= 1:
            return marker_type
        if (
            candidate.line.x0
            <= stack[-1].marker_indent + self.same_type_indent_threshold
        ):
            return marker_type
        case = "upper" if candidate.token.isupper() else "lower"
        return f"roman_{case}"

    def _last_index_for_type(
        self, stack: list[Node], marker_type: str
    ) -> Optional[int]:
        for idx in range(len(stack) - 1, -1, -1):
            if stack[idx].marker_type == marker_type:
                return idx
        return None

    def _is_first_token(self, marker_type: str, token: str) -> bool:
        if marker_type == "numeric":
            return token == "1"
        if marker_type.startswith("alpha"):
            return _alpha_index(token.lower()) == 1
        if marker_type.startswith("roman"):
            return _roman_to_int(token.lower()) == 1
        return False

    def _should_nest_same_type(
        self,
        marker_type: str,
        token: str,
        line_indent: float,
        same_type_node: Node,
    ) -> bool:
        if line_indent <= same_type_node.marker_indent + self.same_type_indent_threshold:
            return False
        return self._is_first_token(marker_type, token)

    def _adjust_stack_for_content(self, stack: list[Node], line_indent: float) -> None:
        while len(stack) > 1:
            node = stack[-1]
            if line_indent + self.indent_same_threshold < node.marker_indent:
                stack.pop()
                continue
            if node.content_indent_min is not None:
                if node.content_indent_min >= node.marker_indent + self.note_indent_threshold:
                    if line_indent <= node.marker_indent + self.indent_same_threshold:
                        stack.pop()
                        continue
            break

    def _adjust_stack_for_marker(self, stack: list[Node], line_indent: float) -> None:
        while (
            len(stack) > 1
            and line_indent + self.indent_same_threshold < stack[-1].marker_indent
        ):
            stack.pop()

    def _update_content_indent(self, node: Node, line_indent: float) -> None:
        if node.content_indent_min is None:
            node.content_indent_min = line_indent
        else:
            node.content_indent_min = min(node.content_indent_min, line_indent)

    def _scan_leading_markers(self, raw: str) -> tuple[list[MarkerSpan], int]:
        markers: list[MarkerSpan] = []
        cursor = 0
        while cursor < len(raw):
            snippet = raw[cursor:]
            match = LEADING_PAREN_RE.match(snippet)
            kind = "paren"
            if not match:
                match = LEADING_DOT_RE.match(snippet)
                kind = "dot"
            if not match:
                break
            token = match.group(1).strip()
            if not _is_structural_token(token):
                break
            start = cursor
            end = cursor + match.end()
            markers.append(MarkerSpan(token=token, kind=kind, start=start, end=end))
            if end <= cursor:
                break
            cursor = end
        return markers, cursor

    def _append_raw_line(self, stack: list[Node], raw: str) -> None:
        if not stack:
            return
        node = stack[-1]
        node.content_lines.append(raw)
        if raw.strip():
            self._update_content_indent(node, node.marker_indent)

    def _start_note(
        self,
        subchapter: str,
        token: str,
        line: TextLine,
        remainder_raw: str,
        line_index: int,
    ) -> tuple[Optional[Node], list[Node]]:
        if token not in self.note_numbers:
            return None, []
        root = self._get_or_create_node(
            subchapter,
            (token,),
            line.x0,
            "note",
            line_index,
        )
        self.seen_notes.add(token)
        stack = [root]
        if remainder_raw:
            markers, remainder_start = self._scan_leading_markers(remainder_raw)
            if markers:
                self._start_marker_chain(
                    subchapter,
                    stack,
                    remainder_raw,
                    markers,
                    remainder_start,
                    line_index,
                    line.x0 + self.note_indent_threshold,
                )
                return stack[-1], stack
            root.content_lines.append(remainder_raw)
            if remainder_raw.strip():
                self._update_content_indent(root, line.x0)
        return root, stack

    def _start_marker_chain(
        self,
        subchapter: str,
        stack: list[Node],
        raw: str,
        markers: list[MarkerSpan],
        remainder_start: int,
        order_index: int,
        base_indent: float,
    ) -> None:
        for idx, marker in enumerate(markers):
            if len(stack) - 1 >= self.max_depth:
                self._append_raw_line(stack, raw[marker.start :])
                return
            marker_indent = base_indent + idx * self.note_indent_threshold
            marker_type = self._classify_marker_type(
                marker.token,
                ClusterStats(),
                None,
                None,
            )
            tokens = stack[-1].tokens + (marker.token,)
            node = self._get_or_create_node(
                subchapter,
                tokens,
                marker_indent,
                marker_type,
                order_index,
            )
            stack.append(node)
        remainder = raw[remainder_start:]
        if remainder:
            stack[-1].content_lines.append(remainder)
            if remainder.strip():
                self._update_content_indent(stack[-1], stack[-1].marker_indent)

    def _start_marker(
        self,
        subchapter: str,
        stack: list[Node],
        candidate: MarkerCandidate,
    ) -> None:
        if not stack:
            return
        self._adjust_stack_for_marker(stack, candidate.line.x0)
        if (
            len(stack) - 1 >= self.max_depth
            and candidate.line.x0 > stack[-1].marker_indent + self.indent_same_threshold
        ):
            # Preserve deeper markers as raw content without altering the stack.
            self._append_raw_line(stack, candidate.line.text)
            return
        marker_type = candidate.marker_type or self._classify_marker_type(
            candidate.token,
            ClusterStats(),
            None,
            None,
        )
        marker_type = self._refine_marker_type(marker_type, candidate, stack)
        same_index = self._last_index_for_type(stack, marker_type)
        if same_index is not None:
            same_node = stack[same_index]
            if not self._should_nest_same_type(
                marker_type,
                candidate.token,
                candidate.line.x0,
                same_node,
            ):
                stack[:] = stack[:same_index]
        if len(stack) - 1 >= self.max_depth:
            self._append_raw_line(stack, candidate.line.text)
            return
        parent = stack[-1]
        tokens = parent.tokens + (candidate.token,)
        node = self._get_or_create_node(
            subchapter,
            tokens,
            candidate.line.x0,
            marker_type,
            candidate.line_index,
        )
        stack.append(node)
        if candidate.remainder_raw:
            markers, remainder_start = self._scan_leading_markers(candidate.remainder_raw)
            if markers:
                self._start_marker_chain(
                    subchapter,
                    stack,
                    candidate.remainder_raw,
                    markers,
                    remainder_start,
                    candidate.line_index,
                    candidate.line.x0 + self.note_indent_threshold,
                )
            else:
                node.content_lines.append(candidate.remainder_raw)
                if candidate.remainder_raw.strip():
                    self._update_content_indent(node, candidate.line.x0)

    def _append_content(self, stack: list[Node], line: TextLine) -> None:
        if not stack:
            return
        self._adjust_stack_for_content(stack, line.x0)
        node = stack[-1]
        node.content_lines.append(line.text)
        if line.text.strip():
            self._update_content_indent(node, line.x0)

    def _clean_node_prefix(self, node: Node) -> None:
        if not node.content_lines:
            return
        token = node.tokens[-1]
        raw_line = node.content_lines[0]
        patterns: list[re.Pattern] = []
        if token.isdigit():
            patterns.append(re.compile(rf"^\s*{re.escape(token)}\.\s*"))
            patterns.append(
                re.compile(rf"^\s*\(\s*{re.escape(token)}\s*\)\s*", re.IGNORECASE)
            )
        else:
            patterns.append(
                re.compile(rf"^\s*\(\s*{re.escape(token)}\s*\)\s*", re.IGNORECASE)
            )
        for pattern in patterns:
            match = pattern.match(raw_line)
            if not match:
                continue
            remainder = raw_line[match.end() :]
            if remainder.strip():
                node.content_lines[0] = remainder
            else:
                node.content_lines.pop(0)
            break

    def _trim_content_lines(self, node: Node) -> None:
        while node.content_lines and not node.content_lines[0].strip():
            node.content_lines.pop(0)
        while node.content_lines and not node.content_lines[-1].strip():
            node.content_lines.pop()

    def parse(self) -> list[Node]:
        subchapters = self._find_subchapters()
        hit_counts: dict[str, int] = {}
        for subchapter, sub_start, sub_end in subchapters:
            us_range = self._find_us_notes_range(sub_start, sub_end)
            if not us_range:
                continue
            range_start, range_end = us_range
            note_indent, allow_paren = self._find_note_start_indent(
                range_start,
                range_end,
            )
            if note_indent is None:
                continue
            seen = set()
            for idx in range(range_start, range_end):
                note_start = self._match_note_start(
                    self.lines[idx],
                    left_margin=note_indent,
                    allow_paren=allow_paren,
                )
                if note_start and note_start[0] in self.note_numbers:
                    seen.add(note_start[0])
            if seen:
                hit_counts[subchapter] = len(seen)

        eligible_subchapters: Optional[set[str]] = None
        if hit_counts:
            eligible_subchapters = set(hit_counts.keys())

        for subchapter, sub_start, sub_end in subchapters:
            if eligible_subchapters is not None and subchapter not in eligible_subchapters:
                continue
            us_range = self._find_us_notes_range(sub_start, sub_end)
            if not us_range:
                continue
            range_start, range_end = us_range
            note_indent, allow_paren = self._find_note_start_indent(
                range_start,
                range_end,
            )
            if note_indent is None:
                continue
            note_start_indices: dict[int, tuple[str, str]] = {}
            for idx in range(range_start, range_end):
                line = self.lines[idx]
                note_start = self._match_note_start(
                    line,
                    left_margin=note_indent,
                    allow_paren=allow_paren,
                )
                if note_start:
                    note_start_indices[idx] = note_start

            candidates = self._collect_marker_candidates(
                range_start,
                range_end,
                set(note_start_indices.keys()),
            )
            self._assign_clusters(candidates)
            stats_map, cluster_map = self._build_cluster_stats(candidates)
            self._apply_marker_types(candidates, stats_map, cluster_map)
            marker_map = {candidate.line_index: candidate for candidate in candidates}

            note_start_list = sorted(note_start_indices.keys())
            for idx_pos, idx in enumerate(note_start_list):
                token, remainder_raw = note_start_indices[idx]
                end_idx = (
                    note_start_list[idx_pos + 1]
                    if idx_pos + 1 < len(note_start_list)
                    else range_end
                )
                if token not in self.note_numbers:
                    continue
                stack: list[Node] = []
                _, stack = self._start_note(
                    subchapter,
                    token,
                    self.lines[idx],
                    remainder_raw,
                    idx,
                )

                for line_index in range(idx + 1, end_idx):
                    line = self.lines[line_index]
                    if _is_header(line.norm_text):
                        continue
                    if US_NOTES_RE.match(line.norm_text):
                        continue
                    if STATISTICAL_NOTES_RE.match(line.norm_text):
                        break
                    marker = marker_map.get(line_index)
                    if marker:
                        self._start_marker(subchapter, stack, marker)
                        continue
                    self._append_content(stack, line)

        for node in self.node_order:
            self._clean_node_prefix(node)
            self._trim_content_lines(node)

        return sorted(self.node_order, key=lambda n: (n.order_index, n.creation_order))


class PdfNoteAnalyzerShallow:
    @staticmethod
    def get_notes(
        pdf_url: Optional[str],
        pdf_path: Optional[str],
        note_list: Iterable[int],
    ) -> list[dict]:
        note_numbers = {str(n) for n in note_list}
        if not note_numbers:
            return []

        pdf_file: Optional[Path] = None
        temp_path: Optional[Path] = None
        if pdf_path:
            path = Path(pdf_path)
            if not path.is_absolute():
                raise ValueError("pdf_path must be an absolute path.")
            if not path.exists():
                raise FileNotFoundError(f"PDF not found: {path}")
            pdf_file = path
        elif pdf_url:
            temp_path = PdfNoteAnalyzerShallow._download_pdf(pdf_url)
            pdf_file = temp_path
        else:
            raise ValueError("pdf_url or pdf_path is required.")

        try:
            lines = _extract_lines(pdf_file, use_cropbox=True)
            parser = ShallowNoteParser(lines, note_numbers)
            nodes = parser.parse()
            if any(node.subchapter is None for node in nodes):
                raise RuntimeError("Subchapter not found for parsed notes.")
            missing = sorted(note_numbers - parser.seen_notes, key=int)
            for note in missing:
                LOGGER.warning("Note %s not found in PDF.", note)
            return [
                {
                    "chapter": CHAPTER_NUMBER,
                    "subchapter": node.subchapter,
                    "label": node.label,
                    "path": _format_path(node.tokens),
                    "parent_label": node.parent_label,
                    "content": "\n".join(node.content_lines),
                }
                for node in nodes
            ]
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    @staticmethod
    def _download_pdf(pdf_url: str) -> Path:
        response = requests.get(pdf_url, stream=True, timeout=60)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
            return Path(handle.name)
