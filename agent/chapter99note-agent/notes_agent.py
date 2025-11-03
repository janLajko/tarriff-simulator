#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HTS Chapter 99 Notes → Postgres
- 拉取: https://hts.usitc.gov/reststop/getChapterNotes?doc=99
- 解析层级: SUBCHAPTER → note N → (a)/(1)/(A)/(i)/(I)/(aa)/(III) … 任意深度
- 入库: PostgreSQL
- 查询: get_note("note(16)(a)"), get_note("note(17)(c)(2)")
"""
import argparse
import os
import re
import json
from typing import Dict, List, Optional, Sequence, Any
import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
from urllib.parse import urljoin
import psycopg2
import psycopg2.extras as pgx

API = "https://hts.usitc.gov/reststop/getChapterNotes?doc=99"

# ---------- DB ----------

DDL = """
CREATE TABLE IF NOT EXISTS hts_notes (
  id BIGSERIAL PRIMARY KEY,
  chapter INTEGER NOT NULL,
  subchapter TEXT,                         -- e.g. "SUBCHAPTER I"
  label TEXT NOT NULL,                     -- e.g. "note(16)(a)(ii)(A)"
  path TEXT[] NOT NULL,                    -- e.g. ['note','16','a','ii','A']
  parent_label TEXT,                       -- nullable
  content TEXT NOT NULL,                   -- plain text of this节点
  raw_html TEXT NOT NULL,                  -- 原始HTML
  UNIQUE (chapter, subchapter, label)
);

CREATE INDEX IF NOT EXISTS idx_hts_notes_path_gin ON hts_notes USING gin (path);
CREATE INDEX IF NOT EXISTS idx_hts_notes_label ON hts_notes(label);
"""

def db_connect(dsn):
    # 使用环境变量或在此处直接写配置
    # export PGHOST, PGUSER, PGPASSWORD, PGPORT, PGDATABASE
    return psycopg2.connect(dsn)

def db_init(conn):
    with conn, conn.cursor() as cur:
        cur.execute(DDL)
        # 兼容旧版唯一约束 (chapter,label)
        cur.execute("ALTER TABLE hts_notes DROP CONSTRAINT IF EXISTS hts_notes_chapter_label_key")
        cur.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM   pg_constraint
                    WHERE  conrelid = 'hts_notes'::regclass
                      AND  conname = 'hts_notes_chapter_subchapter_label_key'
                ) THEN
                    ALTER TABLE hts_notes
                    ADD CONSTRAINT hts_notes_chapter_subchapter_label_key
                    UNIQUE (chapter, subchapter, label);
                END IF;
            END$$;
            """
        )

def upsert_notes(conn, notes):
    sql = """
    INSERT INTO hts_notes(chapter, subchapter, label, path, parent_label, content, raw_html)
    VALUES (%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (chapter, subchapter, label)
    DO UPDATE SET subchapter=EXCLUDED.subchapter,
                  path=EXCLUDED.path,
                  parent_label=EXCLUDED.parent_label,
                  content=EXCLUDED.content,
                  raw_html=EXCLUDED.raw_html;
    """
    with conn, conn.cursor() as cur:
        pgx.execute_batch(cur, sql, [
            (n["chapter"], n.get("subchapter"), n["label"], n["path"],
             n.get("parent_label"), n["content"], n["raw_html"])
            for n in notes
        ])

def get_note(conn, label, subchapter: Optional[str] = None):
    """label 形如 'note(16)(a)(ii)' 或 'note 16 a ii' 都可"""
    norm = normalize_label(label)
    with conn, conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
        if subchapter:
            cur.execute(
                "SELECT * FROM hts_notes WHERE label=%s AND subchapter=%s ORDER BY array_length(path,1), path",
                (norm, subchapter),
            )
        else:
            cur.execute(
                "SELECT * FROM hts_notes WHERE label=%s ORDER BY subchapter, array_length(path,1), path",
                (norm,),
            )
        row = cur.fetchone()
        if not row:
            return None
        # 把该节点及所有后代一次性取出
        cur.execute("""
            SELECT * FROM hts_notes
            WHERE chapter=%s AND subchapter=%s AND path[1:%s] = %s
            ORDER BY array_length(path,1), path
        """, (row["chapter"], row["subchapter"], len(row["path"]), row["path"]))
        return cur.fetchall()

# ---------- Fetch ----------

def fetch_chapter_notes():
    r = requests.get(API, timeout=60)
    r.raise_for_status()
    content_type = r.headers.get("content-type", "").lower()

    # 最新接口直接返回 HTML（text/plain），为兼容旧格式保留 JSON 路径
    if "application/json" in content_type or r.text.strip().startswith("{"):
        try:
            j = r.json()
        except ValueError as exc:
            raise RuntimeError("chapter notes 接口返回 JSON 解析失败") from exc
        html = j.get("html") or j.get("data") or j.get("chapterNotes") or ""
        if isinstance(html, dict):
            html = html.get("notes", "")
    else:
        html = r.text

    html = html.strip()
    if not html:
        raise RuntimeError("chapter notes 接口返回为空或结构改变")
    return html

# ---------- Parse ----------

SUBCHAPTER_RE = re.compile(r'^\s*SUBCHAPTER\b', re.I)
US_NOTES_RE  = re.compile(r'^\s*(U\.S\.\s*)?NOTES?\b', re.I)
TOP_NOTE_HEAD_RE = re.compile(r'^\s*note\s+(\d+)', re.I)  # 少量站点会把“Note 1.”单独成段
PAREN_TOKEN_RE = re.compile(r'^\(\s*([^)]+?)\s*\)\s*')     # 捕获 "(a) " | "(1) " 等

def normalize_label(label: str) -> str:
    # 统一成 note(16)(a)(ii)...
    s = label.strip()
    if not s.lower().startswith("note"):
        s = "note " + s
    # 把空格分隔转为括号
    if "(" not in s:
        parts = s.split()
        head = parts[0].lower()  # note
        tail = "".join(f"({p.strip()})" for p in parts[1:])
        s = head + tail
    # 规范大小写与去空白
    # 把 "Note(16)(A)" → "note(16)(A)"
    s = "note" + "".join(f"({t})" for t in re.findall(r'\(\s*([^)]+?)\s*\)', s))
    return s

def strip_self_text(li_tag):
    """取 li 的自身文本，不含子ul/ol"""
    li = li_tag.__copy__()
    for ul in li.find_all(["ul", "ol"]):
        ul.decompose()
    return li.get_text(" ", strip=True)


def absorb_stray_list_content(list_tag):
    """把列表中游离的节点（如 table、文本）并入前一个 <li>."""
    last_li: Optional[Tag] = None
    for child in list(list_tag.children):
        if isinstance(child, NavigableString):
            if not child.strip() or last_li is None:
                continue
            last_li.append(child.extract())
            continue
        if not isinstance(child, Tag):
            continue
        if child.name == "li":
            last_li = child
            continue
        if last_li is None:
            continue
        last_li.append(child.extract())


def tokens_from_value(value: Optional[str]) -> Optional[List[tuple[str, str]]]:
    if not value:
        return None
    raw = value.strip()
    if not raw or raw.lower() == "null":
        return None
    paren_groups = re.findall(r"\(\s*([^)]+?)\s*\)", raw)
    if paren_groups:
        tokens = [(token.strip(), "paren") for token in paren_groups if token.strip()]
        return tokens or None
    compact = re.sub(r"\s+", "", raw)
    if compact.endswith("."):
        compact = compact[:-1]
    if compact:
        return [(compact, "plain")]
    return None


def tokens_from_text(text_self: str) -> Optional[List[tuple[str, str]]]:
    if not text_self:
        return None
    m = re.match(r"^\s*(\d+)\.?\s*", text_self)
    if m:
        return [(m.group(1), "plain")]
    mt = PAREN_TOKEN_RE.match(text_self)
    if mt:
        return [(mt.group(1).strip(), "paren")]
    return None


def build_label_from_tokens(tokens: Sequence[str]) -> str:
    return "note" + "".join(f"({t})" for t in tokens)


def strip_tokens_from_content(text_self: str, tokens: Sequence[tuple[str, str]]) -> str:
    remaining = text_self
    for token, kind in tokens:
        if kind == "plain":
            pattern = re.compile(rf"^\s*{re.escape(token)}\.?\s*", re.IGNORECASE)
        else:
            pattern = re.compile(rf"^\s*\(\s*{re.escape(token)}\s*\)\s*", re.IGNORECASE)
        remaining, count = pattern.subn("", remaining, count=1)
        if not count:
            break
    stripped = remaining.strip()
    return stripped if stripped else text_self.strip()

def parse_html_to_notes(chapter_html: str, chapter=99):
    soup = BeautifulSoup(chapter_html, "html.parser")

    # 1) 定位各个 SUBCHAPTER 区块
    subchapters = []
    for t in soup.find_all("div", class_=lambda c: c and "misc_title" in c.split()):
        if SUBCHAPTER_RE.search(t.get_text(strip=True)):
            subchapters.append(t)

    # 2) 每个子章内，找到“U.S. Notes”段落后的第一个列表作为顶层 notes
    results = []
    for idx, head in enumerate(subchapters):
        sub_name = head.get_text(" ", strip=True)
        # 取该 subchapter 范围：head 到下一个 subchapter 之间
        seg = []
        node = head.next_sibling
        while node and not (getattr(node, "get_text", None) and isinstance(node, type(head)) and
                            node.name == head.name and SUBCHAPTER_RE.search(node.get_text(" ", strip=True))):
            seg.append(node)
            node = node.next_sibling
        segment = BeautifulSoup("".join(str(x) for x in seg if x), "html.parser")

        # 找 “U.S. Notes” 标识
        notes_anchor = None
        for d in segment.find_all("div", class_=lambda c: c and "misc_title" in c.split()):
            if US_NOTES_RE.search(d.get_text(" ", strip=True)):
                notes_anchor = d
                break
        if notes_anchor:
            top_lists: List[Any] = []
            walker = notes_anchor
            while True:
                walker = walker.find_next(["ol", "ul"])
                if walker is None:
                    break
                if walker.parent and walker.parent.name == "li":
                    continue
                top_lists.append(walker)
        else:
            top_lists = segment.find_all(["ol", "ul"], recursive=False)

        if not top_lists:
            # 没有 notes 列表，跳过
            continue

        # 3) 递归解析列表
        def rec_list(list_tag, subchapter_name, path_tokens: List[str]):
            parent_label = build_label_from_tokens(path_tokens) if path_tokens else None
            absorb_stray_list_content(list_tag)
            for li in list_tag.find_all("li", recursive=False):
                text_self = strip_self_text(li)
                token_entries = tokens_from_value(li.get("value"))
                if not token_entries:
                    token_entries = tokens_from_text(text_self)
                if not token_entries:
                    token_entries = [(f"text-{abs(hash(text_self))%10**8}", "plain")]

                new_tokens = path_tokens + [token for token, _ in token_entries]
                label = build_label_from_tokens(new_tokens)
                content = strip_tokens_from_content(text_self, token_entries)

                results.append({
                    "chapter": chapter,
                    "subchapter": subchapter_name,
                    "label": label,
                    "path": ["note"] + new_tokens,
                    "parent_label": parent_label,
                    "content": content,
                    "raw_html": str(li)
                })

                sub = li.find(["ul", "ol"], recursive=False)
                if sub:
                    rec_list(sub, subchapter_name, new_tokens)

        for root_list in top_lists:
            rec_list(root_list, sub_name, [])

    # 清理与规范 path
    cleaned = []
    for n in results:
        n["label"] = normalize_label(n["label"])
        n["path"]  = normalize_path_from_label(n["label"])
        cleaned.append(n)
    return merge_dedup(cleaned)

def normalize_path_from_label(label:str):
    lab = normalize_label(label)
    parts = re.findall(r'\(\s*([^)]+?)\s*\)', lab)
    return ["note"] + parts

def merge_dedup(items):
    # 去重：以 (chapter, subchapter, label) 为键，保留最长内容的记录
    d = {}
    for x in items:
        k = (x["chapter"], x["subchapter"], x["label"])
        if k not in d or len(x["raw_html"]) > len(d[k]["raw_html"]):
            d[k] = x
    return list(d.values())

# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import HTS CSV data into PostgreSQL.")
    parser.add_argument(
        "--dsn",
        default=os.getenv("DATABASE_DSN"),
        help="PostgreSQL DSN (e.g. postgresql://user:pass@localhost:5432/dbname). "
        "Defaults to DATABASE_DSN environment variable.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    if not args.dsn:
        raise SystemExit("PostgreSQL DSN required via --dsn or DATABASE_DSN.")

    conn = db_connect(args.dsn)
    db_init(conn)
    html = fetch_chapter_notes()
    notes = parse_html_to_notes(html, chapter=99)
    upsert_notes(conn, notes)

    # 示例查询
    hit = get_note(conn, "note(16)(a)")
    if hit:
        print(f"found {len(hit)} rows for note(16)(a)")
        for r in hit[:5]:
            print(r["label"], "=>", r["content"][:80].replace("\n", " "))
    else:
        print("no rows for note(16)(a)")

if __name__ == "__main__":
    main()
