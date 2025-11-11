import argparse
import os
import re
import json
from typing import Dict, List, Optional, Sequence, Any
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import psycopg2
import psycopg2.extras as pgx


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

def db_connect(dsn):
    # 使用环境变量或在此处直接写配置
    # export PGHOST, PGUSER, PGPASSWORD, PGPORT, PGDATABASE
    return psycopg2.connect(dsn)

def get_note(conn, label, subchapter):
    """label 形如 'note(16)(a)(ii)' 或 'note 16 a ii' 都可"""
    norm = normalize_label(label)
    with conn, conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM hts_notes WHERE label=%s ORDER BY subchapter, id, path",
            (norm,),
        )
        row = cur.fetchone()
        if not row:
            return None
        # 把该节点及所有后代一次性取出
        cur.execute("""
            SELECT * FROM hts_notes
            WHERE chapter=%s AND subchapter=%s AND path[1:%s] = %s
            ORDER BY id, path
        """, (row["chapter"], row["subchapter"], len(row["path"]), row["path"]))
        return cur.fetchall()

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
    # db_init(conn)

    hit = get_note(conn, "note(16)(m)", "SUBCHAPTER III")
    if hit:
        print(f"found {len(hit)} rows for note(16)(a)")
        for r in hit[:]:
            print(r["label"], "", r["content"])
    else:
        print("no rows for note(16)(a)")

if __name__ == "__main__":
    main()