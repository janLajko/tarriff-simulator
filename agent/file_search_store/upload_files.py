import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

from google import genai

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.file_search_store.file_search_utils import (
    DEFAULT_STORE_DISPLAY_NAME,
    get_or_create_store,
    wait_for_operation,
)


DOC_DIR = Path("/Users/jan/Documents/7788/tariff-simulate-v2/agent/othercharpter-agent/pdf")
SECTION_CHAPTER_MAP_PATH = Path("section-charpter.json")


def load_section_chapter_map() -> dict[str, list[str]]:
    if not SECTION_CHAPTER_MAP_PATH.exists():
        raise FileNotFoundError(f"Missing {SECTION_CHAPTER_MAP_PATH}")
    data = json.loads(SECTION_CHAPTER_MAP_PATH.read_text())
    return {item["section"]: item["chapter"] for item in data}


def chapter_to_section(chapter: str, section_map: dict[str, list[str]]) -> Optional[str]:
    for section, chapters in section_map.items():
        if chapter in chapters:
            return section
    return None


def parse_chapter_from_hts(hts_number: str) -> Optional[str]:
    digits = re.findall(r"\d+", hts_number)
    if not digits:
        return None
    # Use first two digits for chapter (e.g., 02xx -> Chapter 2, 24xx -> Chapter 24)
    prefix = digits[0][:2]
    if not prefix:
        return None
    return f"Chapter {int(prefix)}"


def collect_csv_chapters(file_path: Path) -> set[str]:
    chapters: set[str] = set()
    with file_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # 兼容含 BOM 的列名
        hts_field = None
        for field in reader.fieldnames or []:
            normalized = field.lstrip("\ufeff").strip().lower()
            if normalized == "hts number":
                hts_field = field
                break
        if not hts_field:
            return chapters
        for row in reader:
            hts = row.get(hts_field) or ""
            chapter = parse_chapter_from_hts(hts)
            if chapter:
                chapters.add(chapter)
    return chapters


def build_pdf_metadata(file_path: Path, section_map: dict[str, list[str]]) -> dict:
    # Extract chapter number from filename like "Chapter 1_2025HTSRev31.pdf"
    match = re.search(r"Chapter\s*(\d+)", file_path.name, re.IGNORECASE)
    chapter = f"Chapter {match.group(1)}" if match else None
    section = chapter_to_section(chapter, section_map) if chapter else None
    meta: dict[str, object] = {"type": "notes"}
    if chapter:
        meta["chapter"] = chapter
    if section:
        meta["section"] = section
    return meta


def build_csv_metadata(chapters: Iterable[str], section_map: dict[str, list[str]]) -> dict:
    chapter_list = sorted(set(chapters))
    sections = sorted(
        {chapter_to_section(ch, section_map) for ch in chapter_list if chapter_to_section(ch, section_map)}
    )
    meta: dict[str, object] = {"type": "csv"}
    if chapter_list:
        meta["chapter"] = chapter_list if len(chapter_list) > 1 else chapter_list[0]
    if sections:
        meta["section"] = sections if len(sections) > 1 else sections[0]
    return meta


def metadata_dict_to_list(meta: dict[str, object]) -> list[dict[str, str]]:
    """Convert dict metadata to the list-of-key/value format required by SDK."""
    items: list[dict[str, str]] = []
    for key, value in meta.items():
        if isinstance(value, (list, tuple, set)):
            for v in value:
                items.append({"key": key, "string_value": str(v)})
        else:
            items.append({"key": key, "string_value": str(value)})
    return items


def get_files_to_upload() -> list[Path]:
    files = sorted(p for p in DOC_DIR.iterdir() if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files found in {DOC_DIR}")
    return files


def upload_all_files():
    client = genai.Client()
    store = get_or_create_store(client, DEFAULT_STORE_DISPLAY_NAME)
    # section_map = load_section_chapter_map()

    for file_path in get_files_to_upload():
        # if file_path.suffix.lower() == ".csv":
        #     chapters = collect_csv_chapters(file_path)
        #     metadata = build_csv_metadata(chapters, section_map)
        # else:
        #     metadata = build_pdf_metadata(file_path, section_map)

        config = {
            "display_name": file_path.name,
            # "custom_metadata": metadata_dict_to_list(metadata),
        }

        operation = client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store.name,
            file=str(file_path),
            config=config,
        )
        wait_for_operation(client, operation, f"Upload {file_path.name}")

    print(f"Uploaded {store.display_name} files to {store.name}")


def list_uploaded_documents(store_display_name: str = DEFAULT_STORE_DISPLAY_NAME):
    """列出指定 File Search Store 中的文档及其元数据。"""
    client = genai.Client()
    store = get_or_create_store(client, store_display_name)
    pager = client.file_search_stores.documents.list(parent=store.name)
    print(f"Documents in {store.name}:")
    for doc in pager:
        metadata = {}
        for item in doc.custom_metadata or []:
            if item.string_value is not None:
                metadata[item.key] = item.string_value
            elif item.numeric_value is not None:
                metadata[item.key] = item.numeric_value
            elif item.string_list_value is not None:
                metadata[item.key] = item.string_list_value.values or []
        print(f"- {doc.display_name} ({doc.name})")
        print(f"  state={doc.state}, mime={doc.mime_type}, size={doc.size_bytes}")
        if metadata:
            print(f"  metadata: {metadata}")
        else:
            print("  metadata: {}")


if __name__ == "__main__":
    upload_all_files()
    list_uploaded_documents()
