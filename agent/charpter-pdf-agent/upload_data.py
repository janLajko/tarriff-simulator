import argparse
import re
import sys
import time
from pathlib import Path

from google import genai
from google.genai.errors import ClientError, ServerError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.file_search_store.file_search_utils import (
    DEFAULT_STORE_DISPLAY_NAME,
    get_or_create_store,
    wait_for_operation,
)

DOC_DIR = Path(__file__).resolve().parent / "charpter-data"
MAX_UPLOAD_RETRIES = 3
RETRY_DELAY_SECONDS = 2
MAX_FALLBACK_RETRIES = 3
UPLOAD_HTTP_TIMEOUT_MS = 600_000
IMPORT_HTTP_TIMEOUT_MS = 600_000
TXT_OUTPUT_DIR = Path(__file__).resolve().parent / "charpter-data-txt"
CHUNKING_CONFIG = {
    "white_space_config": {
        "max_tokens_per_chunk": 512,
        "max_overlap_tokens": 50,
    }
}
FALLBACK_FILE_NAMES = {
    "SubchapterIII_USNote_02.pdf",
    "SubchapterIII_USNote_20.pdf",
}


def get_files_to_upload() -> list[Path]:
    files = sorted(p for p in DOC_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    if not files:
        raise FileNotFoundError(f"No PDF files found in {DOC_DIR}")
    return files


def parse_filename(file_path: Path) -> tuple[str, str]:
    match = re.match(r"^Subchapter([IVXLCDM]+)_USNote_(\d+)\.pdf$", file_path.name)
    if not match:
        raise ValueError(f"Unexpected filename format: {file_path.name}")
    roman = match.group(1)
    note_number = int(match.group(2))
    charpter = f"Subchapter {roman}"
    note = f"note({note_number})"
    return charpter, note


def metadata_dict_to_list(meta: dict[str, str]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for key, value in meta.items():
        items.append({"key": key, "string_value": value})
    return items


def upload_via_files_import(
    client: genai.Client,
    store_name: str,
    file_path: Path,
    metadata: dict[str, str],
) -> None:
    uploaded = None
    for attempt in range(1, MAX_FALLBACK_RETRIES + 1):
        try:
            uploaded = client.files.upload(
                file=str(file_path),
                config={
                    "display_name": file_path.name,
                    "mime_type": "application/pdf",
                    "http_options": {"timeout": UPLOAD_HTTP_TIMEOUT_MS},
                },
            )
            break
        except ServerError as exc:
            if exc.code != 503 or attempt >= MAX_FALLBACK_RETRIES:
                raise
            print(
                f"Retry files.upload for {file_path.name} (attempt {attempt + 1})"
            )
            time.sleep(RETRY_DELAY_SECONDS * attempt)

    if uploaded is None:
        raise RuntimeError(f"Failed to upload {file_path.name} via files.upload")

    for attempt in range(1, MAX_FALLBACK_RETRIES + 1):
        try:
            operation = client.file_search_stores.import_file(
                file_search_store_name=store_name,
                file_name=uploaded.name,
                config={
                    "custom_metadata": metadata_dict_to_list(metadata),
                    "chunking_config": CHUNKING_CONFIG,
                    "http_options": {"timeout": IMPORT_HTTP_TIMEOUT_MS},
                },
            )
            wait_for_operation(client, operation, f"Import {file_path.name}")
            break
        except ServerError as exc:
            if exc.code != 503 or attempt >= MAX_FALLBACK_RETRIES:
                raise
            print(
                f"Retry import for {file_path.name} (attempt {attempt + 1})"
            )
            time.sleep(RETRY_DELAY_SECONDS * attempt)


def convert_pdfs_to_txt(output_dir: Path = TXT_OUTPUT_DIR) -> None:
    try:
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing PDF text extraction dependency. Install pypdf and pdfminer.six."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    laparams = LAParams(
        line_margin=0.2,
        char_margin=2.0,
        word_margin=0.1,
        boxes_flow=0.5,
    )

    for file_path in get_files_to_upload():
        reader = PdfReader(str(file_path), strict=False)
        page_count = len(reader.pages)
        parts: list[str] = []
        for page_index in range(page_count):
            page_text = extract_text(
                str(file_path),
                laparams=laparams,
                page_numbers=[page_index],
            ).strip()
            parts.append(f"--- Page {page_index + 1} ---\n{page_text}")
        text = "\n\n".join(parts).strip() + "\n"
        out_path = output_dir / f"{file_path.stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"Converted {file_path.name} -> {out_path}")


def clear_store_documents(client: genai.Client, store_name: str) -> None:
    docs = list(client.file_search_stores.documents.list(parent=store_name))
    if not docs:
        print(f"No documents to delete in {store_name}")
        return
    for doc in docs:
        result = client.file_search_stores.documents.delete(
            name=doc.name,
            config={"force": True},
        )
        if hasattr(result, "done"):
            wait_for_operation(client, result, f"Delete {doc.display_name}")
        else:
            print(f"Deleted {doc.display_name}")
    print(f"Cleared {len(docs)} documents from {store_name}")


def upload_all_files(store_display_name: str = DEFAULT_STORE_DISPLAY_NAME) -> None:
    client = genai.Client()
    store = get_or_create_store(client, store_display_name)
    clear_store_documents(client, store.name)

    for file_path in get_files_to_upload():
        charpter, note = parse_filename(file_path)
        metadata = {"charpter": charpter, "note": note}
        config = {
            "display_name": file_path.name,
            "custom_metadata": metadata_dict_to_list(metadata),
            "chunking_config": CHUNKING_CONFIG,
            "http_options": {"timeout": UPLOAD_HTTP_TIMEOUT_MS},
        }

        for attempt in range(1, MAX_UPLOAD_RETRIES + 1):
            try:
                operation = client.file_search_stores.upload_to_file_search_store(
                    file_search_store_name=store.name,
                    file=str(file_path),
                    config=config,
                )
                wait_for_operation(client, operation, f"Upload {file_path.name}")
                break
            except ClientError as exc:
                message = str(exc)
                if "Upload has already been terminated" in message:
                    if file_path.name in FALLBACK_FILE_NAMES:
                        print(
                            f"Fallback upload for {file_path.name} via files.upload"
                        )
                        upload_via_files_import(
                            client, store.name, file_path, metadata
                        )
                        break
                    if attempt >= MAX_UPLOAD_RETRIES:
                        raise
                    print(
                        f"Retry upload for {file_path.name} (attempt {attempt + 1})"
                    )
                    time.sleep(RETRY_DELAY_SECONDS * attempt)
                    continue
                raise

    print(f"Uploaded {store.display_name} files to {store.name}")


def list_uploaded_documents(store_display_name: str = DEFAULT_STORE_DISPLAY_NAME) -> None:
    client = genai.Client()
    store = get_or_create_store(client, store_display_name)
    pager = client.file_search_stores.documents.list(parent=store.name)
    print(f"Documents in {store.name}:")
    for doc in pager:
        metadata: dict[str, object] = {}
        for item in doc.custom_metadata or []:
            if item.string_value is not None:
                metadata[item.key] = item.string_value
            elif item.numeric_value is not None:
                metadata[item.key] = item.numeric_value
            elif item.string_list_value is not None:
                metadata[item.key] = item.string_list_value.values or []
        print(f"- {doc.display_name} ({doc.name})")
        print(f"  state={doc.state}, mime={doc.mime_type}, size={doc.size_bytes}")
        print(f"  metadata: {metadata}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload chapter PDFs and/or convert PDFs to text."
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert PDFs to TXT and exit unless --upload/--list is also set.",
    )
    parser.add_argument(
        "--convert-dir",
        default=str(TXT_OUTPUT_DIR),
        help="Output directory for converted TXT files.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload PDFs to the file search store.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List documents in the file search store.",
    )
    args = parser.parse_args()

    if args.convert:
        convert_pdfs_to_txt(Path(args.convert_dir))

    if args.upload or (not args.convert and not args.list):
        upload_all_files()

    if args.list or (not args.convert and not args.upload):
        list_uploaded_documents()


if __name__ == "__main__":
    main()
