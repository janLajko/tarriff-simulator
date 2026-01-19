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


if __name__ == "__main__":
    upload_all_files()
    list_uploaded_documents()
