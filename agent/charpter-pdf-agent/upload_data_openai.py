import argparse
import re
import time
from pathlib import Path
from typing import Iterable, Optional

try:
    from openai import OpenAI
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Missing dependency: openai. Install it with `pip install openai`."
    ) from exc


DOC_DIR = Path(__file__).resolve().parent / "charpter-data"
VECTOR_STORE_NAME = "tarriff-simulate"
POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 900


def get_files_to_upload() -> list[Path]:
    files = sorted(p for p in DOC_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    if not files:
        raise FileNotFoundError(f"No PDF files found in {DOC_DIR}")
    return files


def resolve_files_to_upload(file_arg: Optional[str]) -> list[Path]:
    if not file_arg:
        return get_files_to_upload()
    candidate = Path(file_arg)
    if not candidate.is_absolute():
        doc_candidate = DOC_DIR / candidate
        if doc_candidate.exists():
            candidate = doc_candidate
    if not candidate.exists():
        raise FileNotFoundError(f"PDF file not found: {file_arg}")
    if candidate.is_dir():
        raise IsADirectoryError(f"Expected a PDF file but found directory: {candidate}")
    # if candidate.suffix.lower() != ".pdf":
    #     raise ValueError(f"Expected a .pdf file but found: {candidate.name}")
    return [candidate]


def parse_filename(file_path: Path) -> tuple[str, str]:
    # match = re.match(r"^Subchapter([IVXLCDM]+)_USNote_(\d+)\$", file_path.name)
    # if not match:
    #     raise ValueError(f"Unexpected filename format: {file_path.name}")
    # roman = match.group(1)
    # note_number = int(match.group(2))
    # charpter = f"Subchapter {roman}"
    # note = f"note({note_number})"
    return "Subchapter III", "note(20)"


def _iter_items(paged: object) -> Iterable[object]:
    data = getattr(paged, "data", None)
    if data is None:
        return paged
    return data


def get_or_create_vector_store(client: OpenAI, name: str) -> object:
    stores = client.vector_stores.list()
    for store in _iter_items(stores):
        if getattr(store, "name", None) == name:
            print(f"Using existing vector store: {store.id}")
            return store
    created = client.vector_stores.create(name=name)
    print(f"Created vector store: {created.id}")
    return created


def upload_file(client: OpenAI, file_path: Path) -> object:
    with file_path.open("rb") as file_content:
        return client.files.create(
            file=file_content,
            purpose="assistants",
        )


def add_file_to_vector_store(client: OpenAI, store_id: str, file_id: str) -> object:
    return client.vector_stores.files.create(
        vector_store_id=store_id,
        file_id=file_id,
    )


def wait_for_vector_store_file(
    client: OpenAI,
    store_id: str,
    file_id: str,
    *,
    timeout_seconds: int = POLL_TIMEOUT_SECONDS,
) -> object:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        result = client.vector_stores.files.list(vector_store_id=store_id)
        for item in _iter_items(result):
            if getattr(item, "id", None) != file_id:
                continue
            status = getattr(item, "status", None)
            if status in {"completed", "failed", "cancelled"}:
                return item
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Timed out waiting for vector store file {file_id}")


def upload_all_files(
    *,
    vector_store_name: str = VECTOR_STORE_NAME,
    file_arg: Optional[str] = None,
) -> None:
    client = OpenAI()
    store = get_or_create_vector_store(client, vector_store_name)

    for file_path in resolve_files_to_upload(file_arg):
        charpter, note = parse_filename(file_path)
        print(f"Uploading {file_path.name} ({charpter}, {note})")
        file_obj = upload_file(client, file_path)
        add_file_to_vector_store(client, store.id, file_obj.id)
        status = wait_for_vector_store_file(client, store.id, file_obj.id)
        print(f"Indexed {file_path.name}: status={getattr(status, 'status', None)}")


def list_vector_store_files(vector_store_name: str = VECTOR_STORE_NAME) -> None:
    client = OpenAI()
    store = get_or_create_vector_store(client, vector_store_name)
    result = client.vector_stores.files.list(vector_store_id=store.id)
    print(f"Files in vector store {store.id}:")
    for item in _iter_items(result):
        print(
            f"- {getattr(item, 'id', None)} status={getattr(item, 'status', None)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload chapter PDFs to an OpenAI vector store.")
    parser.add_argument(
        "--vector-store-name",
        default=VECTOR_STORE_NAME,
        help="Vector store name (default: %(default)s).",
    )
    parser.add_argument(
        "--file",
        dest="file_arg",
        help="Specific PDF file to upload (path or filename under charpter-data).",
    )
    args = parser.parse_args()

    upload_all_files(vector_store_name=args.vector_store_name, file_arg=args.file_arg)
    list_vector_store_files(vector_store_name=args.vector_store_name)
