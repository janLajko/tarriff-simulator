"""Shared helpers for Gemini File Search demos."""

import time
from typing import Optional

from google import genai
from google.genai import types


POLL_INTERVAL_SECONDS = 5
DEFAULT_STORE_DISPLAY_NAME = "tarriff-simulate"


def wait_for_operation(
    client: genai.Client, operation: types.Operation, label: str
) -> types.Operation:
    """Polls an LRO until completion, raising on error."""
    while not operation.done:
        print(f"{label}: processing...")
        time.sleep(POLL_INTERVAL_SECONDS)
        operation = client.operations.get(operation)

    if operation.error:
        raise RuntimeError(f"{label} failed: {operation.error}")

    print(f"{label}: done.")
    return operation


def find_store_by_display_name(
    client: genai.Client, display_name: str, *, page_size: int = 10
) -> Optional[types.FileSearchStore]:
    """Returns the first File Search store whose display_name matches."""
    for store in client.file_search_stores.list(
        config={"page_size": page_size}
    ):
        if store.display_name == display_name:
            return store
    return None


def get_or_create_store(
    client: genai.Client, display_name: str = DEFAULT_STORE_DISPLAY_NAME
) -> types.FileSearchStore:
    """Finds a store by display_name; creates one if none exists."""
    existing = find_store_by_display_name(client, display_name)
    if existing:
        print(f"Using existing File Search store: {existing.name}")
        return existing

    created = client.file_search_stores.create(
        config={"display_name": display_name}
    )
    print(f"Created File Search store: {created.name}")
    return created
