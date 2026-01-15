import os
import re
import uuid
from dataclasses import dataclass
from typing import Optional

from google.cloud import storage


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename(name: str) -> str:
    name = (name or "").strip()
    name = os.path.basename(name)
    if not name:
        return "file"
    name = _SAFE_NAME_RE.sub("_", name)
    return name[:180] if len(name) > 180 else name


def _join_prefix(prefix: str) -> str:
    prefix = (prefix or "").strip()
    if not prefix:
        return ""
    return prefix if prefix.endswith("/") else prefix + "/"


@dataclass(frozen=True)
class UploadedObject:
    bucket: str
    object_name: str
    content_type: str
    size_bytes: int

    @property
    def gcs_uri(self) -> str:
        return f"gs://{self.bucket}/{self.object_name}"

    @property
    def public_url(self) -> str:
        return f"https://storage.googleapis.com/{self.bucket}/{self.object_name}"


class GcsUploadService:
    """
    Uploads user attachments to Google Cloud Storage.

    Auth:
    - Uses Application Default Credentials (Cloud Run default service account).
    - No explicit credential handling here by design.
    """

    def __init__(self, *, bucket: Optional[str] = None, prefix: Optional[str] = None) -> None:
        self.bucket = bucket or os.getenv("GCS_BUCKET", "aitryon-images")
        self.prefix = prefix if prefix is not None else os.getenv("GCS_PREFIX", "classification/")
        self.client = storage.Client()

    def upload_fileobj(
        self,
        *,
        fileobj,
        filename: str,
        content_type: str,
        case_id: Optional[str] = None,
    ) -> UploadedObject:
        safe_name = _sanitize_filename(filename)
        prefix = _join_prefix(self.prefix)
        if case_id:
            prefix = _join_prefix(prefix + str(case_id).strip())

        object_name = f"{prefix}{uuid.uuid4().hex}_{safe_name}"

        bucket = self.client.bucket(self.bucket)
        blob = bucket.blob(object_name)

        # Try to compute size without loading into memory.
        size_bytes = 0
        try:
            pos = fileobj.tell()
            fileobj.seek(0, os.SEEK_END)
            size_bytes = int(fileobj.tell())
            fileobj.seek(pos, os.SEEK_SET)
        except Exception:
            size_bytes = 0

        blob.upload_from_file(fileobj, content_type=content_type, rewind=True)

        return UploadedObject(
            bucket=self.bucket,
            object_name=object_name,
            content_type=content_type or "application/octet-stream",
            size_bytes=size_bytes,
        )

