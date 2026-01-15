"""Feedback API routes and handlers."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import psycopg2
from psycopg2.extras import Json
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from .gcs_upload_service import GcsUploadService

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

DATABASE_DSN = (
    os.getenv("DATABASE_DSN")
    or os.getenv("POSTGRES_DSN")
    or os.getenv("PG_DSN")
)

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


@contextmanager
def db_connection():
    if not DATABASE_DSN:
        raise RuntimeError(
            "DATABASE_DSN (or POSTGRES_DSN/PG_DSN) environment variable is not configured."
        )
    conn = psycopg2.connect(DATABASE_DSN)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class ApiResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    code: str
    message: str
    data: Optional[Dict[str, Any]] = None


class TariffSimulatorFeedbackRequest(BaseModel):
    email: str = Field(..., min_length=3)
    company_name: str = Field(..., min_length=1)
    comment: str = Field(..., min_length=1)
    attachments: Optional[list[str]] = None
    context: Dict[str, Any]


class TariffSimulatorAgreeRequest(BaseModel):
    agree: bool
    context: Dict[str, Any]
    comment: Optional[str] = None

def _get_upload_size(file: UploadFile) -> int:
    try:
        current_pos = file.file.tell()
        file.file.seek(0, os.SEEK_END)
        size = int(file.file.tell())
        file.file.seek(current_pos, os.SEEK_SET)
        return size
    except Exception:
        return 0


def _execute_insert(sql: str, params: tuple[Any, ...]) -> None:
    try:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
    except Exception as exc:
        logger.exception("Failed to insert feedback record.")
        raise HTTPException(status_code=500, detail="SERVER_ERROR") from exc


@router.post("/attachments/upload", response_model=ApiResponse)
def upload_attachment(file: UploadFile = File(...)) -> ApiResponse:
    size_bytes = _get_upload_size(file)
    if size_bytes and size_bytes > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 10MB).")

    prefix = datetime.now(timezone.utc).strftime("feedback/%Y/%m/")
    uploader = GcsUploadService(prefix=prefix)
    uploaded = uploader.upload_fileobj(
        fileobj=file.file,
        filename=file.filename or "file",
        content_type=file.content_type or "application/octet-stream",
    )

    return ApiResponse(
        code="OK",
        message="success",
        data={
            "object_url": uploaded.public_url,
            "gcs_uri": uploaded.gcs_uri,
            "object_name": uploaded.object_name,
            "content_type": uploaded.content_type,
            "size_bytes": uploaded.size_bytes,
        },
    )


@router.post("/tariff-simulator/feedback", response_model=ApiResponse)
def create_tariff_simulator_feedback(payload: TariffSimulatorFeedbackRequest) -> ApiResponse:
    sql = """
        INSERT INTO feedback_tariff_simulator
        (email, company_name, comment, attachments, context)
        VALUES (%s, %s, %s, %s, %s)
    """
    _execute_insert(
        sql,
        (
            payload.email,
            payload.company_name,
            payload.comment,
            Json(payload.attachments) if payload.attachments is not None else None,
            Json(payload.context),
        ),
    )
    return ApiResponse(code="OK", message="success")


@router.post("/tariff-simulator/agree", response_model=ApiResponse)
def create_tariff_simulator_agree(payload: TariffSimulatorAgreeRequest) -> ApiResponse:
    sql = """
        INSERT INTO feedback_tariff_simulator_agree
        (agree, comment, context)
        VALUES (%s, %s, %s)
    """
    _execute_insert(
        sql,
        (
            payload.agree,
            payload.comment or "",
            Json(payload.context),
        ),
    )
    return ApiResponse(code="OK", message="success")


def register_feedback_routes(app) -> None:
    app.include_router(router)
