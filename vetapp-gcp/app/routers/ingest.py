"""
routers/ingest.py — Direct image submission endpoint.

POST /ingest
    Accepts a base64-encoded image plus optional metadata.
    Validates, deduplicates, and routes the image through the same
    dataset pipeline.

POST /ingest/upload
    Multipart alternative: send as a file upload instead of base64.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

router = APIRouter(tags=["Ingest"])

_ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_MAX_BYTES = 20 * 1024 * 1024  # 20 MB


class IngestRequest(BaseModel):
    """Submit an image as a base64 data URL."""
    image: str          # "data:image/jpeg;base64,..."
    caption: str = ""   # e.g. "dog bacterial" or "cat ringworm"
    user_id: str = ""   # optional submitter identifier
    user_name: str = "" # optional human-readable name
    timestamp: str = "" # optional ISO timestamp; defaults to now


class IngestResponse(BaseModel):
    status: str
    reason: str = ""
    gcs_path: str = ""
    pred_label: str = ""
    confidence: float = 0.0
    split: str = ""


def _b64_to_bytes(data_url: str) -> bytes:
    """Decode a data URL or raw base64 string to bytes."""
    if data_url.startswith("data:"):
        header, _, b64 = data_url.partition(",")
        mime = header.split(";")[0].split(":")[1] if ":" in header else ""
        if mime and mime not in _ALLOWED_MIME:
            raise ValueError(f"Unsupported image type: {mime}")
    else:
        b64 = data_url
    return base64.b64decode(b64)


@router.post("/ingest", response_model=IngestResponse, tags=["Ingest"])
def ingest_base64(req: IngestRequest):
    """
    Submit an image as a base64 data URL for pipeline processing.

    The **caption** field drives routing — include the species and disease name,
    e.g. `"dog bacterial"` or `"cat ringworm"`.  Without a recognisable caption
    the image lands in the *preclean* queue for manual vet labelling.
    """
    try:
        image_bytes = _b64_to_bytes(req.image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    if len(image_bytes) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 20 MB limit")

    return _run_pipeline(
        image_bytes,
        caption=req.caption,
        user_id=req.user_id or "api",
        user_name=req.user_name or "API User",
        timestamp=req.timestamp or datetime.utcnow().isoformat(),
    )


@router.post("/ingest/upload", response_model=IngestResponse, tags=["Ingest"])
async def ingest_upload(
    file: UploadFile = File(...),
    caption: str = Form(""),
    user_id: str = Form(""),
    user_name: str = Form(""),
    timestamp: str = Form(""),
):
    """
    Submit an image as a multipart file upload for pipeline processing.

    Equivalent to `/ingest` but accepts `multipart/form-data`.
    """
    if file.content_type and file.content_type not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}",
        )

    image_bytes = await file.read()
    if len(image_bytes) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 20 MB limit")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    return _run_pipeline(
        image_bytes,
        caption=caption,
        user_id=user_id or "api",
        user_name=user_name or "API User",
        timestamp=timestamp or datetime.utcnow().isoformat(),
    )


def _run_pipeline(
    image_bytes: bytes,
    caption: str,
    user_id: str,
    user_name: str,
    timestamp: str,
) -> IngestResponse:
    from app.pipeline import processor

    result = processor.process_incoming(
        image_bytes, caption, user_id, user_name, timestamp
    )

    return IngestResponse(
        status=result.get("status", "unknown"),
        reason=result.get("reason", ""),
        gcs_path=result.get("gcs_path", ""),
        pred_label=result.get("pred_label", result.get("label", "")),
        confidence=result.get("confidence", 0.0),
        split=result.get("split", ""),
    )
