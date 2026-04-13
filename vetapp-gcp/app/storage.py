"""
storage.py — Google Cloud Storage helpers.

All permanent files (model weights, incoming images, dataset, preclean,
vet_queue) live in the GCS bucket defined by GCS_BUCKET.
The /tmp/vetapp scratch space is used only for ephemeral processing
within a single request; Cloud Run's /tmp is wiped between instances.

GCS path conventions:
  models/best_model.pth
  models/dog_fold1.keras  …  dog_fold5.keras
  models/cat_fold1.keras  …  cat_fold5.keras
  incoming/<filename>
  preclean/<filename>
  vet_queue/<filename>
  dataset/<species>/<split>/<label>/<filename>
  rejected/<filename>
"""

from __future__ import annotations

import datetime
import os
from typing import Optional

from google.cloud import storage as gcs_lib

from app.config import GCS_BUCKET, TMP_DIR

_client: Optional[gcs_lib.Client] = None


def get_client() -> gcs_lib.Client:
    global _client
    if _client is None:
        _client = gcs_lib.Client()
    return _client


# ── Upload ─────────────────────────────────────────────────────────────────────

def upload_bytes(data: bytes, gcs_path: str,
                 content_type: str = "image/jpeg") -> str:
    """Upload raw bytes and return the GCS path."""
    bucket = get_client().bucket(GCS_BUCKET)
    bucket.blob(gcs_path).upload_from_string(data, content_type=content_type)
    return gcs_path


def upload_file(local_path: str, gcs_path: str,
                content_type: str = "image/jpeg") -> str:
    """Upload a local file and return the GCS path."""
    bucket = get_client().bucket(GCS_BUCKET)
    bucket.blob(gcs_path).upload_from_filename(local_path,
                                                content_type=content_type)
    return gcs_path


# ── Download ───────────────────────────────────────────────────────────────────

def download_bytes(gcs_path: str) -> bytes:
    """Download a GCS object as raw bytes."""
    return get_client().bucket(GCS_BUCKET).blob(gcs_path).download_as_bytes()


def download_to_tmp(gcs_path: str, subdir: str = "") -> str:
    """Download a GCS object to /tmp/vetapp[/subdir] and return the local path."""
    local_dir = os.path.join(TMP_DIR, subdir) if subdir else TMP_DIR
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, os.path.basename(gcs_path))
    get_client().bucket(GCS_BUCKET).blob(gcs_path).download_to_filename(local_path)
    return local_path


# ── Signed URL (for serving images to the dashboard) ──────────────────────────

def signed_url(gcs_path: str, expiry_minutes: int = 30) -> str:
    """Return a V4 signed URL valid for *expiry_minutes* minutes."""
    blob = get_client().bucket(GCS_BUCKET).blob(gcs_path)
    return blob.generate_signed_url(
        expiration=datetime.timedelta(minutes=expiry_minutes),
        method="GET",
        version="v4",
    )


# ── List / count ───────────────────────────────────────────────────────────────

def list_blobs(prefix: str) -> list[str]:
    """Return all GCS object names under *prefix*."""
    return [
        b.name
        for b in get_client().bucket(GCS_BUCKET).list_blobs(prefix=prefix)
    ]


def count_images(prefix: str) -> int:
    """Count objects under *prefix* that look like images."""
    return sum(
        1 for f in list_blobs(prefix)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


# ── Delete ─────────────────────────────────────────────────────────────────────

def delete_blob(gcs_path: str) -> None:
    blob = get_client().bucket(GCS_BUCKET).blob(gcs_path)
    if blob.exists():
        blob.delete()


def move_blob(src: str, dst: str) -> None:
    """Copy src → dst then delete src (GCS has no native move)."""
    bucket = get_client().bucket(GCS_BUCKET)
    bucket.copy_blob(bucket.blob(src), bucket, dst)
    bucket.blob(src).delete()
