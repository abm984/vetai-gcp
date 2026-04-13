"""
pipeline/processor.py — Validate, deduplicate, and route incoming images.

Ported from Arch/pipeline.py with three key adaptations:
  1. Images are stored in GCS instead of local directories.
  2. State (hashes, queues, logs) lives in PostgreSQL instead of SQLite.
  3. Input is raw bytes (from WhatsApp download) instead of local file paths.

Routing logic mirrors the original:
  ┌─ Caption has species + label → run ensemble
  │     ├─ conf ≥ threshold  → dataset/{species}/{split}/{label}/
  │     └─ conf < threshold  → vet_queue/
  └─ Caption missing / unrecognisable → preclean/
"""

from __future__ import annotations

import hashlib
import io
import re
from datetime import datetime
from difflib import get_close_matches
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import (
    ALIASES, CLASSES, SPECIES_TRIGGERS,
    BLUR_THRESHOLD, MIN_FILE_KB, SPLITS, SPLIT_RATIO,
    VET_CONFIDENCE_THRESHOLD, IMG_SIZE,
)
from app import database as db
from app import storage as gcs


# ── Utility helpers ────────────────────────────────────────────────────────────

def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _open_image(data: bytes) -> Optional[Image.Image]:
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return ImageOps.exif_transpose(img)
    except Exception:
        return None


def _blur_score(pil_img: Image.Image) -> float:
    gray = np.array(pil_img.convert("L"))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _normalize(pil_img: Image.Image) -> Image.Image:
    return pil_img.resize(IMG_SIZE, Image.LANCZOS)


def _pil_to_jpeg_bytes(pil_img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ── Label extraction ───────────────────────────────────────────────────────────

def _extract_species_and_label(
    caption: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    if not caption:
        return None, None
    text = re.sub(r"[^a-z0-9 _]", " ", caption.strip().lower())

    species = None
    for sp, triggers in SPECIES_TRIGGERS.items():
        if any(t in text for t in triggers):
            species = sp
            break
    if not species:
        return None, None

    # Remove species trigger words before matching label
    for t in SPECIES_TRIGGERS[species]:
        text = text.replace(t, " ")
    text = text.strip()

    label = _match_label(text, species)
    return species, label


def _match_label(text: str, species: str) -> Optional[str]:
    aliases = ALIASES[species]
    cls_list = CLASSES[species]
    for alias, cls in aliases.items():
        if alias in text:
            return cls
    candidates = cls_list + list(aliases.keys())
    for token in text.split():
        matches = get_close_matches(token, candidates, n=1, cutoff=0.82)
        if matches:
            hit = matches[0]
            return aliases.get(hit, hit if hit in cls_list else None)
    return None


# ── Split selection ────────────────────────────────────────────────────────────

def _pick_split(species: str, label: str) -> str:
    """Choose train / valid / test to maintain the target SPLIT_RATIO."""
    counts = db.get_split_counts(species, label)
    total = sum(counts.values()) + 1
    ratios = dict(zip(SPLITS, SPLIT_RATIO))
    return min(SPLITS, key=lambda s: counts.get(s, 0) / total - ratios[s])


def _unique_filename(species: str, label: str, sender_id: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    safe = re.sub(r"\W", "_", sender_id)[:12]
    return f"{species}_{label}_{safe}_{ts}.jpg"


# ── Main entry point ───────────────────────────────────────────────────────────

def process_incoming(
    image_bytes: bytes,
    caption: str,
    sender_id: str,
    sender_name: str,
    timestamp: str,
) -> dict:
    """
    Validate, deduplicate, and route raw image bytes.

    Returns a status dict:
        {"status": "accepted"|"vet_queue"|"preclean"|"rejected", "reason": "...", ...}
    """
    now = datetime.utcnow().isoformat()

    # 1. Minimum size check
    if len(image_bytes) < MIN_FILE_KB * 1024:
        return _finish("rejected", "too_small", sender_id, sender_name, timestamp, now)

    # 2. Duplicate check (MD5)
    h = _md5(image_bytes)
    if db.is_duplicate(h):
        return _finish("rejected", "duplicate", sender_id, sender_name, timestamp, now)
    db.record_hash(h, now)

    # 3. Decode image
    pil_img = _open_image(image_bytes)
    if pil_img is None:
        return _finish("rejected", "image_unreadable",
                       sender_id, sender_name, timestamp, now)

    # 4. Blur check
    blur = _blur_score(pil_img)
    if blur < BLUR_THRESHOLD:
        return _finish("rejected", f"too_blurry(score={blur:.1f})",
                       sender_id, sender_name, timestamp, now)

    # 5. Normalise image (used for all downstream paths)
    clean_img = _normalize(pil_img)
    clean_bytes = _pil_to_jpeg_bytes(clean_img)

    # 6. Extract species + label from caption
    species, label = _extract_species_and_label(caption)

    # ── PATH A: no label → preclean ────────────────────────────────────────────
    if not label:
        filename = _unique_filename(species or "unknown", "preclean", sender_id)
        gcs_path = f"preclean/{filename}"
        gcs.upload_bytes(clean_bytes, gcs_path)

        with db.get_conn() as conn:
            conn.cursor().execute(
                """
                INSERT INTO preclean
                  (filepath, species, caption, sender_id, sender_name,
                   wa_timestamp, queued_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (filepath) DO NOTHING
                """,
                (gcs_path, species or "", caption,
                 sender_id, sender_name, timestamp, now),
            )

        db.log_dataset_entry(
            filename, species or "", "", sender_id, sender_name,
            timestamp, now, "preclean",
            f"no_label (caption='{caption}')",
        )
        print(f"[PRECLEAN] {filename} → awaiting vet label")
        return {"status": "preclean", "reason": "no_label", "gcs_path": gcs_path}

    # ── PATH B: label found → run ensemble ────────────────────────────────────
    confidence = 1.0
    pred_label = label
    try:
        from app.models import ensemble
        result = ensemble.predict(
            clean_bytes, species, sender_id, sender_name, timestamp,
            auto_queue_vet=False,
        )
        if "error" not in result:
            confidence = result["confidence"]
            pred_label = result["label"]
    except Exception as exc:
        print(f"[PIPELINE] Ensemble unavailable ({exc}), using caption label")

    # ── LOW confidence → vet_queue ─────────────────────────────────────────────
    if confidence < VET_CONFIDENCE_THRESHOLD:
        filename = _unique_filename(species, pred_label, sender_id)
        gcs_path = f"vet_queue/{filename}"
        gcs.upload_bytes(clean_bytes, gcs_path)

        with db.get_conn() as conn:
            conn.cursor().execute(
                """
                INSERT INTO vet_queue
                  (filepath, species, pred_label, confidence,
                   sender_id, sender_name, wa_timestamp, queued_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (filepath) DO NOTHING
                """,
                (gcs_path, species, pred_label, confidence,
                 sender_id, sender_name, timestamp, now),
            )

        db.log_dataset_entry(
            filename, species, pred_label, sender_id, sender_name,
            timestamp, now, "vet_queue",
            f"conf={confidence:.2f}<threshold",
        )
        print(f"[VET QUEUE] {filename} conf={confidence:.2f}")
        return {
            "status": "vet_queue",
            "reason": f"confidence={confidence:.2f}",
            "gcs_path": gcs_path,
            "pred_label": pred_label,
            "confidence": confidence,
        }

    # ── HIGH confidence → dataset ──────────────────────────────────────────────
    split = _pick_split(species, pred_label)
    filename = _unique_filename(species, pred_label, sender_id)
    gcs_path = f"dataset/{species}/{split}/{pred_label}/{filename}"
    gcs.upload_bytes(clean_bytes, gcs_path)
    db.increment_split_count(species, pred_label, split)

    db.log_dataset_entry(
        filename, species, pred_label, sender_id, sender_name,
        timestamp, now, "accepted", f"split={split}",
    )
    print(f"[OK] {filename} → {species}/{split}/{pred_label} blur={blur:.1f} conf={confidence:.2f}")
    return {
        "status": "accepted",
        "reason": f"split={split}",
        "gcs_path": gcs_path,
        "label": pred_label,
        "confidence": confidence,
        "split": split,
    }


# ── Vet review actions (called from dashboard router) ─────────────────────────

def approve_preclean(
    item_id: int, vet_species: str, vet_label: str,
    vet_id: str, notes: str = "",
) -> bool:
    """Move a preclean image to the dataset after vet approval."""
    now = datetime.utcnow().isoformat()

    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT filepath, sender_id FROM preclean WHERE id = %s", (item_id,))
        row = cur.fetchone()
    if not row:
        return False

    src_gcs, sender_id = row
    try:
        img_bytes = gcs.download_bytes(src_gcs)
    except Exception:
        return False

    split = _pick_split(vet_species, vet_label)
    filename = _unique_filename(vet_species, vet_label, f"vet_{vet_id}")
    dst_gcs = f"dataset/{vet_species}/{split}/{vet_label}/{filename}"

    gcs.upload_bytes(img_bytes, dst_gcs)
    gcs.delete_blob(src_gcs)
    db.increment_split_count(vet_species, vet_label, split)

    with db.get_conn() as conn:
        conn.cursor().execute(
            """
            UPDATE preclean
            SET vet_species=%s, vet_label=%s, vet_id=%s,
                reviewed_at=%s, status='approved', notes=%s
            WHERE id=%s
            """,
            (vet_species, vet_label, vet_id, now, notes, item_id),
        )

    db.log_dataset_entry(
        filename, vet_species, vet_label, "vet_" + vet_id, "",
        "", now, "accepted", f"preclean_approved → {split}",
    )
    return True


def reject_preclean(item_id: int, vet_id: str, notes: str = "") -> None:
    now = datetime.utcnow().isoformat()
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT filepath FROM preclean WHERE id = %s", (item_id,))
        row = cur.fetchone()
    if row:
        gcs.move_blob(row[0], f"rejected/{row[0].split('/')[-1]}")
        with db.get_conn() as conn:
            conn.cursor().execute(
                "UPDATE preclean SET vet_id=%s, reviewed_at=%s, status='rejected', notes=%s WHERE id=%s",
                (vet_id, now, notes, item_id),
            )


def apply_vet_queue_decision(
    filepath: str, vet_label: Optional[str],
    vet_id: str, status: str,
) -> None:
    """Move a vet_queue image to the dataset or rejected/ based on vet decision."""
    now = datetime.utcnow().isoformat()

    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE vet_queue SET vet_label=%s, vet_id=%s, reviewed_at=%s, status=%s WHERE filepath=%s",
            (vet_label, vet_id, now, status, filepath),
        )

    if status == "rejected":
        gcs.move_blob(filepath, f"rejected/{filepath.split('/')[-1]}")
        return

    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT species, pred_label FROM vet_queue WHERE filepath = %s",
            (filepath,),
        )
        row = cur.fetchone()
    if not row:
        return

    species = row[0]
    label = vet_label if vet_label else row[1]

    try:
        img_bytes = gcs.download_bytes(filepath)
    except Exception as exc:
        print(f"[VET QUEUE] Cannot download {filepath}: {exc}")
        return

    split = _pick_split(species, label)
    filename = _unique_filename(species, label, f"vet_{vet_id}")
    dst_gcs = f"dataset/{species}/{split}/{label}/{filename}"
    gcs.upload_bytes(img_bytes, dst_gcs)
    gcs.delete_blob(filepath)
    db.increment_split_count(species, label, split)
    print(f"[VET OK] {filename} → {species}/{split}/{label} ({status})")


def add_custom_class(species: str, class_name: str) -> bool:
    """Register a new class in the DB."""
    now = datetime.utcnow().isoformat()
    with db.get_conn() as conn:
        conn.cursor().execute(
            """
            INSERT INTO custom_classes (species, class_name, added_at) VALUES (%s,%s,%s)
            ON CONFLICT (species, class_name) DO NOTHING
            """,
            (species, class_name, now),
        )
    print(f"[CLASS] Added '{class_name}' for {species}")
    return True


def _finish(
    status: str, reason: str,
    sender_id: str, sender_name: str,
    timestamp: str, now: str,
) -> dict:
    db.log_dataset_entry(
        "", "", "", sender_id, sender_name, timestamp, now, status, reason,
    )
    return {"status": status, "reason": reason}
