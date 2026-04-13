"""
routers/dashboard.py — Vet dashboard REST API.

All endpoints return JSON; the frontend HTML (dashboard.html) can be
hosted separately on Firebase Hosting / GCS static site and call this
API with the appropriate CORS origin set in main.py.

Ported from Arch/api.py with PostgreSQL + GCS backends.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app import database as db
from app import storage as gcs
from app.config import CLASSES, SPLITS
from app.schemas import AddClassRequest, PrecleanDecision, VetDecision

router = APIRouter(prefix="/api", tags=["Dashboard"])


# ── Classes ───────────────────────────────────────────────────────────────────

@router.get("/classes")
def get_classes():
    """All disease classes (config + custom DB entries)."""
    return db.get_all_classes()


@router.post("/classes/add")
def add_class(body: AddClassRequest):
    if body.species not in CLASSES:
        raise HTTPException(400, f"Unknown species: {body.species}")
    class_name = body.class_name.strip().replace(" ", "_")
    if not class_name:
        raise HTTPException(400, "class_name is required")
    all_cls = db.get_all_classes()
    if class_name in all_cls.get(body.species, []):
        return {"ok": True, "class_name": class_name, "message": "already exists"}
    from app.pipeline.processor import add_custom_class
    add_custom_class(body.species, class_name)
    return {"ok": True, "class_name": class_name}


# ── Dataset statistics ────────────────────────────────────────────────────────

@router.get("/stats")
def get_stats():
    """Per-class image counts split by train / valid / test."""
    result = {}
    for species, cls_list in db.get_all_classes().items():
        result[species] = {}
        for cls in cls_list:
            counts = db.get_split_counts(species, cls)
            result[species][cls] = {s: counts.get(s, 0) for s in SPLITS}
    return result


@router.get("/overview")
def get_overview():
    """Summary counts: total images, pending queues, acceptance rate."""
    stats = get_stats()
    totals = {
        sp: sum(sum(splits.values()) for splits in cls.values())
        for sp, cls in stats.items()
    }

    vet: dict[str, int] = {"pending": 0, "approved": 0, "corrected": 0, "rejected": 0}
    pc:  dict[str, int] = {"pending": 0, "approved": 0, "rejected": 0}
    rejected_count = 0

    with db.get_conn() as conn:
        cur = conn.cursor()

        cur.execute("SELECT status, COUNT(*) FROM vet_queue GROUP BY status")
        for row in cur.fetchall():
            vet[row[0]] = row[1]

        cur.execute("SELECT status, COUNT(*) FROM preclean GROUP BY status")
        for row in cur.fetchall():
            pc[row[0]] = row[1]

        cur.execute("SELECT COUNT(*) FROM dataset_log WHERE status = 'rejected'")
        rejected_count = cur.fetchone()[0]

    grand = sum(totals.values())
    total_processed = grand + rejected_count
    return {
        "total":            grand,
        "dog":              totals.get("dog", 0),
        "cat":              totals.get("cat", 0),
        "rejected":         rejected_count,
        "preclean_pending": pc.get("pending", 0),
        "vet_pending":      vet.get("pending", 0),
        "acceptance_rate":  round(grand / total_processed * 100, 1)
                            if total_processed else 0,
    }


@router.get("/rejections")
def get_rejections():
    """Rejection reason breakdown."""
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT SPLIT_PART(reason, '(', 1) AS short_reason, COUNT(*)
            FROM dataset_log
            WHERE status = 'rejected'
            GROUP BY short_reason
            ORDER BY COUNT(*) DESC
            """
        )
        return {row[0]: row[1] for row in cur.fetchall()}


@router.get("/activity")
def get_activity(limit: int = 50):
    """Most recent *limit* pipeline events."""
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT filename, species, label, sender_name, sender_id,
                   status, reason, processed_at
            FROM dataset_log
            ORDER BY id DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
    return [
        {
            "filename":     r[0],
            "species":      r[1],
            "label":        r[2],
            "sender_name":  r[3],
            "sender_id":    r[4],
            "status":       r[5],
            "reason":       r[6],
            "processed_at": r[7],
        }
        for r in rows
    ]


# ── Preclean queue ────────────────────────────────────────────────────────────

@router.get("/preclean")
def get_preclean():
    """Images awaiting vet labelling."""
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, filepath, species, caption,
                   sender_name, queued_at, status, vet_label, vet_species, notes
            FROM preclean
            ORDER BY queued_at DESC
            """
        )
        rows = cur.fetchall()
    return [
        {
            "id":          r[0],
            "filepath":    r[1],
            "filename":    r[1].split("/")[-1],
            "species":     r[2] or "unknown",
            "caption":     r[3] or "",
            "sender_name": r[4] or "Unknown",
            "queued_at":   r[5],
            "status":      r[6],
            "vet_label":   r[7],
            "vet_species": r[8],
            "notes":       r[9] or "",
            "image_url":   _preclean_signed_url(r[1]),
        }
        for r in rows
    ]


def _preclean_signed_url(gcs_path: str) -> str:
    try:
        return gcs.signed_url(gcs_path, expiry_minutes=30)
    except Exception:
        return ""


@router.post("/preclean/decide")
def decide_preclean(body: PrecleanDecision):
    if body.status not in ("approved", "rejected"):
        raise HTTPException(400, "status must be 'approved' or 'rejected'")

    from app.pipeline.processor import approve_preclean, reject_preclean

    if body.status == "approved":
        if not body.vet_species or not body.vet_label:
            raise HTTPException(400, "vet_species and vet_label required for approval")
        ok = approve_preclean(
            body.id, body.vet_species, body.vet_label, body.vet_id, body.notes
        )
        if not ok:
            raise HTTPException(404, "Item not found or file missing in GCS")
    else:
        reject_preclean(body.id, body.vet_id, body.notes)

    return {"ok": True}


# ── Vet queue ─────────────────────────────────────────────────────────────────

@router.get("/vet-queue")
def get_vet_queue():
    """Low-confidence model predictions awaiting vet review."""
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, filepath, species, pred_label, confidence,
                   sender_name, queued_at, status, vet_label
            FROM vet_queue
            ORDER BY queued_at DESC
            """
        )
        rows = cur.fetchall()
    return [
        {
            "id":          r[0],
            "filepath":    r[1],
            "filename":    r[1].split("/")[-1],
            "species":     r[2],
            "pred_label":  r[3],
            "confidence":  round(r[4], 3) if r[4] else 0,
            "sender_name": r[5] or "Unknown",
            "queued_at":   r[6],
            "status":      r[7],
            "vet_label":   r[8],
            "image_url":   _preclean_signed_url(r[1]),
        }
        for r in rows
    ]


@router.post("/vet-decision")
def post_vet_decision(body: VetDecision):
    if body.status not in ("approved", "corrected", "rejected"):
        raise HTTPException(400, "status must be approved / corrected / rejected")

    # Fetch filepath from DB
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT filepath FROM vet_queue WHERE id = %s", (body.id,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(404, "Vet queue item not found")

    from app.pipeline.processor import apply_vet_queue_decision
    apply_vet_queue_decision(
        row[0],
        body.vet_label or None,
        "dashboard_vet",
        body.status,
    )
    return {"ok": True}
