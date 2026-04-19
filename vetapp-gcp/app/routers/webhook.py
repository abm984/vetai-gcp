"""
routers/webhook.py — WhatsApp Business API webhook.

GET  /webhook   — Meta hub verification (returns hub.challenge)
POST /webhook   — Receive messages, validate HMAC, dispatch to pipeline

Ported from Arch/webhook.py with async httpx and GCS/PostgreSQL backend.
"""

from __future__ import annotations

import asyncio
import hmac
import hashlib
import json
import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse

from app.config import (
    WA_VERIFY_TOKEN, WA_APP_SECRET, SPECIES_TRIGGERS, VET_NUMBERS,
)
from app.pipeline import whatsapp as wa

router = APIRouter(tags=["WhatsApp Webhook"])


# ── 1. Hub verification ────────────────────────────────────────────────────────

@router.get("/webhook")
async def verify_webhook(request: Request):
    """Respond to Meta's webhook verification challenge."""
    params = dict(request.query_params)
    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == WA_VERIFY_TOKEN
    ):
        return PlainTextResponse(params["hub.challenge"])
    raise HTTPException(status_code=403, detail="Verification failed")


# ── 2. Receive messages ────────────────────────────────────────────────────────

@router.post("/webhook")
async def receive_message(request: Request):
    """Handle incoming WhatsApp messages; validate HMAC-SHA256 signature."""
    body_bytes = await request.body()

    # Validate HMAC-SHA256 signature (skip only if secret not configured — dev mode)
    if WA_APP_SECRET:
        sig = request.headers.get("X-Hub-Signature-256", "")
        expected = "sha256=" + hmac.new(
            WA_APP_SECRET.encode(), body_bytes, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise HTTPException(status_code=401, detail="Invalid signature")
    else:
        print("[WEBHOOK WARN] WA_APP_SECRET not set — skipping signature validation")

    data = json.loads(body_bytes)
    try:
        changes = data["entry"][0]["changes"][0]["value"]
        messages = changes.get("messages", [])
        contact  = changes.get("contacts", [{}])[0]
        for msg in messages:
            sender_num = msg.get("from", "")
            if sender_num in VET_NUMBERS:
                asyncio.create_task(_handle_vet_message(msg, contact))
            else:
                asyncio.create_task(_handle_collector_message(msg, contact))
    except (KeyError, IndexError):
        pass

    return {"status": "ok"}


# ── 3a. Collector message — images from the field ─────────────────────────────

async def _handle_collector_message(msg: dict, contact: dict):
    msg_type  = msg.get("type")
    sender_id = msg.get("from", "unknown")
    sender    = contact.get("profile", {}).get("name", sender_id)
    timestamp = msg.get("timestamp", "")
    caption   = ""
    media_id  = None

    if msg_type == "image":
        media_id = msg["image"]["id"]
        caption  = msg["image"].get("caption", "")
    elif msg_type == "video":
        media_id = msg["video"]["id"]
        caption  = msg["video"].get("caption", "")
    elif msg_type == "document":
        media_id = msg["document"]["id"]
        caption  = msg["document"].get("caption", "")
    else:
        return

    if media_id:
        asyncio.create_task(
            _download_and_process(media_id, caption, sender_id, sender, timestamp)
        )


async def _download_and_process(
    media_id: str, caption: str,
    sender_id: str, sender: str, timestamp: str,
):
    # Download media
    try:
        image_bytes, mime_type = await wa.download_media(media_id)
    except Exception as exc:
        print(f"[WEBHOOK] Media download failed: {exc}")
        await wa.send_text(
            sender_id,
            "❌ Could not download your image. Please try again.",
        )
        return

    # Run pipeline (blocking IO → thread)
    from app.pipeline import processor
    result = await asyncio.to_thread(
        processor.process_incoming,
        image_bytes, caption, sender_id, sender, timestamp,
    )

    status = result.get("status")

    # Run ensemble and send reply
    text_lower = re.sub(r"[^a-z0-9 ]", " ", caption.lower())
    species = None
    for sp, triggers in SPECIES_TRIGGERS.items():
        if any(t in text_lower for t in triggers):
            species = sp
            break

    if species and status in ("accepted", "vet_queue"):
        try:
            from app.models import ensemble
            ens_result = await asyncio.to_thread(
                ensemble.predict,
                image_bytes, species, sender_id, sender, timestamp,
                False,
            )
            if "label" in ens_result:
                ens_result["flagged_vet"] = status == "vet_queue"
                reply = wa.build_diagnosis_reply(ens_result)
                await wa.send_text(sender_id, reply)
                return
        except Exception as exc:
            print(f"[WEBHOOK] Ensemble error: {exc}")

    if status == "preclean":
        await wa.send_text(
            sender_id,
            "✅ Image received!\n"
            "Please add species + disease in the caption next time.\n"
            "Example: *dog bacterial* or *cat ringworm*",
        )
    elif status == "rejected":
        reason = result.get("reason", "unknown")
        await wa.send_text(
            sender_id,
            f"❌ Image could not be used ({reason}).\n"
            "Please send a clearer, well-lit photo.",
        )
    else:
        await wa.send_text(
            sender_id,
            "✅ Image saved to dataset.\n_(Inference unavailable — no trained model found)_",
        )


# ── 3b. Vet message — label corrections / approvals ───────────────────────────

async def _handle_vet_message(msg: dict, contact: dict):
    """
    Vets send text commands to manage the vet_queue:
        APPROVE <filename>
        CORRECT <filename> <label>
        REJECT  <filename>
    """
    if msg.get("type") != "text":
        return

    vet_id = msg.get("from", "unknown")
    text   = msg.get("text", {}).get("body", "").strip()
    parts  = text.split(None, 2)

    if len(parts) < 2:
        await wa.send_text(
            vet_id,
            "Commands:\nAPPROVE <filename>\nCORRECT <filename> <label>\nREJECT <filename>",
        )
        return

    cmd, filename = parts[0].upper(), parts[1]
    filepath = f"vet_queue/{filename}"

    from app.pipeline.processor import apply_vet_queue_decision

    if cmd == "APPROVE":
        await asyncio.to_thread(
            apply_vet_queue_decision, filepath, None, vet_id, "approved"
        )
        await wa.send_text(vet_id, f"✅ Approved: {filename}")

    elif cmd == "CORRECT" and len(parts) == 3:
        new_label = parts[2]
        await asyncio.to_thread(
            apply_vet_queue_decision, filepath, new_label, vet_id, "corrected"
        )
        await wa.send_text(vet_id, f"✏️ Corrected: {filename} → {new_label}")

    elif cmd == "REJECT":
        await asyncio.to_thread(
            apply_vet_queue_decision, filepath, None, vet_id, "rejected"
        )
        await wa.send_text(vet_id, f"🗑 Rejected: {filename}")

    else:
        await wa.send_text(vet_id, "Unknown command. Use APPROVE / CORRECT / REJECT")
