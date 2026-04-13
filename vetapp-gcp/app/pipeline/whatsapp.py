"""
pipeline/whatsapp.py — Async WhatsApp Business API client.

Handles:
  • Downloading media (image / video / document) from the Graph API.
  • Sending text replies back to a sender.

All network calls use httpx with async/await so they never block
the FastAPI event loop.
"""

from __future__ import annotations

import httpx

from app.config import WA_ACCESS_TOKEN, WA_PHONE_ID, WA_API_VERSION

_BASE = f"https://graph.facebook.com/{WA_API_VERSION}"
_HEADERS = lambda: {
    "Authorization": f"Bearer {WA_ACCESS_TOKEN}",
    "Content-Type": "application/json",
}


async def download_media(media_id: str) -> tuple[bytes, str]:
    """
    Fetch media bytes from the WhatsApp CDN.

    Returns:
        (raw_bytes, mime_type)  e.g.  (b"...", "image/jpeg")
    """
    async with httpx.AsyncClient(timeout=60) as client:
        # Step 1: resolve the media URL
        r = await client.get(
            f"{_BASE}/{media_id}",
            headers={"Authorization": f"Bearer {WA_ACCESS_TOKEN}"},
        )
        r.raise_for_status()
        meta = r.json()
        media_url = meta["url"]
        mime_type = meta.get("mime_type", "image/jpeg")

        # Step 2: download the actual bytes
        r2 = await client.get(
            media_url,
            headers={"Authorization": f"Bearer {WA_ACCESS_TOKEN}"},
        )
        r2.raise_for_status()

    return r2.content, mime_type


async def send_text(to: str, body: str) -> None:
    """Send a plain-text WhatsApp message to *to* (E.164 number)."""
    url = f"{_BASE}/{WA_PHONE_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body},
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=_HEADERS(), json=payload)
        if r.status_code >= 400:
            print(f"[WA SEND ERROR] {r.status_code} {r.text}")


def build_diagnosis_reply(result: dict) -> str:
    """Format a friendly WhatsApp diagnosis message from an ensemble result."""
    conf_pct = result["confidence"] * 100
    flag_line = (
        "⚠️ _Low confidence — sent to vet for review_"
        if result.get("flagged_vet") else "✅ _High confidence_"
    )
    top3 = sorted(result["all_probs"].items(), key=lambda x: -x[1])[:3]
    prob_lines = "\n".join(f"  {cls}: {p*100:.1f}%" for cls, p in top3)

    return (
        f"🐾 *PetDerm Diagnosis*\n"
        f"━━━━━━━━━━━━━━━\n"
        f"Species: {result['species'].capitalize()}\n"
        f"Diagnosis: *{result['label']}*\n"
        f"Confidence: *{conf_pct:.1f}%*\n"
        f"\nTop predictions:\n{prob_lines}\n\n"
        f"{flag_line}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"_Models used: {result['n_models']} folds_"
    )
