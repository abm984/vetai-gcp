"""
routers/llm.py — Gemini-powered LLM treatment plan endpoints.

All responses are Server-Sent Events (SSE) so the client receives
streamed text as it is generated.

GET /llm/treatment?disease=…&conf=…&pet_name=…&pet_age=…&pet_breed=…
    Stream a full clinical management plan for the diagnosed condition.

GET /llm/followup?disease=…&conf=…&q=…&pet_name=…&pet_age=…&pet_breed=…
    Stream an answer to a vet/owner follow-up question in the context
    of the current diagnosis.
"""

from __future__ import annotations

import json
from typing import Generator

import requests as _requests
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.config import GEMINI_API_KEY, GEMINI_MODEL, LLM_SYSTEM_PROMPT

router = APIRouter(tags=["LLM"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",   # disables Nginx proxy buffering
}


# ── Prompt builders (ported from vetai/app_web.py) ────────────────────────────

def _build_treatment_prompt(
    disease: str, confidence: float,
    pet_name: str = "", pet_age: str = "", pet_breed: str = "",
) -> str:
    profile_parts = [p for p in [
        f"Name: {pet_name}" if pet_name else "",
        f"Age: {pet_age} years" if pet_age else "",
        f"Breed: {pet_breed}" if pet_breed else "",
    ] if p]
    patient_header = (
        f"**Patient Profile:** {', '.join(profile_parts)}\n\n"
        if profile_parts else ""
    )

    if disease.lower() == "healthy":
        return (
            f"{patient_header}"
            f"Clinical Assessment: The integumentary scan indicates a 'Healthy' status "
            f"with {confidence:.1f}% confidence. Provide a **Proactive Maintenance Plan** "
            "focusing on barrier function, nutrition, and early detection of lesions."
            + (f" Tailor advice specifically for {pet_name} the {pet_breed}."
               if pet_name and pet_breed else "")
        )
    return (
        f"{patient_header}"
        f"The patient presents with clinical signs consistent with **{disease}** "
        f"(Diagnostic Confidence: {confidence:.1f}%).\n\n"
        "As a veterinarian, draft a **Clinical Management Strategy** including:\n"
        "1. **Immediate Stabilization & Home Care**: (Triage steps to reduce discomfort)\n"
        "2. **Pharmacological Intervention**: (Commonly prescribed meds and their MOA)\n"
        "3. **Clinical Red Flags**: (Specific triggers for emergency intervention)\n"
        "4. **Prognosis & Recovery**: (Expected healing milestones)\n"
        "5. **Long-term Prophylaxis**: (Strategies to prevent recurrence)\n\n"
        "Maintain a professional, diagnostic tone throughout."
        + (f"\n\nNote: The patient is {pet_name}, a {pet_age}-year-old {pet_breed}. "
           "Adjust dosing and care advice accordingly." if pet_name else "")
    )


# ── Gemini SSE generator ───────────────────────────────────────────────────────

def _sse_gemini(prompt: str) -> Generator[str, None, None]:
    if not GEMINI_API_KEY:
        yield f"data: {json.dumps({'error': 'GEMINI_API_KEY not configured — set the secret in Cloud Run'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:streamGenerateContent?alt=sse&key={GEMINI_API_KEY}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": LLM_SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": prompt}]}],
    }
    try:
        with _requests.post(url, json=payload, stream=True, timeout=90) as resp:
            if resp.status_code == 429:
                yield f"data: {json.dumps({'error': 'Gemini quota exceeded. Free-tier daily limit reached — wait ~24 h or enable billing.'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            if resp.status_code == 400:
                msg = f"Gemini API bad request (400) — model \"{GEMINI_MODEL}\" may be unavailable or the request is malformed."
                yield f"data: {json.dumps({'error': msg})}\n\n"
                yield "data: [DONE]\n\n"
                return
            if resp.status_code == 403:
                yield f"data: {json.dumps({'error': 'Gemini API key is invalid or lacks permission (403). Update the GEMINI_API_KEY secret.'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            if not resp.ok:
                msg = f"Gemini API error {resp.status_code}. Check your API key and try again."
                yield f"data: {json.dumps({'error': msg})}\n\n"
                yield "data: [DONE]\n\n"
                return
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8") if isinstance(line, bytes) else line
                if line.startswith("data:"):
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        obj = json.loads(raw)
                        text = (
                            obj.get("candidates", [{}])[0]
                               .get("content", {})
                               .get("parts", [{}])[0]
                               .get("text", "")
                        )
                        if text:
                            yield f"data: {json.dumps({'text': text})}\n\n"
                    except Exception:
                        pass
    except _requests.exceptions.Timeout:
        yield f"data: {json.dumps({'error': 'Gemini request timed out. Check your internet connection and try again.'})}\n\n"
    except _requests.exceptions.ConnectionError:
        yield f"data: {json.dumps({'error': 'Could not connect to Gemini API. Check network connectivity.'})}\n\n"
    except Exception as exc:
        msg = f"Gemini error: {type(exc).__name__}"
        yield f"data: {json.dumps({'error': msg})}\n\n"

    yield "data: [DONE]\n\n"


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/llm/treatment")
def llm_treatment(
    disease: str = "",
    conf: float = 0.0,
    pet_name: str = "",
    pet_age: str = "",
    pet_breed: str = "",
):
    """
    Stream a clinical management plan as SSE.

    Connect via `EventSource('/llm/treatment?disease=Mange&conf=87.3')`.
    Each event carries `{"text": "..."}`.  The stream ends with `[DONE]`.
    """
    prompt = _build_treatment_prompt(disease, conf, pet_name, pet_age, pet_breed)
    return StreamingResponse(
        _sse_gemini(prompt),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.get("/llm/followup")
def llm_followup(
    disease: str = "",
    conf: float = 0.0,
    q: str = "",
    pet_name: str = "",
    pet_age: str = "",
    pet_breed: str = "",
):
    """
    Stream an answer to a follow-up question as SSE.

    The question *q* is answered in the context of the current diagnosis.
    """
    parts = [p for p in [
        pet_name,
        f"{pet_age}-year-old" if pet_age else "",
        pet_breed,
    ] if p]
    patient_ctx = f"Patient: {' '.join(parts)}. " if parts else ""

    prompt = (
        f"{patient_ctx}Context: patient diagnosed with '{disease}' "
        f"at {conf:.1f}% confidence.\n"
        f"Pet owner follow-up question: {q}"
    )
    return StreamingResponse(
        _sse_gemini(prompt),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
