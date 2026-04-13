"""
routers/llm.py — Gemini-powered LLM treatment plan endpoints.

All responses are Server-Sent Events (SSE) so the client receives
streamed text as it is generated.

GET /llm/treatment?disease=…&conf=…&dog_name=…&dog_age=…&dog_species=…
    Stream a full clinical management plan for the diagnosed condition.

GET /llm/followup?disease=…&conf=…&q=…&dog_name=…&dog_age=…&dog_species=…
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
    dog_name: str = "", dog_age: str = "", dog_species: str = "",
) -> str:
    profile_parts = [p for p in [
        f"Name: {dog_name}" if dog_name else "",
        f"Age: {dog_age} years" if dog_age else "",
        f"Breed/Species: {dog_species}" if dog_species else "",
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
            + (f" Tailor advice specifically for {dog_name} the {dog_species}."
               if dog_name and dog_species else "")
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
        + (f"\n\nNote: The patient is {dog_name}, a {dog_age}-year-old {dog_species}. "
           "Adjust dosing and care advice accordingly." if dog_name else "")
    )


# ── Gemini SSE generator ───────────────────────────────────────────────────────

def _sse_gemini(prompt: str) -> Generator[str, None, None]:
    if not GEMINI_API_KEY:
        yield f"data: {json.dumps({'error': 'GEMINI_API_KEY not configured'})}\n\n"
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
            resp.raise_for_status()
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
    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    yield "data: [DONE]\n\n"


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/llm/treatment")
def llm_treatment(
    disease: str = "",
    conf: float = 0.0,
    dog_name: str = "",
    dog_age: str = "",
    dog_species: str = "",
):
    """
    Stream a clinical management plan as SSE.

    Connect via `EventSource('/llm/treatment?disease=Mange&conf=87.3')`.
    Each event carries `{"text": "..."}`.  The stream ends with `[DONE]`.
    """
    prompt = _build_treatment_prompt(disease, conf, dog_name, dog_age, dog_species)
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
    dog_name: str = "",
    dog_age: str = "",
    dog_species: str = "",
):
    """
    Stream an answer to a follow-up question as SSE.

    The question *q* is answered in the context of the current diagnosis.
    """
    parts = [p for p in [
        dog_name,
        f"{dog_age}-year-old" if dog_age else "",
        dog_species,
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
