"""
routers/predict.py — Disease detection endpoint.

POST /predict
    Takes a base64-encoded image + optional patient profile.
    Returns top-3 predictions, colour, severity, GradCAM overlay, bbox.

GET /status
    Returns model readiness + device info.
"""

from fastapi import APIRouter, HTTPException
from app.schemas import PredictRequest, PredictResponse
from app.models import detection
from app.config import (
    DETECTION_CLASS_COLORS_HEX,
    DETECTION_CLASS_SEVERITY,
    DETECTION_CLASS_ADVICE,
)

router = APIRouter(tags=["Detection"])


@router.get("/status")
def status():
    """System status: model readiness, device, version."""
    st = detection.get_status()
    return {
        "model_ready":  st["ready"],
        "model_error":  st["error"],
        "device":       str(detection.DEVICE),
        "gemini_key_set": bool(__import__("app.config", fromlist=["GEMINI_API_KEY"]).GEMINI_API_KEY),
    }


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Run skin-disease detection on a base64 image.

    **image** must be a data URL: `data:image/jpeg;base64,<base64data>`
    """
    st = detection.get_status()
    if not st["ready"]:
        raise HTTPException(
            status_code=503,
            detail=st["error"] or "Detection model not yet loaded",
        )

    try:
        pil_img = detection.b64_to_pil(req.image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    try:
        top3, overlay, bbox = detection.run_predict(pil_img)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    top_name = top3[0][0]
    return PredictResponse(
        top3=top3,
        color=DETECTION_CLASS_COLORS_HEX[top_name],
        severity=DETECTION_CLASS_SEVERITY[top_name],
        advice=DETECTION_CLASS_ADVICE[top_name],
        overlay=overlay,
        bbox=bbox,
    )
