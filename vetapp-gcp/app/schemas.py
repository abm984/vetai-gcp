"""
schemas.py — Pydantic request / response models.
"""

from typing import Optional, Dict, List, Tuple
from pydantic import BaseModel, Field


# ── /predict ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """Send a base64-encoded image for disease detection + GradCAM."""
    image: str = Field(..., description="Data URL: 'data:image/jpeg;base64,…'")
    dog_name: str = ""
    dog_age: str = ""
    dog_species: str = ""


class PredictResponse(BaseModel):
    top3: List[Tuple[str, float]]       # [(class_name, prob), ...]
    color: str                          # hex colour of top class
    severity: str                       # HIGH / MEDIUM / LOW / HEALTHY
    advice: str                         # one-line clinical note
    overlay: Optional[str] = None       # GradCAM overlay as data URL PNG
    bbox: Optional[Dict[str, int]] = None  # {x, y, w, h}


# ── /api/classes ──────────────────────────────────────────────────────────────
class AddClassRequest(BaseModel):
    species: str
    class_name: str


# ── /api/preclean/decide ──────────────────────────────────────────────────────
class PrecleanDecision(BaseModel):
    id: int
    status: str              # "approved" | "rejected"
    vet_species: str = ""
    vet_label: str = ""
    vet_id: str = "dashboard_vet"
    notes: str = ""


# ── /api/vet-decision ─────────────────────────────────────────────────────────
class VetDecision(BaseModel):
    id: int
    status: str              # "approved" | "corrected" | "rejected"
    vet_label: str = ""
