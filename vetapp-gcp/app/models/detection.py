"""
models/detection.py — PyTorch EfficientNet-V2-S + GradCAM.

Ported from vetai/app_web.py.

Model weights (best_model.pth) are downloaded from GCS on first use
and cached in memory.  A threading lock prevents concurrent forward
passes from corrupting GradCAM hooks.
"""

from __future__ import annotations

import io
import base64
import threading
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from app.config import (
    DETECTION_NUM_CLASSES, DETECTION_IMG_SIZE,
    DETECTION_CLASS_NAMES, DETECTION_CLASS_COLORS_RGB,
    DETECTION_CLASS_COLORS_HEX, DETECTION_CLASS_SEVERITY,
    DETECTION_CLASS_ADVICE, GCS_MODELS_PREFIX,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_GCS_PATH = f"{GCS_MODELS_PREFIX}best_model.pth"

_transform = transforms.Compose([
    transforms.Resize((DETECTION_IMG_SIZE, DETECTION_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Module-level singletons
_detection_model: Optional[nn.Module] = None
_gradcam_instance: Optional["GradCAM"] = None
_model_lock = threading.Lock()       # serialises forward+backward passes
_load_lock = threading.Lock()        # prevents concurrent model downloads
_status: dict = {"ready": False, "error": ""}


# ── GradCAM ───────────────────────────────────────────────────────────────────

class GradCAM:
    """Gradient-weighted Class Activation Mapping on the last feature block."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._lock = threading.Lock()
        target = model.features[-1]
        self._h_fwd = target.register_forward_hook(self._save_act)
        self._h_bwd = target.register_full_backward_hook(self._save_grad)

    def _save_act(self, _m, _inp, out):
        self.activations = out

    def _save_grad(self, _m, _gi, go):
        self.gradients = go[0]

    def generate(self, pil_img: Image.Image, class_idx: int) -> np.ndarray:
        with self._lock:
            t = _transform(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
            self.model.zero_grad()
            out = self.model(t)
            out[0, class_idx].backward()
            weights = self.gradients.mean(dim=[0, 2, 3])
            acts = self.activations.squeeze(0)
            cam = (weights[:, None, None] * acts).sum(0)
            cam = torch.relu(cam)
            cam -= cam.min()
            if cam.max() > 1e-8:
                cam /= cam.max()
            return cam.detach().cpu().numpy()

    def remove(self):
        self._h_fwd.remove()
        self._h_bwd.remove()


# ── Model construction ────────────────────────────────────────────────────────

def _build_arch() -> nn.Module:
    m = models.efficientnet_v2_s(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_f, 512),
        nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, DETECTION_NUM_CLASSES),
    )
    return m


# ── Loading ───────────────────────────────────────────────────────────────────

def load_model() -> None:
    """Download weights from GCS and initialise the model + GradCAM hooks.
    Safe to call multiple times — subsequent calls are no-ops."""
    global _detection_model, _gradcam_instance

    with _load_lock:
        if _status["ready"]:
            return
        try:
            from app.storage import download_bytes
            weights_bytes = download_bytes(_MODEL_GCS_PATH)
            m = _build_arch()
            state = torch.load(io.BytesIO(weights_bytes), map_location=DEVICE)
            m.load_state_dict(state)
            m.to(DEVICE).eval()
            _detection_model = m
            _gradcam_instance = GradCAM(m)
            _status["ready"] = True
            print(f"[DETECTION] Loaded best_model.pth ({DEVICE})")
        except Exception as exc:
            _status["error"] = str(exc)
            print(f"[DETECTION ERROR] {exc}")


def get_status() -> dict:
    return dict(_status)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_predict(
    pil_img: Image.Image,
) -> Tuple[List[Tuple[str, float]], Optional[str], Optional[Dict[str, int]]]:
    """
    Run detection + GradCAM on a PIL image.

    Returns:
        top3       — [(class_name, probability), ...] sorted by confidence
        overlay    — GradCAM heatmap as a data-URL PNG (or None on failure)
        bbox       — bounding box dict {x, y, w, h} (or None on failure)

    Raises RuntimeError if the model is not loaded.
    """
    if not _status["ready"]:
        raise RuntimeError(_status["error"] or "Detection model not yet loaded")

    with _model_lock:
        t = _transform(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(_detection_model(t), dim=1).squeeze().cpu().numpy()

    top3_raw = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
    top3 = [(DETECTION_CLASS_NAMES[i], float(p)) for i, p in top3_raw]

    top_name = top3[0][0]
    top_idx = DETECTION_CLASS_NAMES.index(top_name)
    color_rgb = DETECTION_CLASS_COLORS_RGB[top_name]

    overlay = bbox = None
    try:
        cam = _gradcam_instance.generate(pil_img, top_idx)
        overlay = _cam_to_overlay_b64(cam, pil_img.width, pil_img.height, color_rgb)
        bbox = _cam_to_bbox(cam, pil_img.width, pil_img.height)
    except Exception as exc:
        print(f"[GRADCAM] {exc}")

    return top3, overlay, bbox


# ── GradCAM post-processing ───────────────────────────────────────────────────

def _cam_to_overlay_b64(cam: np.ndarray, w: int, h: int,
                         color_rgb: tuple) -> str:
    resized = cv2.resize(cam, (w, h))
    r, g, b = color_rgb
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = r
    rgba[:, :, 1] = g
    rgba[:, :, 2] = b
    rgba[:, :, 3] = (resized * 180).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _cam_to_bbox(cam: np.ndarray, w: int, h: int,
                  threshold: float = 0.35) -> Optional[Dict[str, int]]:
    resized = cv2.resize(cam, (w, h))
    binary = (resized > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    pts = np.concatenate(contours)
    x, y, bw, bh = cv2.boundingRect(pts)
    pad = 12
    x = max(0, x - pad)
    y = max(0, y - pad)
    bw = min(w - x, bw + 2 * pad)
    bh = min(h - y, bh + 2 * pad)
    return {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)}


# ── Convenience ───────────────────────────────────────────────────────────────

def b64_to_pil(data_url: str) -> Image.Image:
    """Decode a base64 data URL to a PIL Image."""
    _, b64 = data_url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
