"""
models/ensemble.py — TensorFlow/Keras k-fold ensemble for pipeline inference.

Ported from Arch/predict.py.

All five fold models per species are downloaded from GCS on first use
and cached in-process.  Downloading 5 × ~123 MB models takes roughly
30 seconds; this is done lazily on the first prediction request for
each species so startup time stays fast.
"""

from __future__ import annotations

import os
import numpy as np
from typing import Optional
from PIL import Image

from app.config import (
    CLASSES, N_FOLDS, GCS_MODELS_PREFIX, VET_CONFIDENCE_THRESHOLD, TMP_DIR,
)

# Per-species in-memory model caches: {"dog": [model, ...], "cat": [...]}
_model_cache: dict[str, list] = {}
_load_status: dict[str, str] = {}   # species → "ok" | "error:<msg>" | "empty"


def _gcs_path(species: str, fold: int) -> str:
    return f"{GCS_MODELS_PREFIX}{species}_fold{fold}.keras"


def _local_path(species: str, fold: int) -> str:
    return os.path.join(TMP_DIR, "models", f"{species}_fold{fold}.keras")


def _load_models(species: str) -> list:
    """Download (if needed) and cache all fold models for *species*."""
    if species in _model_cache:
        return _model_cache[species]

    from tensorflow import keras  # lazy import — TF takes ~5 s to load
    from app.storage import download_to_tmp

    loaded = []
    for fold_i in range(1, N_FOLDS + 1):
        gcs_path = _gcs_path(species, fold_i)
        try:
            local = download_to_tmp(gcs_path, subdir="models")
            m = keras.models.load_model(local, compile=False)
            loaded.append(m)
            print(f"[ENSEMBLE] Loaded {species} fold {fold_i}")
        except Exception as exc:
            print(f"[ENSEMBLE WARN] Could not load {gcs_path}: {exc}")

    _model_cache[species] = loaded
    _load_status[species] = "ok" if loaded else "empty"
    return loaded


def predict(
    image_bytes: bytes,
    species: str,
    sender_id: str = "",
    sender_name: str = "",
    timestamp: str = "",
    auto_queue_vet: bool = True,
) -> dict:
    """
    Run ensemble inference on raw image bytes.

    Returns a dict compatible with Arch/predict.py:
        {
          "species":     "dog",
          "label":       "Fungal_infections",
          "confidence":  0.83,
          "all_probs":   {"Bacterial_dermatosis": 0.05, ...},
          "flagged_vet": False,
          "n_models":    5
        }

    If no trained models exist for the species the return value will
    contain an "error" key instead.
    """
    import io as _io

    cls_list = CLASSES[species]
    num_classes = len(cls_list)
    models = _load_models(species)

    if not models:
        return {
            "error": f"No trained models found for '{species}'. "
                     f"Upload fold weights to GCS at {GCS_MODELS_PREFIX}."
        }

    # Read input size from the first model
    _, h, w, _ = models[0].input_shape
    model_size = (w, h)

    # Preprocess: open from bytes → resize → normalise
    pil_img = Image.open(_io.BytesIO(image_bytes)).convert("RGB").resize(
        model_size, Image.LANCZOS
    )
    arr = np.expand_dims(np.array(pil_img, dtype=np.float32) / 255.0, 0)

    # Ensemble averaging across folds
    ensemble_preds = np.zeros((1, num_classes))
    for m in models:
        _, mh, mw, _ = m.input_shape
        if (mw, mh) != model_size:
            img_m = Image.open(_io.BytesIO(image_bytes)).convert("RGB").resize(
                (mw, mh), Image.LANCZOS
            )
            arr_m = np.expand_dims(np.array(img_m, dtype=np.float32) / 255.0, 0)
        else:
            arr_m = arr
        ensemble_preds += m.predict(arr_m, verbose=0)
    ensemble_preds /= len(models)

    probs = ensemble_preds[0]
    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    label = cls_list[top_idx]
    all_probs = {cls_list[i]: float(probs[i]) for i in range(num_classes)}

    flagged_vet = False
    if auto_queue_vet and confidence < VET_CONFIDENCE_THRESHOLD:
        # Caller (pipeline/processor.py) handles the DB insert;
        # we just set the flag here to keep concerns separate.
        flagged_vet = True

    return {
        "species":    species,
        "label":      label,
        "confidence": confidence,
        "all_probs":  all_probs,
        "flagged_vet": flagged_vet,
        "n_models":   len(models),
    }


def is_loaded(species: str) -> bool:
    return bool(_model_cache.get(species))


def get_load_status(species: str) -> str:
    return _load_status.get(species, "not_loaded")
