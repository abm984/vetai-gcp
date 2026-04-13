"""
config.py — Single source of truth for the GCP deployment.

All runtime values are read from environment variables so that
the same Docker image can run locally, in Cloud Run, and in CI.
Secrets (API keys, DB password) are injected via Cloud Run secret
references pointing to Google Secret Manager entries.
"""

import os
from typing import Set

# ── Species & Classes ──────────────────────────────────────────────────────────
SPECIES = ["dog", "cat"]

DOG_CLASSES = [
    "Bacterial_dermatosis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity_allergic_dermatosis",
]
CAT_CLASSES = [
    "Flea_Allergy",
    "Healthy",
    "Ringworm",
    "Scabies",
]
CLASSES = {"dog": DOG_CLASSES, "cat": CAT_CLASSES}

DOG_ALIASES = {
    "bacterial":              "Bacterial_dermatosis",
    "bacteria":               "Bacterial_dermatosis",
    "bacterial dermatosis":   "Bacterial_dermatosis",
    "pyoderma":               "Bacterial_dermatosis",
    "fungal":                 "Fungal_infections",
    "fungus":                 "Fungal_infections",
    "fungal infection":       "Fungal_infections",
    "yeast":                  "Fungal_infections",
    "healthy":                "Healthy",
    "normal":                 "Healthy",
    "no disease":             "Healthy",
    "hypersensitivity":       "Hypersensitivity_allergic_dermatosis",
    "allergic":               "Hypersensitivity_allergic_dermatosis",
    "allergy":                "Hypersensitivity_allergic_dermatosis",
    "atopy":                  "Hypersensitivity_allergic_dermatosis",
    "hypersensitivity allergic": "Hypersensitivity_allergic_dermatosis",
}
CAT_ALIASES = {
    "flea":            "Flea_Allergy",
    "flea allergy":    "Flea_Allergy",
    "flea bite":       "Flea_Allergy",
    "fad":             "Flea_Allergy",
    "healthy":         "Healthy",
    "normal":          "Healthy",
    "no disease":      "Healthy",
    "ringworm":        "Ringworm",
    "ring worm":       "Ringworm",
    "dermatophytosis": "Ringworm",
    "fungal":          "Ringworm",
    "scabies":         "Scabies",
    "notoedric":       "Scabies",
    "notoedres":       "Scabies",
    "mange":           "Scabies",
    "mite":            "Scabies",
}
ALIASES = {"dog": DOG_ALIASES, "cat": CAT_ALIASES}

SPECIES_TRIGGERS = {
    "dog": ["dog", "canine", "puppy", "pup"],
    "cat": ["cat", "feline", "kitten", "kitty"],
}

# ── Pipeline thresholds ────────────────────────────────────────────────────────
IMG_SIZE = (300, 300)
BLUR_THRESHOLD = 60.0
MIN_FILE_KB = 5
SPLITS = ["train", "valid", "test"]
SPLIT_RATIO = (0.80, 0.10, 0.10)
N_FOLDS = 5
VET_CONFIDENCE_THRESHOLD = 0.63

# ── Detection model — 11-class EfficientNet-V2-S (vetai) ─────────────────────
DETECTION_NUM_CLASSES = 11
DETECTION_IMG_SIZE = 224

DETECTION_CLASS_NAMES = [
    "Bacterial Dermatosis", "Demodicosis", "Dermatitis",
    "Flea Allergy",         "Fungal Infection", "Healthy",
    "Hotspot",              "Hypersensitivity",
    "Hypersensitivity Allergic Dermatosis", "Mange", "Ringworm",
]

DETECTION_CLASS_COLORS_RGB = {
    "Bacterial Dermatosis":                 (248, 113, 113),
    "Demodicosis":                          (251, 146,  60),
    "Dermatitis":                           (251, 191,  36),
    "Flea Allergy":                         (251, 146,  60),
    "Fungal Infection":                     (248, 113, 113),
    "Healthy":                              ( 74, 222, 128),
    "Hotspot":                              (251, 191,  36),
    "Hypersensitivity":                     (251, 191,  36),
    "Hypersensitivity Allergic Dermatosis": (251, 146,  60),
    "Mange":                                (248, 113, 113),
    "Ringworm":                             (251, 146,  60),
}
DETECTION_CLASS_COLORS_HEX = {
    k: "#{:02x}{:02x}{:02x}".format(*v)
    for k, v in DETECTION_CLASS_COLORS_RGB.items()
}
DETECTION_CLASS_SEVERITY = {
    "Bacterial Dermatosis": "HIGH",   "Demodicosis": "MEDIUM",
    "Dermatitis":           "MEDIUM", "Flea Allergy": "LOW",
    "Fungal Infection":     "HIGH",   "Healthy": "HEALTHY",
    "Hotspot":              "LOW",    "Hypersensitivity": "MEDIUM",
    "Hypersensitivity Allergic Dermatosis": "HIGH",
    "Mange":                "HIGH",   "Ringworm": "MEDIUM",
}
DETECTION_CLASS_ADVICE = {
    "Bacterial Dermatosis":                 "Bacterial infection detected. Vet-prescribed antibiotics required.",
    "Demodicosis":                          "Demodex mite infestation. Common in young dogs. Vet treatment needed.",
    "Dermatitis":                           "Skin inflammation. May be contact or atopic. Consult a vet.",
    "Flea Allergy":                         "Allergic reaction to flea bites. Flea treatment + antihistamines.",
    "Fungal Infection":                     "Fungal condition detected. Antifungal medication required.",
    "Healthy":                              "No disease detected. Skin appears healthy!",
    "Hotspot":                              "Moist, irritated skin patch. Keep area clean and dry. Vet advised.",
    "Hypersensitivity":                     "Allergic reaction. Identify & remove the trigger. Vet check recommended.",
    "Hypersensitivity Allergic Dermatosis": "Severe allergic condition. Prompt veterinary attention needed.",
    "Mange":                                "Parasitic mite infestation. Treatable — requires vet care.",
    "Ringworm":                             "Fungal infection. Contagious to humans. Antifungal treatment needed.",
}

# ── GCP — Google Cloud ─────────────────────────────────────────────────────────
# GCS bucket that stores models, incoming images, dataset, preclean, vet_queue.
GCS_BUCKET: str = os.getenv("GCS_BUCKET", "vetapp-data")

# GCS path prefix where model files live, e.g. "models/best_model.pth"
GCS_MODELS_PREFIX: str = os.getenv("GCS_MODELS_PREFIX", "models/")

# Cloud SQL connection string.
# Cloud Run (Unix socket):
#   postgresql://USER:PASS@/DB?host=/cloudsql/PROJECT:REGION:INSTANCE
# Local dev (TCP):
#   postgresql://vetapp:vetapp@localhost:5432/vetapp
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://vetapp:vetapp@localhost:5432/vetapp",
)

# ── LLM ───────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

LLM_SYSTEM_PROMPT = (
    "You are Dr. VetAI, a Senior Veterinary Dermatologist. Your tone is clinical, "
    "authoritative, yet empathetic. Use professional medical terminology (e.g., 'pruritus' "
    "instead of 'itching') but explain it briefly for the owner. "
    "STRICT FORMATTING RULE: Use **Bold Headers** for every section. "
    "Do not use plain text headers. "
    "Always include a professional disclaimer that this is a digital triage and not a "
    "replacement for an in-person physical exam."
)

# ── WhatsApp Business API ──────────────────────────────────────────────────────
WA_VERIFY_TOKEN: str = os.getenv("WA_VERIFY_TOKEN", "my_verify_token")
WA_ACCESS_TOKEN: str = os.getenv("WA_ACCESS_TOKEN", "")
WA_APP_SECRET: str = os.getenv("WA_APP_SECRET", "")
WA_PHONE_ID: str = os.getenv("WA_PHONE_ID", "")
WA_API_VERSION: str = os.getenv("WA_API_VERSION", "v19.0")

# Comma-separated list of vet WhatsApp numbers, e.g. "923001234567,923009876543"
VET_NUMBERS: Set[str] = set(filter(None, os.getenv("VET_NUMBERS", "").split(",")))

# ── Runtime ────────────────────────────────────────────────────────────────────
# Cloud Run sets PORT automatically; default 8080 matches Cloud Run convention.
PORT: int = int(os.getenv("PORT", "8080"))

# Ephemeral scratch space for downloaded images/models within the container.
TMP_DIR: str = os.getenv("TMP_DIR", "/tmp/vetapp")
