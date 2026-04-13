"""
main.py — FastAPI application entry point.

Startup sequence:
  1. Create /tmp/vetapp scratch directory.
  2. Initialise PostgreSQL connection pool.
  3. Create tables (idempotent).
  4. Begin loading the detection model (EfficientNet-V2-S) in a background thread
     so the first request is not blocked by the ~30-second GCS download.

All routers are mounted here.  CORS is opened so that a separately-hosted
frontend (Firebase, GCS bucket) can call the API.
"""

from __future__ import annotations

import os
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import PORT, TMP_DIR
from app.routers import predict, llm, webhook, dashboard


# ── Application factory ────────────────────────────────────────────────────────

app = FastAPI(
    title="Ovetra VetApp API",
    description=(
        "Unified veterinary skin-disease diagnostic API.\n\n"
        "**Two capabilities in one service:**\n"
        "- `/predict` — EfficientNet-V2-S detection + GradCAM explainability\n"
        "- `/llm/*` — Gemini 2.0 Flash treatment plan (streaming SSE)\n"
        "- `/webhook` — WhatsApp Business data collection pipeline\n"
        "- `/api/*` — Vet review dashboard (preclean, vet_queue, stats)"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
# Restrict to your real frontend origin(s) in production.
_allow_origins = os.getenv("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(predict.router)
app.include_router(llm.router)
app.include_router(webhook.router)
app.include_router(dashboard.router)


# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    # 1. Scratch dir
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(os.path.join(TMP_DIR, "models"), exist_ok=True)

    # 2. Database
    from app.database import init_pool, init_tables
    init_pool()
    init_tables()

    # 3. Load detection model in background (non-blocking)
    from app.models.detection import load_model
    threading.Thread(target=load_model, daemon=True).start()

    print("[STARTUP] VetApp API ready")


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Liveness probe for Cloud Run / load balancer."""
    return {"status": "ok"}


@app.get("/", tags=["System"])
def root():
    return {
        "service": "Ovetra VetApp API",
        "docs":    "/docs",
        "health":  "/health",
        "status":  "/status",
    }


# ── Local dev entrypoint ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=True)
