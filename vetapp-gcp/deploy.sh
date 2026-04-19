#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# deploy.sh — One-command deploy to Cloud Run
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# Prerequisites:
#   • gcloud CLI authenticated:  gcloud auth login
#   • Project set:               gcloud config set project YOUR_PROJECT_ID
#   • Docker logged in:          gcloud auth configure-docker REGION-docker.pkg.dev
#   • infra/setup.sh run once:   bash infra/setup.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ID=$(gcloud config get-value project)
REGION=${REGION:-asia-south1}           # override: REGION=us-central1 ./deploy.sh
SERVICE=${SERVICE:-vetapp-api}
REPO=${REPO:-vetapp}
IMAGE_NAME=${IMAGE_NAME:-vetapp-api}
TAG=${TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%s)}

REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Deploying VetApp API"
echo "  Project : ${PROJECT_ID}"
echo "  Region  : ${REGION}"
echo "  Service : ${SERVICE}"
echo "  Image   : ${REGISTRY}:${TAG}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Build ──────────────────────────────────────────────────────────────────────
echo "[1/3] Building Docker image…"
docker build \
    --tag "${REGISTRY}:${TAG}" \
    --tag "${REGISTRY}:latest" \
    --cache-from "${REGISTRY}:latest" \
    .

# ── Push ───────────────────────────────────────────────────────────────────────
echo "[2/3] Pushing to Artifact Registry…"
docker push "${REGISTRY}:${TAG}"
docker push "${REGISTRY}:latest"

# ── Deploy ─────────────────────────────────────────────────────────────────────
echo "[3/3] Deploying to Cloud Run…"
gcloud run deploy "${SERVICE}" \
    --image "${REGISTRY}:${TAG}" \
    --region "${REGION}" \
    --platform managed \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --concurrency 10 \
    --timeout 300 \
    --set-env-vars "PORT=8080,GCS_BUCKET=${GCS_BUCKET:-vetapp-data},GCS_MODELS_PREFIX=${GCS_MODELS_PREFIX:-models/},GEMINI_MODEL=gemini-2.5-flash" \
    --update-secrets \
        "GEMINI_API_KEY=gemini-api-key:latest,\
WA_ACCESS_TOKEN=wa-access-token:latest,\
WA_APP_SECRET=wa-app-secret:latest,\
WA_VERIFY_TOKEN=wa-verify-token:latest,\
WA_PHONE_ID=wa-phone-id:latest,\
VET_NUMBERS=vet-numbers:latest,\
DATABASE_URL=vetapp-db-url:latest" \
    --add-cloudsql-instances "${PROJECT_ID}:${REGION}:vetapp-db"

URL=$(gcloud run services describe "${SERVICE}" \
    --region "${REGION}" \
    --format "value(status.url)")

echo ""
echo "✅  Deployed successfully!"
echo "    URL : ${URL}"
echo "    Docs: ${URL}/docs"
echo ""
echo "Set your WhatsApp webhook URL to: ${URL}/webhook"
