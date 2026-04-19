#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# deploy.sh — One-command deploy to Cloud Run
#
# Default mode: Cloud Build (recommended for Cloud Shell — no local Docker push)
# Local mode:   LOCAL=1 ./deploy.sh  (requires Docker + Artifact Registry auth)
#
# Prerequisites:
#   • gcloud CLI authenticated:  gcloud auth login
#   • Project set:               gcloud config set project YOUR_PROJECT_ID
#   • infra/setup.sh run once:   bash infra/setup.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ID=$(gcloud config get-value project 2>/dev/null || true)
if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo ""
  echo "❌  GCP project is not set."
  echo "    Run:  gcloud config set project YOUR_PROJECT_ID"
  echo "    Then re-run this script."
  exit 1
fi

REGION=${REGION:-asia-south1}
SERVICE=${SERVICE:-vetapp-api}
REPO=${REPO:-vetapp}
IMAGE_NAME=${IMAGE_NAME:-vetapp-api}
TAG=${TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%s)}
LOCAL=${LOCAL:-0}          # set LOCAL=1 to use local Docker build+push instead

REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Deploying VetApp API"
echo "  Project : ${PROJECT_ID}"
echo "  Region  : ${REGION}"
echo "  Service : ${SERVICE}"
echo "  Image   : ${REGISTRY}:${TAG}"
echo "  Mode    : $([ "${LOCAL}" = "1" ] && echo 'local Docker' || echo 'Cloud Build')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "${LOCAL}" == "1" ]]; then
  # ── Local mode: build + push with local Docker daemon ─────────────────────
  echo "[1/3] Building Docker image (local)…"
  docker build \
      --tag "${REGISTRY}:${TAG}" \
      --tag "${REGISTRY}:latest" \
      --cache-from "${REGISTRY}:latest" \
      .

  echo "[2/3] Pushing to Artifact Registry…"
  docker push "${REGISTRY}:${TAG}"
  docker push "${REGISTRY}:latest"

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

else
  # ── Cloud Build mode (default): build + push + deploy run entirely on GCP ──
  # No local Docker push needed — avoids Cloud Shell networking issues.
  echo "[1/1] Submitting to Cloud Build (build + push + deploy)…"
  gcloud builds submit \
      --config cloudbuild.yaml \
      --substitutions "_REGION=${REGION},_SERVICE=${SERVICE},_REPO=${REPO},_IMAGE=${IMAGE_NAME},_TAG=${TAG}" \
      --project "${PROJECT_ID}" \
      .
fi

# ── Print the service URL ─────────────────────────────────────────────────────
URL=$(gcloud run services describe "${SERVICE}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --format "value(status.url)" 2>/dev/null || echo "")

if [[ -n "${URL}" ]]; then
  echo ""
  echo "✅  Deployed successfully!"
  echo "    URL      : ${URL}"
  echo "    Docs     : ${URL}/docs"
  echo "    Dashboard: ${URL}/dashboard"
  echo ""
  echo "Set your WhatsApp webhook URL to: ${URL}/webhook"
fi
