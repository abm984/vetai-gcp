#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# infra/setup.sh — One-time GCP project setup
#
# Run this ONCE before the first deploy.  It is safe to re-run (idempotent).
#
# Usage:
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID
#   bash infra/setup.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ID=$(gcloud config get-value project)
REGION=${REGION:-asia-south1}
DB_INSTANCE=${DB_INSTANCE:-vetapp-db}
DB_NAME=${DB_NAME:-vetapp}
DB_USER=${DB_USER:-vetapp}
GCS_BUCKET=${GCS_BUCKET:-vetapp-data}
REPO_NAME=${REPO_NAME:-vetapp}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  VetApp GCP Setup"
echo "  Project: ${PROJECT_ID}"
echo "  Region : ${REGION}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Enable APIs ────────────────────────────────────────────────────────────────
echo "[1/7] Enabling required APIs…"
gcloud services enable \
    run.googleapis.com \
    sqladmin.googleapis.com \
    storage.googleapis.com \
    secretmanager.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    --project "${PROJECT_ID}"

# ── Artifact Registry ─────────────────────────────────────────────────────────
echo "[2/7] Creating Artifact Registry repository…"
gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format docker \
    --location "${REGION}" \
    --description "VetApp Docker images" \
    --project "${PROJECT_ID}" 2>/dev/null || echo "  (already exists)"

# ── GCS bucket ────────────────────────────────────────────────────────────────
echo "[3/7] Creating GCS bucket: gs://${GCS_BUCKET}…"
gcloud storage buckets create "gs://${GCS_BUCKET}" \
    --location "${REGION}" \
    --uniform-bucket-level-access \
    --project "${PROJECT_ID}" 2>/dev/null || echo "  (already exists)"

# Create logical GCS "folders"
for PREFIX in models/ dataset/ preclean/ vet_queue/ rejected/ incoming/; do
    echo "  mkdir gs://${GCS_BUCKET}/${PREFIX}"
    echo "" | gcloud storage cp - "gs://${GCS_BUCKET}/${PREFIX}.keep" \
        --project "${PROJECT_ID}" 2>/dev/null || true
done

# ── Cloud SQL (PostgreSQL 15) ──────────────────────────────────────────────────
echo "[4/7] Creating Cloud SQL instance (PostgreSQL 15)…"
echo "  NOTE: This may take 5–10 minutes on first run."
gcloud sql instances create "${DB_INSTANCE}" \
    --database-version POSTGRES_15 \
    --tier db-f1-micro \
    --region "${REGION}" \
    --storage-type SSD \
    --storage-size 10GB \
    --no-backup \
    --project "${PROJECT_ID}" 2>/dev/null || echo "  (already exists)"

echo "[4b] Creating database and user…"
gcloud sql databases create "${DB_NAME}" \
    --instance "${DB_INSTANCE}" \
    --project "${PROJECT_ID}" 2>/dev/null || echo "  (already exists)"

DB_PASS=$(LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 24)
gcloud sql users create "${DB_USER}" \
    --instance "${DB_INSTANCE}" \
    --password "${DB_PASS}" \
    --project "${PROJECT_ID}" 2>/dev/null || {
    echo "  (user already exists — reusing existing password)"
    # Try to get existing password from Secret Manager
    DB_PASS=$(gcloud secrets versions access latest \
        --secret vetapp-db-url \
        --project "${PROJECT_ID}" 2>/dev/null \
        | grep -oP 'postgresql://[^:]+:\K[^@]+' || echo "UNKNOWN")
}

DB_URL="postgresql://${DB_USER}:${DB_PASS}@/${DB_NAME}?host=/cloudsql/${PROJECT_ID}:${REGION}:${DB_INSTANCE}"

# ── Secret Manager ────────────────────────────────────────────────────────────
echo "[5/7] Storing secrets in Secret Manager…"

_create_secret() {
    local NAME=$1
    local VALUE=$2
    gcloud secrets create "${NAME}" --replication-policy automatic \
        --project "${PROJECT_ID}" 2>/dev/null || true
    echo -n "${VALUE}" | gcloud secrets versions add "${NAME}" \
        --data-file - --project "${PROJECT_ID}"
    echo "  ✓ ${NAME}"
}

_create_secret "vetapp-db-url" "${DB_URL}"

echo ""
echo "  ── Secrets requiring YOUR values ──"
echo "  The following secrets are created with placeholder values."
echo "  Replace them in Secret Manager → Console before deploying:"
echo ""
for SECRET in gemini-api-key wa-access-token wa-app-secret wa-verify-token wa-phone-id vet-numbers; do
    gcloud secrets create "${SECRET}" --replication-policy automatic \
        --project "${PROJECT_ID}" 2>/dev/null || true
    echo -n "REPLACE_ME" | gcloud secrets versions add "${SECRET}" \
        --data-file - --project "${PROJECT_ID}" 2>/dev/null || true
    echo "  ⚠  ${SECRET}  (set real value before deploying)"
done

# ── IAM for Cloud Run service account ─────────────────────────────────────────
echo "[6/7] Granting IAM permissions to Cloud Run service account…"
SA="${PROJECT_ID}@appspot.gserviceaccount.com"
# Use the Compute Engine default SA that Cloud Run uses by default
SA="$(gcloud iam service-accounts list \
    --filter "displayName:Compute Engine default service account" \
    --format "value(email)" --project "${PROJECT_ID}")"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${SA}" \
    --role roles/storage.objectAdmin --quiet
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${SA}" \
    --role roles/secretmanager.secretAccessor --quiet
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${SA}" \
    --role roles/cloudsql.client --quiet

# ── Upload model files reminder ───────────────────────────────────────────────
echo "[7/7] Model file upload instructions…"
echo ""
echo "  Upload your trained model files to GCS before the first deploy:"
echo ""
echo "  # PyTorch detection model (from vetai/):"
echo "  gcloud storage cp vetai/best_model.pth gs://${GCS_BUCKET}/models/"
echo ""
echo "  # Keras k-fold ensemble (from Arch/):"
echo "  for FOLD in 1 2 3 4 5; do"
echo "    gcloud storage cp Arch/dog_fold\${FOLD}.keras gs://${GCS_BUCKET}/models/"
echo "    gcloud storage cp Arch/cat_fold\${FOLD}.keras gs://${GCS_BUCKET}/models/"
echo "  done"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅  Setup complete!"
echo "    Cloud SQL instance : ${PROJECT_ID}:${REGION}:${DB_INSTANCE}"
echo "    GCS bucket         : gs://${GCS_BUCKET}"
echo "    Artifact Registry  : ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"
echo ""
echo "Next steps:"
echo "  1. Set real values for secrets in Secret Manager (Console or CLI)."
echo "  2. Upload model files to GCS (see above)."
echo "  3. Run:  bash deploy.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
