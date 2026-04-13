# ── Stage 1: dependency builder ───────────────────────────────────────────────
# Using a builder stage keeps the final image lean: we install wheels here
# then copy only the installed packages, not pip's cache.
FROM python:3.11-slim AS builder

WORKDIR /build

# System libraries needed by OpenCV and psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libpq-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch wheels from the official index first,
# then the rest of the dependencies from PyPI.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.3.0 torchvision==0.18.0 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system libraries (no compiler, no headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# App code
WORKDIR /app
COPY app/ ./app/

# Cloud Run injects PORT; default to 8080
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV TMP_DIR=/tmp/vetapp

# Non-root user for security
RUN useradd -m -u 1000 vetapp && chown -R vetapp /app
USER vetapp

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", "--timeout-keep-alive", "120"]
