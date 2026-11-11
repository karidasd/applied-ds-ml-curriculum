# ==========================================
# Production-Grade Multi-Stage Dockerfile
# ==========================================

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies to wheels cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime curl for container health probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create secure non-privileged user
RUN useradd -u 10001 -m -s /bin/bash appuser

# Copy installed site-packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application artifacts
COPY . /app

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

# Container Healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch ASGI Server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
