# ── AeroCast API — production serving image ───────────────────────────────
#
# Installs only the runtime deps needed for FastAPI + model serving.
# Training deps (torch, lightning, airflow, great-expectations, dvc) are
# deliberately excluded to keep the image lean.
#
# Build:
#   docker build -f api/Dockerfile -t aerocast-api .
#
# Run:
#   docker run -p 8000:8000 --env-file .env aerocast-api

FROM python:3.12-slim

# ── System deps ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── App user (non-root) ────────────────────────────────────────────────────
RUN useradd --create-home appuser
WORKDIR /app
USER appuser

# ── Python deps ────────────────────────────────────────────────────────────
COPY --chown=appuser:appuser pyproject.toml ./
RUN pip install --no-cache-dir --user \
    "fastapi>=0.111.0,<0.200.0" \
    "uvicorn[standard]>=0.30.0,<1.0.0" \
    "pandas>=2.2.0,<3.0.0" \
    "numpy>=1.26.0,<2.0.0" \
    "prometheus-client>=0.20.0,<1.0.0" \
    "pydantic>=2.7.0,<3.0.0" \
    "pydantic-settings>=2.3.0,<3.0.0" \
    "python-dotenv>=1.0.0,<2.0.0" \
    "wandb>=0.17.0,<1.0.0" \
    "lightgbm>=4.3.0,<5.0.0" \
    "scikit-learn>=1.5.0,<2.0.0" \
    "torch>=2.0.0,<3.0.0" \
    "lightning>=2.0.0,<3.0.0"

ENV PATH="/home/appuser/.local/bin:${PATH}"

# ── Application code ────────────────────────────────────────────────────────
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser api/ ./api/

ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
