# AeroCast Airflow image
# Extends the official Apache Airflow image with the project's training
# dependencies so DAG tasks can import aerocast.* without a separate install step.
#
# Build:
#   docker build -t aerocast-airflow:latest -f airflow/Dockerfile .
#
# The project source (src/) is volume-mounted at runtime (see docker-compose.yml),
# so code changes are reflected without rebuilding the image.

FROM apache/airflow:2.11.0-python3.12

# Switch to root only for system packages (none needed here beyond base image)
USER airflow

# Install CPU-only PyTorch first (avoids pulling the large CUDA wheel in CI)
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining training-group dependencies
RUN pip install --no-cache-dir \
    lightning==2.6.5 \
    lightgbm==4.3.0 \
    scikit-learn==1.5.0 \
    wandb==0.17.0 \
    evidently==0.4.0 \
    great-expectations==0.18.0 \
    "dvc[s3]==3.51.0" \
    dagshub==0.3.0 \
    pandas==2.2.0 \
    numpy==1.26.0 \
    requests==2.32.0 \
    python-dotenv==1.0.0 \
    pydantic==2.7.0 \
    pydantic-settings==2.3.0 \
    prometheus-client==0.20.0

# src/ is mounted at /opt/airflow/project/src — add to PYTHONPATH so that
# `import aerocast` works inside tasks without installing the package.
ENV PYTHONPATH="/opt/airflow/project/src:${PYTHONPATH}"
