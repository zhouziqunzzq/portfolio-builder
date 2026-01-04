# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/v2

# Install Python dependencies first (better layer caching)
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# Copy application code
COPY . ./

# Default: run the Alpaca prod config (override with `--config ...` at runtime)
CMD ["python", "run_app.py", "--config", "config/app_prod_alpaca.yml"]
