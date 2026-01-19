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

# Copy v2 application code
COPY v2/. ./

# Copy algotrading project config and package source code
# Work in an isolated directory to avoid conflicts
WORKDIR /app/algotrading
COPY pyproject.toml ./pyproject.toml
COPY src/ ./src
# Install local package from repository (algotrading)
RUN python -m pip install .
# Switch back to v2 working directory
WORKDIR /app/v2

# Default: run the app (override with `--config ...` at runtime)
CMD ["python", "run_app.py", "--config", "config/app.yml"]
