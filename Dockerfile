# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install security updates and curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy lock and setup files
COPY requirements.lock pyproject.toml README.md /app/

# Install locked dependencies and package
RUN pip install --no-cache-dir -r requirements.lock

# Copy source code and application
COPY src /app/src
COPY api /app/api
COPY configs /app/configs
COPY models/v0.1.0 /app/models/v0.1.0

RUN pip install --no-cache-dir --no-deps -e .

# Switch to non-root user
USER appuser

EXPOSE 8000

ENV YIELDGUARD_BUNDLE_DIR=/app/models/v0.1.0 \
    PYTHONUNBUFFERED=1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
