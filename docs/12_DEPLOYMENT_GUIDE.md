# 12. Deployment and Cloud Hosting Guide

This guide details operations and deployment procedures for YieldGuard services across local container orchestration, public demos, and cloud container hosting.

YieldGuard is an offline benchmark demonstrator on public datasets, not a production fab system. On the UCI SECOM holdout no detector beat chance (see the README), so any hosted instance is a demo of the engineering, not a working fault predictor.

---

## 1. Container Architecture Overview

YieldGuard provides dual containerized services:
1. **Inference API (`api`)**: FastAPI service running via Uvicorn on port `8000`. Serves single-run scoring (`/score`), batch processing, model metadata (`/model-info`), Phase II SPC charting (`/spc`), and health probes (`/health`, `/ready`).
2. **Operations Dashboard (`dashboard`)**: Interactive Streamlit web console on port `8501`. Serves 6 diagnostic and process control views with Plotly interactive charts.

---

## 2. Local Multi-Container Orchestration (`docker compose`)

### Quick Start
To build and launch both services on an internal bridge network:

```bash
# Build and start both API and Dashboard in background
docker compose up -d --build

# Verify container health status
docker compose ps

# Check logs
docker compose logs -f
```

### Verification & Health Probes
* **API Readiness**: `curl http://localhost:8000/ready` -> `{"status": "ready", "bundle_loaded": true, "bundle_version": "v0.1.0"}`
* **API Health**: `curl http://localhost:8000/health` -> `{"status": "healthy", ...}`
* **Dashboard Access**: Open browser at `http://localhost:8501`
* **Automated API Smoke Test**: `python scripts/smoke_api.py --url http://localhost:8000`

### Teardown
```bash
docker compose down
```

---

## 3. Public Web Demo: Hugging Face Spaces

To deploy the interactive Streamlit dashboard as a free public web application on Hugging Face Spaces:

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Select **Streamlit** as the Space SDK and choose public visibility.
3. Push the repository files to the Space repository:
   * Keep `app/dashboard.py` as the entrypoint (or configure `app_file: app/dashboard.py` in `README.md` YAML frontmatter).
   * Ensure `requirements.lock`, `src/`, `models/v0.1.0/`, and `reports/` are pushed.
4. The dashboard will automatically launch in **Synthetic Demo Mode** (or with bundled holdout reports) without requiring the raw UCI SECOM download (a public dataset, CC BY 4.0).

---

## 4. Cloud Container Deployment: Google Cloud Run & AWS ECS

The API has no authentication and no rate limiting, so deploy it as a private service and use the dashboard (section 3) as the public demo. Cloud hosting is a paid resource; `--min-instances 0` lets the service scale to zero when idle.

### Google Cloud Run (Serverless Container)

```bash
# 1. Build and submit container image to Google Artifact Registry
gcloud builds submit --tag gcr.io/$PROJECT_ID/yieldguard-api:latest .

# 2. Deploy as a private service (callers need the Cloud Run invoker role) that scales to zero
gcloud run deploy yieldguard-api \
  --image gcr.io/$PROJECT_ID/yieldguard-api:latest \
  --platform managed \
  --region us-central1 \
  --no-allow-unauthenticated \
  --port 8000 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --set-env-vars YIELDGUARD_BUNDLE_DIR=/app/models/v0.1.0
```

### AWS ECS / Fargate (Container Tasks)
* **Task Definition**: Use `yieldguard:ci` built from `Dockerfile`.
* **Port Mapping**: Container port `8000` mapped to Target Group.
* **Health Check Path**: `/ready` with 10s interval and 2 retries.
* **Grace Period**: 15 seconds to allow model bundle integrity verification and initialization.

---

## 5. Security & Isolation Controls

1. **Non-Root User Execution**: Both `Dockerfile` and `Dockerfile.dashboard` enforce non-root execution under UID `1000` (`appuser`).
2. **Payload Size Limits**: Strict middleware in `api/main.py` enforces maximum request payload size of 256 KiB for `/score` and 2 MiB for `/spc`.
3. **Bundle Tamper Evidence**: Model artifacts are verified via SHA-256 against `metadata.json` on startup. If any weight or configuration is altered, the service refuses initialization and reports `HTTP 503`.
4. **Raw Data Exclusion**: Raw wafer and SECOM datasets are never copied into the Docker images. The dashboard image does include `reports/` and the split manifest.
5. **No Authentication or Rate Limiting**: The API accepts requests from any caller that can reach it. Keep it on a private network or behind the platform's identity layer; do not expose it directly to the internet.
