# Dummy Sсorign

A tiny FastAPI app that provide a **dummy credit‑scoring model** with a single `POST /predict` endpoint. Useful as a template or sandbox for ML‑model packaging, Dockerisation and Helm deployment.

## Features

* **Simple model** – logistic regression trained on a toy tabular dataset with `scikit‑learn`.
* **Fast inference API** – lightweight FastAPI + Uvicorn server.
* **Docker‑first** – single‑stage Dockerfile (\~140 MB image) for reproducible builds.
* **Kubernetes‑ready** – Helm chart in `deployment/` for easy rollout (replica count, image tag, env vars).
* **CI pipeline** – GitHub Actions workflow builds & pushes the image on every commit.

---

## Quick start

### 1. Run with Docker

```bash
# Build the image
docker build -t dummy-scoring:latest .

# Start the container
docker run -p 8000:8000 dummy-scoring:latest
```

### 2. Request a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{ "age": 30, "income": 85000, "years_employed": 5, "children": 1 }'
```

Response example:

```json
{
  "approved": true,
  "probability": 0.74
}
```

---

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Retrain the model (generates model.pkl)
python train.py

# Launch the API with live‑reload
uvicorn app:app --reload
```
---
