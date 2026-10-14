# User Guide

Everything you can do with Intelligent AutoML, from finding a dataset to serving a
trained model inside your own application.

---

## Part 1 — Find a dataset

1. **Describe your problem.** From the landing page, click **Discover datasets** and type
   a plain-language request:
   > *"Find datasets about crop yield in India between 2020 and 2025"*
   > *"Telecom customer churn classification data"*
   > *"Weather data for Chennai for the last 5 years"*
2. **Watch the pipeline run.** The wizard parses your request, discovers candidate
   sources, and audits each one live (reachability, official API, robots.txt, terms).
3. **Approve sources (Gate 1).** Source cards show capabilities (API, search, metadata,
   preview, download) and policy posture. Only approvable sources can be selected.
   Click **Approve selected sources**.
4. **Search and rank.** The engine searches only your approved sources and returns
   ranked candidates. Click the ⓘ icon on any score to see its six-component breakdown.
5. **Inspect a candidate.** Open **Details** for: overview, live preview rows from the
   source, schema, files, provenance (source, URL, license, retrieval time), and the
   score breakdown.

## Part 2 — Get the data

6. **Select the dataset (Gate 2).** Click **Select dataset**. The platform downloads the
   actual file, validates it (format, encoding, malformed rows), stores it with a
   SHA-256 hash, runs automatic EDA, and builds a preprocessed ML-ready copy — all as a
   visible job chain.
7. **Review the EDA.** Shape, missing values, duplicates, distributions, correlations,
   class balance, quality warnings, and suggested target candidates with confidence.
8. **Confirm the target.** Pick the prediction target and task (or accept the
   suggestion). Click **Confirm target**.
9. **Optional tools on the dataset page:**
   - **Download CSV** / **Download preprocessed CSV** — take the data anywhere
   - **Run preprocessing** — rebuild the ML-ready copy on demand
   - **Delete** — two-step confirmation removes artifacts + reports (the discovery
     candidate stays for re-selection)

## Part 3 — Train (Gate 3)

10. **Start AutoML run.** After confirming the target, click **Start AutoML run**. The
    RL planner profiles the dataset and proposes ranked training pipelines.
11. **Approve pipelines (Gate 3).** Review each proposal (model, preprocessing steps,
    estimated score, cost, rationale). Approve all or a selection — **training starts
    immediately on approval**.
12. **Watch it run.** Live stage progress: executing → evaluating → comparing →
    selecting → packaging.
13. **Read the leaderboard.** Validation and test scores, training time, and inference
    latency per pipeline, with the best model highlighted. Download the verified
    **model.zip** (self-contained inference package) and **results.zip**.

## Part 4 — Integrate into your application

### A. Serve predictions directly (no code to write)

Open the run page and use the model you just trained. The backend serves the packaged
model itself:

```bash
# 1. Get the feature schema (build a form from this)
curl http://localhost:8000/api/automl/runs/{run_id}/features

# 2. Predict — send rows matching the feature names
curl -X POST http://localhost:8000/api/automl/runs/{run_id}/predict \
  -H "Content-Type: application/json" \
  -d '{"records": [{"tenure_months": 24, "support_calls": 3, "monthly_charges": 70.5}]}'

# Response:
# {"predictions": [0], "count": 1,
#  "probabilities": [[0.69, 0.31]], "classes": ["0", "1"]}
```

The **Models** page in the app renders this as a form for you: pick a model, fill the
feature fields, predict.

### B. Embed the model package in your own code

Download **model.zip** from the run page — it is self-contained (model + preprocessor +
schema + a generated loader):

```bash
unzip model.zip && pip install -r model_package/requirements.txt
```

```python
from model_package.inference.model_loader import load_model

model = load_model("model_package")
result = model.predict([
    {"tenure_months": 24, "support_calls": 3, "monthly_charges": 70.5}
])
print(result)          # predictions, decoded to original labels
print(model.predict_proba)  # also available
```

Works in any Python app — FastAPI, Flask, Django, a notebook, a batch job — with only
the requirements from the package installed.

### C. Use the platform as an API from any language

The whole product is a REST API (OpenAPI schema at `http://localhost:8000/openapi.json`).
Generate a typed client and drive the flow programmatically:

```bash
npx openapi-typescript http://localhost:8000/openapi.json -o api-types.ts
```

```python
import requests

BASE = "http://localhost:8000/api"

req = requests.post(f"{BASE}/discovery/requests", json={"query": "churn data"}).json()
requests.post(f"{BASE}/discovery/requests/{req['id']}/audit")
sources = requests.get(f"{BASE}/discovery/requests/{req['id']}/sources").json()
requests.post(f"{BASE}/discovery/requests/{req['id']}/approve-sources",
              json={"source_ids": [s["id"] for s in sources]})
requests.post(f"{BASE}/discovery/requests/{req['id']}/search")
results = requests.get(f"{BASE}/discovery/requests/{req['id']}/results").json()
```

### D. Embed the RL engine as a Python library

If your app has its own data and only wants the AutoML engine:

```python
import pandas as pd
from rl_automl.orchestrator import AutoMLOrchestrator, PlanRequest

orch = AutoMLOrchestrator(listener=lambda e: print(e.stage, e.message))
run = orch.plan(PlanRequest(problem_statement="...", frame=df, target="y"))
run = orch.approve(run, selection="all")
run = orch.execute(run, frame=df)
```

This is exactly how the platform's backend integrates the engine.

## Part 5 — Sources that need credentials

Kaggle, data.gov.in, and Hugging Face tokens require credentials:

1. **Profile → Connections** — pick the source, paste the key
   (Kaggle format: `username:key` from kaggle.com/settings → API)
2. The key is **validated against the source's real API**, then encrypted at rest.
   Only a masked hint (`abcd…wxyz`) is ever visible.
3. Approve the connected source in your next discovery — it becomes searchable, and
   its datasets become downloadable with your credentials (decrypted server-side only).

## Troubleshooting

| Symptom | What it means | What to do |
|---|---|---|
| "Cannot reach the backend service" | FastAPI not running on :8000 | Start `uvicorn app.main:app --port 8000` |
| Preview fails with `PREVIEW_UNAVAILABLE` | Source has no machine-readable files or its preview service is down | Try again later, or select the dataset — download may still work |
| `API_KEY_REQUIRED` on select | Source needs credentials | Add a key in Profile → Connections |
| `DATASET_LIMIT_REACHED` | Download quota hit | Delete datasets you no longer need |
| Run bounces to approval | Backend restarted mid-run (dev) | Runs execute in a detached subprocess and survive restarts — just execute again; the stale job is cleaned up automatically |
