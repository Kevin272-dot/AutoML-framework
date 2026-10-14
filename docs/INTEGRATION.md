# Integration Guide — every step, every path

Complete instructions for taking a model trained in Intelligent AutoML and using it
inside your own application. Four paths, from zero-code to library-level. Pick the one
that matches your stack; all of them start from the same place.

> Prerequisites for everything: the backend is running (`uvicorn app.main:app --port
> 8000` from `backend/`) and you have completed an AutoML run (status `complete` on the
> run page). Each run has an id like `run_8c1bd5ad0740` — replace `{run_id}` below.

---

## Path 0 — What the run gives you

Every completed run produces three servable things:

| Artifact | Endpoint | What it is |
|---|---|---|
| Feature schema | `GET /api/automl/runs/{run_id}/features` | The exact input columns the model expects, with types |
| Prediction API | `POST /api/automl/runs/{run_id}/predict` | Live predictions served by the backend |
| `model.zip` | `GET /api/automl/runs/{run_id}/artifacts/model` | Self-contained package to run anywhere |

Inspect them in the UI: **AutoML Runs → your run**, or **Models → Predict** (built-in form).

### Step 0.1 — Get the feature schema

```bash
curl http://localhost:8000/api/automl/runs/{run_id}/features
```

Response:

```json
{
  "task_type": "classification",
  "target": "churn",
  "features": [
    { "name": "tenure_months", "kind": "numeric", "required": true },
    { "name": "support_calls", "kind": "numeric", "required": true },
    { "name": "monthly_charges", "kind": "numeric", "required": true }
  ],
  "classes": ["0", "1"]
}
```

`features[].name` is what your requests must contain. `kind` tells you the input type
(`numeric` → numbers, `categorical`/`text` → strings). `classes` are the possible
outputs for classification tasks.

---

## Path 1 — Predict via the API (zero setup)

No downloads. Your app calls the backend exactly like the built-in form does.

### Step 1.1 — Single prediction

```bash
curl -X POST http://localhost:8000/api/automl/runs/{run_id}/predict \
  -H "Content-Type: application/json" \
  -d '{"records": [{"tenure_months": 24, "support_calls": 3, "monthly_charges": 70.5}]}'
```

Response:

```json
{
  "predictions": [0],
  "count": 1,
  "probabilities": [[0.694265, 0.305735]],
  "classes": ["0", "1"]
}
```

- `predictions` — the model's output, decoded to the original labels when applicable
- `probabilities` — per-class confidence (classification models only)
- `records` accepts **multiple rows** at once (up to 1000) — batch predictions in one call

### Step 1.2 — From JavaScript / TypeScript

```ts
const BASE = "http://localhost:8000/api";

export async function predict(runId: string, records: Record<string, unknown>[]) {
  const res = await fetch(`${BASE}/automl/runs/${runId}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ records }),
  });
  if (!res.ok) throw new Error(`prediction failed: ${res.status}`);
  return res.json() as Promise<{
    predictions: (string | number)[];
    probabilities?: number[][];
    classes?: string[];
  }>;
}

// in your app:
const out = await predict("run_8c1bd5ad0740", [{ tenure_months: 24, support_calls: 3 }]);
console.log(out.predictions[0]);
```

### Step 1.3 — From Python

```python
import requests

BASE = "http://localhost:8000/api/automl/runs"

def predict(run_id: str, records: list[dict]) -> dict:
    resp = requests.post(f"{BASE}/{run_id}/predict", json={"records": records}, timeout=30)
    resp.raise_for_status()
    return resp.json()

print(predict("run_8c1bd5ad0740", [{"tenure_months": 24, "support_calls": 3}]))
```

### Step 1.4 — Error codes you may see

| Code | Meaning | Fix |
|---|---|---|
| 404 | No packaged model for this run | The run never completed — check the run page |
| 400 `PREDICT_FAILURE` | Rows don't match the schema | Compare against `/features`; missing/extra columns are named in the message |
| 400 (missing columns) | Input rows lack required features | Every `features[].name` must be present |

---

## Path 2 — Embed the model package (offline, no backend dependency)

The most portable path: the model runs inside *your* process, no network calls.

### Step 2.1 — Download and unpack

```bash
# via the UI: Run page → "Download model.zip"
curl -o model.zip http://localhost:8000/api/automl/runs/{run_id}/artifacts/model
unzip model.zip -d inference
```

The unpacked tree:

```
model_package/
├── model/model.joblib              # the trained model (joblib pickle)
├── preprocessing/preprocessor.joblib  # the fitted preprocessing pipeline
├── metadata/feature_schema.json    # expected input columns + types
├── metadata/model_metadata.json    # task, metric, scores, label mapping
├── metadata/dataset_metadata.json  # where the training data came from
├── inference/model_loader.py       # generated loader (the documented entry point)
├── inference/predict.py            # CLI: python predict.py --input rows.csv
├── inference/example_input.csv     # example rows in the exact expected format
├── runtime/                        # the loader's own dependencies, bundled
└── requirements.txt                # pip install this
```

### Step 2.2 — Install its requirements

```bash
pip install -r model_package/requirements.txt
```

Typical contents: `scikit-learn`, `pandas`, `numpy`, and the model's own library
(`xgboost`/`lightgbm`) — versions pinned to match the pickle.

### Step 2.3 — Predict from your Python application

```python
from model_package.inference.model_loader import load_model

# load once at app startup (it deserializes the model + preprocessor)
model = load_model("model_package")

print(model.describe())
# {'model': 'xgboost', 'task': 'classification', 'primary_metric': 'f1',
#  'validation_score': 0.98, 'test_score': 0.97, 'n_features': 3,
#  'classes': ['0', '1']}

# single row (dict) — extra columns are ignored, order does not matter
prediction = model.predict({"tenure_months": 24, "support_calls": 3, "monthly_charges": 70.5})

# batch (list of dicts)
predictions = model.predict([
    {"tenure_months": 24, "support_calls": 3, "monthly_charges": 70.5},
    {"tenure_months": 3,  "support_calls": 8, "monthly_charges": 95.0},
])

# probabilities (classification models)
probabilities = model.predict_proba([
    {"tenure_months": 24, "support_calls": 3, "monthly_charges": 70.5},
])
```

### Step 2.4 — Wrap it in your own FastAPI/Flask service

```python
from fastapi import FastAPI
from pydantic import BaseModel
from model_package.inference.model_loader import load_model

app = FastAPI()
model = load_model("model_package")

class Row(BaseModel):
    tenure_months: float
    support_calls: float
    monthly_charges: float

@app.post("/predict")
def predict(rows: list[Row]):
    preds = model.predict([r.model_dump() for r in rows])
    proba = model.predict_proba([r.model_dump() for r in rows])
    return {"predictions": preds.tolist(), "probabilities": proba.tolist()}
```

Run: `uvicorn myservice:app --port 9000` — your app now serves the model independently.

### Step 2.5 — CLI (no code at all)

```bash
python model_package/inference/predict.py --input model_package/inference/example_input.csv
```

---

## Path 3 — Drive the whole platform as an API

Automate discovery → selection → training from another service or script.

### Step 3.1 — The full lifecycle

```python
import requests, time

BASE = "http://localhost:8000/api"

# 1. describe the problem
req = requests.post(f"{BASE}/discovery/requests",
                    json={"query": "telecom churn classification data"}).json()
rid = req["id"]

# 2. audit sources (runs live probes)
requests.post(f"{BASE}/discovery/requests/{rid}/audit")

# 3. approve the sources you want searched
sources = requests.get(f"{BASE}/discovery/requests/{rid}/sources").json()
requests.post(f"{BASE}/discovery/requests/{rid}/approve-sources",
              json={"source_ids": [s["id"] for s in sources if s["audit"]["technical_status"] == "API_AVAILABLE"]})

# 4. search
requests.post(f"{BASE}/discovery/requests/{rid}/search")
while True:
    job = requests.get(f"{BASE}/discovery/requests/{rid}").json()
    if job["status"] == "DATASETS_READY":
        break
    time.sleep(2)

# 5. pick a candidate (top-ranked) and select it
candidates = requests.get(f"{BASE}/discovery/requests/{rid}/results").json()
sel = requests.post(f"{BASE}/datasets/select", json={"candidate_id": candidates[0]["id"]}).json()
dataset_id = sel["dataset_id"]

# 6. wait for download + EDA + preprocessing to finish
while True:
    jobs = requests.get(f"{BASE}/datasets/{dataset_id}/jobs").json()
    if all(j["status"] in ("COMPLETED", "FAILED") for j in jobs):
        break
    time.sleep(2)

# 7. confirm target (or read the EDA suggestion first)
requests.get(f"{BASE}/datasets/{dataset_id}/eda")
requests.post(f"{BASE}/datasets/{dataset_id}/confirm-target",
              json={"target": "churn", "task": "classification"})

# 8. plan, approve, execute — training starts on approval
run = requests.post(f"{BASE}/automl/runs",
                    json={"dataset_id": dataset_id, "target": "churn", "task": "classification"}).json()
requests.post(f"{BASE}/automl/runs/{run['run_id']}/approve", json={"selection": "all"})
while True:
    run = requests.get(f"{BASE}/automl/runs/{run['run_id']}").json()
    if run["status"] in ("complete", "failed"):
        break
    time.sleep(3)
print(run["best_model"], run["best_test_score"])
```

### Step 3.2 — Generate a typed client for any language

```bash
# TypeScript
npx openapi-typescript http://localhost:8000/openapi.json -o api-types.ts

# Python
python -m openapi_client_gen http://localhost:8000/openapi.json   # or use openapi-python-client

# Any other language: the schema at /openapi.json works with every OpenAPI generator
```

---

## Path 4 — Import the RL engine as a Python library

For applications that have their **own data** and want the AutoML engine only.

### Step 4.1 — Make the package importable

The engine lives in `backend/rl_automl`. Either add `backend/` to your `PYTHONPATH`:

```bash
export PYTHONPATH="/path/to/AutoML-framework/backend:$PYTHONPATH"
```

or copy `backend/rl_automl` + `backend/configs` into your project.

### Step 4.2 — Requirements

```bash
pip install scikit-learn xgboost lightgbm pandas numpy pyyaml psutil
# optional, only for PPO policy training:
pip install torch
```

### Step 4.3 — Use it

```python
import pandas as pd
from rl_automl.orchestrator import AutoMLOrchestrator, PlanRequest

df = pd.read_csv("your_data.csv")

def on_event(event):
    print(f"{event.stage}: {event.message}")

orch = AutoMLOrchestrator(listener=on_event)

# plan: task classification, profiling, pipeline recommendations (nothing trains)
run = orch.plan(PlanRequest(
    problem_statement="Predict customer churn from account activity",
    frame=df,
    dataset_name="my_churn_data",
    target="churn",                 # or let the task classifier infer it
    task_type="classification",     # or "regression"
    use_policy=False,               # True once a PPO policy is trained
))

# approve: YOUR gate — pass "all", ranks [1, 2], or model names
run = orch.approve(run, selection="all", approved_by="me")

# execute: trains, evaluates, compares, packages
run = orch.execute(run, frame=df)

print(run.status)                      # RunStatus.COMPLETE
print(run.best_model.model, run.best_model.test_score)
print(run.artifacts.model_zip)         # path to the verified model package
```

### Step 4.4 — Dependencies of each stage

| Stage | Needs | Notes |
|---|---|---|
| plan | pandas, numpy, pyyaml | No torch required (`use_policy=False`) |
| execute | scikit-learn, xgboost, lightgbm, psutil | Thread-capped by the platform config |
| packaging | stdlib zipfile + joblib | Produces verified `model.zip` |
| PPO training | torch (2.x) | `python -m rl_automl.training.train_rl` — separate, opt-in |

---

## Common questions

**Where are my models stored?** Under `backend/data/automl-artifacts/runs/<run_id>/` —
the orchestrator's own artifact tree (`package/model_package/`, `model_package.zip`,
`results.zip`, `run.json`).

**Can two apps use the same model?** Yes — Path 1 (shared backend API) for many apps,
or download `model.zip` (Path 2) into each app separately.

**How do I retrain when data changes?** Run the flow again for the same dataset
(select it from Datasets) — the previous run and its models stay available.

**Is there authentication on the prediction API?** The backend itself is unauthenticated
this phase (the UI is Clerk-protected). Put it behind your own gateway/token check for
production, or bind it to localhost and access it via your app's server only.
