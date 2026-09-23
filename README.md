# INTELLIGENT AUTOML

**Dataset Discovery + Automated Machine Learning.**

You don't need to already have a dataset. Describe your ML problem in plain language:

> *"Find datasets about weather in Chennai for the last 5 years"*

The system discovers potential **data sources**, audits them (reachability, official API,
robots.txt, terms), and **stops — waiting for you to approve which sources may be searched**.
Only then does it search those sources, rank real datasets with an explainable score, let you
inspect and select one, download and validate the actual data, run automatic EDA, preprocess
it into an ML-ready copy, and finally train models via an RL-driven pipeline-search engine —
whose proposed pipelines you also approve before anything trains.

- Full extraction-engine deep dive: [`DAT.md`](DAT.md)
- Pre-rebuild audit: [`docs/CURRENT_UI_AUDIT.md`](docs/CURRENT_UI_AUDIT.md)

---

## The three approval gates

The product is built around explicit human control. Nothing happens without you:

| Gate | What you approve | Where |
|---|---|---|
| 1. Source approval | Which data sources the system may search | Discovery wizard → source cards |
| 2. Dataset selection | Which dataset gets downloaded | Candidate detail → "Select dataset" |
| 3. Pipeline approval | Which model pipelines get trained | AutoML run → recommended pipelines |

---

## Architecture

```
Next.js (localhost:3000)
        │  REST / JSON
        ▼
FastAPI backend (localhost:8000)
        │
        ├── Requirement parser          (deterministic NL extraction)
        ├── Resource registry + audit   (httpx probes, robots.txt, capabilities)
        ├── Source adapters             (Hugging Face: search/metadata/preview/download)
        ├── Normalize → dedup → rank    (explainable 30/20/20/10/10/10 scoring)
        ├── Download → validate → store (CSV/Parquet → MinIO/S3 or local disk)
        ├── EDA + Preprocessing         (pandas; read-only over raw data)
        ├── Connections                 (Fernet-encrypted per-source API keys)
        └── RL-AutoML engine            (rl_automl: PPO pipeline search, 17 models)
                │
                ▼
   Celery + Redis (job workers)  ·  PostgreSQL  ·  MinIO
```

| Layer | Tech | Notes |
|---|---|---|
| Frontend | Next.js 16, TypeScript, Tailwind v4, TanStack Table/Query, Recharts, Clerk | App Router, bright-default theme with dark preference |
| Backend | FastAPI, Pydantic, SQLAlchemy 2 | One service; the RL engine is imported as a library, not a sidecar |
| DB | PostgreSQL / SQLite fallback | `automl_runs` mirrors RL-engine state into our DB |
| Jobs | Celery + Redis / eager in-process fallback | Download → EDA → preprocessing → AutoML chains |
| Storage | MinIO (S3 API) / local filesystem | Every artifact hashed (SHA-256) and versioned by key |
| ML | pandas, NumPy, pyarrow, scikit-learn, XGBoost, LightGBM | Torch/PPO is a lazy dependency — see "RL engine" below |

## End-to-end flow

```
NL query
  → requirement parsing (keywords, domain, location, date range, task, format, size)
  → resource discovery (registry match) + resource audit (live probes)
  → GATE 1: approve sources
  → deep search of approved sources only (synonym + location expanded queries)
  → metadata normalization → deduplication → explainable ranking
  → dataset detail (real preview rows, schema, provenance, score breakdown)
  → GATE 2: select dataset
  → download → validation → object storage → automatic EDA → preprocessing
  → target confirmation (suggested target + confidence + alternatives)
  → AutoML run: plan (recommend pipelines) → GATE 3: approve pipelines
  → execute (train/compare/select) → leaderboard → model.zip / results.zip
```

## Feature map

**Dataset discovery**
- Deterministic NL parser (dates incl. "last N years", domains, locations with region
  expansion, task phrases, format/size constraints) — swap-in LLM parser later
- Registry of known sources with live capability auditing and robots.txt evaluation
- Deep HF search: synonym + location-expanded query formulations, 40 results per query
- Explainable ranking (every score ships with its six weighted components)
- Resilient preview: retries on transient 5xx, falls back to converted parquet, then to
  original repo files; handles numpy-array cells; honest "unavailable" errors otherwise

**Dataset workspace**
- Real preview rows, schema, statistics, quality warnings, provenance, score breakdown
- Automatic EDA: missingness, duplicates, distributions, correlations, class balance,
  potential-ID detection, target candidates with confidence + rationale
- Preprocessing: dedupe, constant/ID removal, median/mode imputation, one-hot/ordinal
  encoding, IQR outlier flags — with before/after metrics and a full applied-steps report
- CSV export (raw + preprocessed), SHA-256 provenance, per-artifact validation reports

**Trust & safety**
- Download quota (`MAX_STORED_DATASETS`, default 25) with a structured, actionable error
- Per-source API keys (Kaggle, data.gov.in, HF tokens) encrypted at rest with Fernet;
  only masked hints ever reach the browser; keys validated against the real source APIs
- Deletion is explicit two-step; removing a dataset deletes artifacts + EDA + reports
  but keeps the discovery candidate for re-selection

**AutoML (RL engine)**
- PPO agent + surrogate planner propose ranked pipelines from a 17-model registry
- sklearn-first: runs work today with the surrogate planner; PPO activates once a policy
  checkpoint is trained (`rl_automl/training/train_rl.py`) — torch is a lazy import
- Leaderboard with validation/test scores, training + inference times; verified
  `model.zip` (self-contained inference package) and `results.zip` downloads

## Running

### 1. Frontend environment

```bash
cp .env.example .env.local    # add your Clerk keys (see below)
```

**Clerk** (required for auth): create an app at [dashboard.clerk.com](https://dashboard.clerk.com),
copy the keys into `.env.local`. Sign-up: `/sign-up` · Sign-in: `/sign-in` · Profile &
preferences (theme, API-key connections): `/profile`. Without keys the app runs anonymously.

### 2. Infra + backend (full stack)

```bash
docker compose up -d                    # Postgres, Redis, MinIO

cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                    # points at the Docker services
uvicorn app.main:app --port 8000 --reload
celery -A app.workers.celery_app worker -l info
```

### 3. Frontend

```bash
npm install --include=dev
npm run dev                             # http://localhost:3000
```

### Dev profile without Docker

Backend defaults (`sqlite://`, `JOB_BACKEND=eager`, `USE_S3_STORAGE=false`) run everything
in-process with zero external services — EDA/ML jobs execute in background threads:

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --port 8000 --reload
```

## API reference

```
# Discovery (gate 1)
POST /api/discovery/requests                    {query} → request_id, parsed_requirements
POST /api/discovery/requests/{id}/audit         discover + audit sources, stop at gate
GET  /api/discovery/requests/{id}/sources       audited source cards (capabilities + policy)
POST /api/discovery/requests/{id}/approve-sources  {source_ids[]}
POST /api/discovery/requests/{id}/search        approved sources only
GET  /api/discovery/requests/{id}/results       ranked candidates + score breakdown

# Datasets (gate 2)
GET  /api/datasets/candidates/{id}              full metadata + provenance
GET  /api/datasets/candidates/{id}/preview      real sample rows
POST /api/datasets/select                       quota + API-key gates, then download chain
GET  /api/datasets/{id} · /files · /jobs · /eda · /preprocess
POST /api/datasets/{id}/confirm-target          {target, task}
POST /api/datasets/{id}/preprocess              re-run preprocessing on demand
GET  /api/datasets/{id}/export?variant=raw|processed   CSV download
DELETE /api/datasets/{id}                       artifacts + records (candidate kept)

# Connections (protected secrets)
GET  /api/connections                           masked list
POST /api/connections                           {source_id, secret} — validated, encrypted
DELETE /api/connections/{source_id}

# AutoML (gate 3, RL engine)
POST /api/automl/runs                           {dataset_id, target, task} → plan
GET  /api/automl/runs · /runs/{id} · /runs/{id}/job
POST /api/automl/runs/{id}/approve              {selection: "all" | ranks | names}
POST /api/automl/runs/{id}/execute              train approved pipelines
GET  /api/automl/runs/{id}/artifacts/{model|results}   verified ZIP downloads

# Misc
GET  /api/jobs/{id} · /api/dashboard/stats · /api/sources · /api/health
```

## Configuration (backend env)

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./automl.db` | Postgres via `postgresql+psycopg://…` with Docker |
| `JOB_BACKEND` | `eager` | `celery` requires Redis |
| `USE_S3_STORAGE` | `false` | MinIO/S3 artifact storage |
| `S3_ENDPOINT_URL` / `S3_ACCESS_KEY` / `S3_SECRET_KEY` / `S3_BUCKET` | MinIO defaults | Object storage |
| `SECRET_ENCRYPTION_KEY` | derived (dev) | Fernet key for connection secrets — set a real one in production |
| `MAX_STORED_DATASETS` | `25` | Download quota |
| `MAX_DATASET_DOWNLOAD_BYTES` | 512 MB | Per-file download cap |
| `HF_API_BASE` / `HF_DATASETS_SERVER_BASE` | HF defaults | Documented public endpoints |

## Tests

```bash
cd backend
pytest -m "not live"     # 55 offline tests (parser, robots, adapters, ranking,
                         # validation, EDA, preprocessing, deletion, AutoML e2e)
pytest -m live           # live Hugging Face smoke test
```

## Project structure

```
backend/
  app/
    api/            discovery, datasets, connections, automl, jobs routers
    automl/         rl_automl adapter (plan/approve/execute bridge)
    parsing/        requirement parser + location expansion
    discovery/      registry, audit engine, robots, source adapters (huggingface)
    datasets/       normalize, dedupe, rank, download, eda, preprocess
    workers/        Celery app, task definitions, eager dispatcher
    security/       Fernet secret encryption for connections
  rl_automl/        teammate's RL-AutoML engine (agent, environment, execution,
                    search, packaging) — integrated as a library
  configs/          engine default config
  tests/            our test suite
src/                Next.js app (app/, components/, lib/)
docs/               audit + reference docs
DAT.md              how the data extraction engine works, concept by concept
```

## Principles

- **No fake data.** Unimplemented features render honest "not available yet" states.
- **Three approval gates.** Sources, datasets, and training pipelines all require you.
- **Explainable ranking.** No unexplained numbers — every score breaks down into components.
- **Provenance.** Source, URL, license, access method, retrieval time, SHA-256 per artifact.
- **robots.txt is a technical signal only.** Policy status is reported separately; no legal claims.
- **Only documented APIs.** The HF adapter uses official REST endpoints — never HTML scraping.
