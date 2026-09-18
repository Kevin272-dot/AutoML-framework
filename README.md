# INTELLIGENT AUTOML

Dataset Discovery + Automated Machine Learning.

Describe your ML problem in natural language. The system discovers potential
data sources, audits them (reachability, official API, robots.txt, terms),
and **waits for you to approve which sources may be searched** — only then does
it search, rank, and let you select a real dataset, download and validate it,
and run automatic EDA.

> Reference docs: [`docs/CURRENT_UI_AUDIT.md`](docs/CURRENT_UI_AUDIT.md) (pre-rebuild audit)
> · Plan: `.commandcode/plans/intelligent-automl-phase1-7.md`

## Architecture

| Layer | Tech |
|---|---|
| Frontend | Next.js 16 (App Router), TypeScript, Tailwind v4, TanStack Table/Query, Recharts |
| Backend | FastAPI, Pydantic, SQLAlchemy 2, Alembic |
| DB | PostgreSQL (SQLite dev fallback) |
| Jobs | Celery + Redis (eager in-process fallback) |
| Storage | S3/MinIO (local filesystem fallback) |
| ML (this phase) | pandas, pyarrow for retrieval + EDA; scikit-learn/XGBoost/Optuna/SHAP arrive in the AutoML phase |

## Workflow (implemented)

```
NL query → requirement parsing → resource discovery → resource audit
→ USER SOURCE APPROVAL (hard gate) → dataset search (approved sources only)
→ metadata normalization → dedup → explainable ranking
→ USER DATASET SELECTION → download → validation → object storage
→ automatic EDA → target confirmation → (AutoML: next phase)
```

## Running

### With Docker (full stack)

```bash
docker compose up -d          # Postgres, Redis, MinIO

cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # defaults point at the Docker services
alembic upgrade head          # or rely on create_all on startup
uvicorn app.main:app --port 8000
celery -A app.workers.celery_app worker -l info

# frontend (repo root)
npm install
npm run dev                   # http://localhost:3000
```

### Dev profile without Docker

Backend defaults (`sqlite://`, `JOB_BACKEND=eager`, `USE_S3_STORAGE=false`)
run everything in-process with no external services:

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --port 8000
```

## Tests

```bash
cd backend && pytest -m "not live"   # full offline suite (39 tests)
pytest -m live                        # live Hugging Face smoke test
```

## API (this phase)

```
POST /api/discovery/requests                  {query} → request_id, parsed_requirements
POST /api/discovery/requests/{id}/audit       discover + audit sources, stop at approval gate
GET  /api/discovery/requests/{id}/sources     audited source cards
POST /api/discovery/requests/{id}/approve-sources
POST /api/discovery/requests/{id}/search      search approved sources only
GET  /api/discovery/requests/{id}/results     ranked candidates + score breakdown
GET  /api/datasets/candidates/{id}            full metadata
GET  /api/datasets/candidates/{id}/preview    real sample rows
POST /api/datasets/select                     download + validate + store + EDA chain
GET  /api/datasets/{id} | /files | /jobs | /eda
POST /api/datasets/{id}/confirm-target
GET  /api/jobs/{id} · /api/dashboard/stats · /api/sources · /api/datasets
```

## Principles

- **No fake data.** Unimplemented features render honest "not available yet" states.
- **Approval gates.** Sources are never searched without explicit user approval; datasets
  are never downloaded without explicit selection.
- **Explainable ranking.** Every relevance score ships with its six weighted components.
- **Provenance.** Every dataset keeps source, URL, license, access method, retrieval time, file hash.
- **robots.txt is a technical signal only.** Policy status is reported separately; no legal claims.
