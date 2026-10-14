# Architecture — how everything works

One platform, two engines, three user gates. This document explains how the pieces
fit, module by module. The data extraction engine has its own deep dive in
[DATA_EXTRACTION.md](DATA_EXTRACTION.md); step-by-step usage is in
[USER_GUIDE.md](USER_GUIDE.md).

---

## System overview

```
┌────────────────────────────────────────────────────────────┐
│ Next.js frontend (:3000)                                   │
│   (marketing)  landing · sign-in · sign-up                 │
│   (workspace)  dashboard · discovery · datasets ·          │
│                automl-runs · models · profile · settings   │
└──────────────┬─────────────────────────────────────────────┘
               │ REST / JSON (typed client, no any)
┌──────────────▼─────────────────────────────────────────────┐
│ FastAPI backend (:8000)                                    │
│                                                            │
│  api/          routers: discovery, datasets, connections,  │
│                automl, jobs                                │
│  parsing/      requirement parser + location expansion     │
│  discovery/    registry → audit → source adapters          │
│                (huggingface, eu, kaggle)                   │
│  datasets/     normalize → dedupe → rank → download →      │
│                eda → preprocess                            │
│  automl/       bridge to the RL engine (plan/approve/      │
│                execute/predict)                            │
│  workers/      Celery app + eager dispatcher               │
│  security/     Fernet secret encryption                    │
│  storage.py    MinIO/S3 or local filesystem                │
└──────────────┬─────────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐   ┌──────────────┐   ┌──────────────┐
    │ PostgreSQL/SQLite   │   │ Redis        │   │ MinIO / local│
    │ metadata + reports  │   │ Celery broker│   │ artifacts    │
    └─────────────────────┘   └──────────────┘   └──────────────┘

    backend/rl_automl/  ← teammate's RL-AutoML engine (library import)
```

## The three gates

| Gate | Enforced by | Why |
|---|---|---|
| Source approval | `discovery/service.py` — search only queries rows in `source_approvals` with `approved=true`, and only sources whose audit passed (or that have a verified connection) | The engine must never crawl/search without consent |
| Dataset selection | `datasets.py select` — download runs only from an explicit select, behind quota + credential gates | Downloading is a real cost (bandwidth, storage) |
| Pipeline approval | `api/automl.py approve` → `rl_automl`'s own `require_approval` — execution raises without an approved experiment list | Training costs compute; the planner proposes, the human decides |

## Backend modules

### `parsing/` — requirement understanding
`requirement_parser.py` converts free text into `ParsedRequirements` with pure
regex + lexicons (dates incl. relative ranges, domains, locations, task phrases,
format/size constraints). Deterministic = reproducible, testable, free; an LLM parser
can replace `parse_requirements()` later behind the same signature. `locations.py`
expands locations to parent regions (chennai → tamil nadu → india) for search and ranking.

### `discovery/` — sources
- `registry.py`: idempotent seed of known sources with declared capabilities,
  domain coverage, credential requirements, and adapter binding.
- `audit.py` + `robots.py`: concurrent live probes (reachability, documented API
  liveness, robots.txt via `urllib.robotparser`, terms/license URLs). robots.txt is a
  technical signal; policy status is reported separately. Results are cached with a TTL.
- `adapters/base.py`: the adapter contract (`search / get_metadata / get_preview /
  get_files / download`). Adapters use only documented APIs.
- `adapters/huggingface.py`: catalog search, metadata, datasets-server
  rows/parquet/statistics, repo tree, official resolve downloads; retry + preview
  fallback chain (rows → converted parquet → original repo files).
- `adapters/eu.py`: data.europa.eu hub search API; multilingual field normalization;
  distribution-based preview/download.
- `adapters/kaggle.py`: API v1 with basic auth from a stored Connection; zip download
  with primary-tabular extraction. Refuses to run without credentials.
- `service.py`: the discovery state machine
  (`REQUEST_CREATED → … → WAITING_FOR_SOURCE_APPROVAL → SOURCES_APPROVED →
  DATASET_SEARCHING → DATASETS_READY → …`) plus approve gating (audit + connection aware).

### `datasets/` — candidates to trained-ready data
- `normalize.py`: adapter results → internal `DatasetMetadata`; `raw_metadata` preserved.
- `dedupe.py`: exact-identity merging only (source+id, canonical URL).
- `rank.py`: the 6-component weighted score (keyword+location 30%, date 20%,
  domain 20%, completeness 10%, size 10%, feature quality 10%), every component returned.
- `download.py`: streamed, size-capped download → validation (format, encoding,
  malformed rows) → storage → registration with SHA-256.
- `eda.py`: read-only profiling (missingness, duplicates, distributions, IQR outliers,
  correlations, class balance, ID detection, target candidates with confidence).
- `preprocess.py`: dedupe → constant/ID removal → median/mode imputation →
  one-hot/ordinal encoding → outlier flags; processed parquet stored separately with a
  full report.

### `automl/` — the RL engine bridge
- `runner.py`: wraps `rl_automl.orchestrator` — `plan_run` (profile + recommend),
  `approve_run`, `execute_run` (orchestrator events → job progress), with the run's
  authoritative state in the orchestrator's artifact tree and a DB mirror
  (`automl_runs`) for the UI.
- `exec_one.py`: subprocess entry point — AutoML execution runs detached from uvicorn,
  so backend restarts never kill in-flight training; the job records its pid.
- `predictor.py`: loads the generated `model_loader.py` from a run's packaged model and
  serves `/predict` (predictions + class probabilities) and `/features` (schema for
  dynamic forms). This is the integration surface for external applications.

### `workers/` — job execution
Celery + Redis when `JOB_BACKEND=celery`; an eager dispatcher otherwise (threads for
fast jobs, **detached subprocess for AutoML runs** so restarts never kill training).
Startup recovery marks dead-process jobs `STALE_JOB` (pid-aware: live detached
subprocesses are left alone).

### `security/` — secrets
Fernet encryption for per-source credentials; validation against the real source APIs
before storage; only masked hints ever reach the client.

## Frontend structure

```
src/app/
  (marketing)/    landing (public), sign-in, sign-up — own chrome, no shell
  (workspace)/    dashboard, discovery wizard, datasets, automl-runs,
                  models, profile, settings — sidebar + topbar shell
  layout.tsx      theme (light default + dark), providers (Clerk, QueryClient)
src/components/   layout/ (sidebar, topbar), ui/ (tokens-driven primitives),
                  discovery/ (wizard, source cards, results table),
                  theme/ (provider + toggle)
src/lib/          api-client.ts (typed fetch), api-types.ts (mirrors Pydantic)
```

- Design tokens live in `globals.css` (light default + `.dark` overrides) with an
  elevation system (`--shadow-card/raised/pop`) tuned per theme.
- Data fetching is TanStack Query with polling for job/run states.
- The RL engine's own FastAPI app and CLI are intentionally unused — one service, one UI.

## Key design decisions

| Decision | Reasoning |
|---|---|
| Library import of the RL engine, not a sidecar | One deployment, one API, one auth domain; their state machine is the contract |
| Deterministic requirement parser first | Reproducible + testable; LLM hook planned behind the same interface |
| Conservative deduplication | A false merge hides a real dataset; a surviving duplicate costs one row |
| Detached subprocess for AutoML execution | Dev restarts (`--reload`) and deploys must never kill minutes-long training |
| Eager profile for local dev | The whole platform runs with zero external services; flip to Docker via env |
| Approval gates in-product, not policy docs | The engine structurally cannot skip human decisions |

## Failure handling

Structured error envelopes end to end: a machine code
(`DATASET_MALFORMED`, `PREVIEW_UNAVAILABLE`, `API_KEY_REQUIRED`,
`DATASET_LIMIT_REACHED`, `RUN_ALREADY_RUNNING`, `STALE_JOB`, …), a human message, and
actionable UI (retry, link to Connections, link to dataset management). The orchestrator
mirrors its authoritative state on failure so interrupted runs return to their approval
gate instead of dying.
