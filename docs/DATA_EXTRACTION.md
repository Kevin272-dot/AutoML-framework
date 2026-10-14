# DAT.md — How the Data Extraction Engine Works

A concept-by-concept explanation of the data extraction engine: every stage, every
library it leans on, and the reasoning behind each decision. The engine lives in
`backend/app/` (`parsing/`, `discovery/`, `datasets/`, `storage.py`, `workers/`,
`security/`) with the RL training engine in `backend/rl_automl/` (see README).

The one rule that shapes everything: **the engine never extracts until a human
approves.** Resource discovery → audit → *user approves sources* → search → rank →
*user selects dataset* → download → validate → EDA → preprocess. Every stage below
sits somewhere on that chain.

---

## 0. The pipeline at a glance

```
"Find weather data for Chennai for the last 5 years"
   │
   ▼
[1] REQUIREMENT PARSER      free text → structured, reproducible requirements
[2] RESOURCE DISCOVERY      match requirements against the source registry
[3] RESOURCE AUDIT          live probes: reachable? official API? robots? terms?
[4] USER APPROVAL           hard stop — nothing is searched yet
[5] DEEP SEARCH             only approved sources, expanded queries
[6] NORMALIZATION           every source's shape → one internal schema
[7] DEDUPLICATION           conservative: exact identity only
[8] RANKING                 explainable 30/20/20/10/10/10 score
[9] PREVIEW + DETAIL        real rows from the source, retries + fallbacks
[10] USER DATASET SELECTION hard stop — download quota + API-key gates
[11] DOWNLOAD               streamed, size-capped, SHA-256 hashed
[12] VALIDATION             format, encoding, emptiness, malformed rows
[13] STORAGE                MinIO/S3 or local disk, provenance registered
[14] EDA                    read-only profiling of the real file
[15] PREPROCESSING          ML-ready copy + before/after metrics
```

Stages 11–15 run as background jobs that chain automatically; 1–10 are synchronous
API calls the frontend drives step by step.

---

## 1. Requirement parsing (`app/parsing/requirement_parser.py`)

**Purpose:** turn free text into a structured `ParsedRequirements` object — keywords,
domains, locations, date range, task type, target hints, format, size constraints —
that the rest of the engine can reason about.

**How:** pure regex + lexicons. No network, no model, no randomness.

| Concept | Implementation | Reasoning |
|---|---|---|
| Date ranges | Regex: `between X and Y`, `from X to Y`, `2020-2025`, `since 2020`, `in 2023` | The most machine-parseable signal in a query; exact string capture avoids guessing |
| Relative dates | `last/past/previous N years` → `(current_year − N, current_year)` | Users say "last 5 years" more often than explicit ranges |
| Domains | 12-domain lexicon (agriculture, climate, healthcare, …) with trigger words | Maps fuzzy topics onto the registry's domain coverage |
| Locations | ~35-entry lexicon; matched as whole words | Drives location-expanded search + ranking boosts |
| Task type | Signal phrases: "classify/churn/spam" → classification; "predict/forecast/estimate" + target nouns → regression | Detects the ML task without an LLM; feeds both search and target confirmation |
| Keywords | Content words minus stopwords, minus detected years/locations, plus matched bigrams | Feeds HF search and the keyword component of ranking |
| Formats / size | `csv`/`parquet` mentions; `at least/at most N rows` | Becomes hard filters/constraints downstream |

**Why deterministic?** Reproducible (same query → same requirements), zero latency,
zero cost, fully testable — the parser has its own unit tests. The design keeps a
single `parse_requirements()` seam so an LLM-backed parser can replace it later
without touching anything downstream. Honest limitation: lexicons miss things a
model would catch; that's the intended trade for v1.

**Location expansion** (`app/parsing/locations.py`): every location maps to parent
regions (`chennai → tamil nadu → india`). Used by deep search (so city queries find
country datasets) and ranking (so location mentions count as matches).

---

## 2. Resource discovery (`app/discovery/registry.py`)

**Purpose:** answer *"what data sources could plausibly have this data?"* — the
spec's core insight that sources are first-class, and a source is **not** a dataset.

- A seed registry of known sources (Hugging Face, Kaggle, UCI, data.gov.in, data.gov,
  EU Open Data Portal) is idempotently inserted into Postgres at first use.
- Each entry declares: base URL, source type (`DATASET_CATALOG`, `OPEN_DATA_PORTAL`,
  `REPOSITORY`…), documented API URL, robots/terms URLs, preferred access method,
  capability flags, **whether credentials are required**, and which domains it covers.
- Discovery = scoring the registry against the parsed requirements (domain overlap);
  sources with an implemented adapter always stay in the pool. No network happens here.

**Reasoning:** declaring capabilities up front (rather than discovering by scraping)
means the audit step can *verify* claims against reality, and the approval UI can
show honest cards instead of guesses.

---

## 3. Resource audit (`app/discovery/audit.py`, `app/discovery/robots.py`)

**Purpose:** before asking permission to search, verify each source is *technically*
usable and surface its *policy* posture. Uses **httpx** (async HTTP client) for all
probes — chosen for HTTP/2, timeouts, streaming, and pytest-friendly mocking (respx).

| Probe | Method | Result |
|---|---|---|
| Reachability | HEAD, fallback GET | `reachable` |
| Official API liveness | GET a cheap documented endpoint per source | `api_available`, `search_available` |
| robots.txt | GET + Python's `urllib.robotparser` | `robots_status` (FOUND/NOT_FOUND/ERROR), `robots_allowed` for the API path |
| Terms / license | HEAD/GET the declared URLs | `terms_status`, `license_status` |

**The legal-signal separation (deliberate):** robots.txt is reported as a *technical*
signal only. `policy_status` (KNOWN / UNKNOWN / RESTRICTED) is computed separately
from terms+license availability. The engine never claims "robots allows it, so it's
legal" — it just refuses to access paths robots disallows and prefers official APIs.

`technical_status` decides approvability: `API_AVAILABLE` / `DOWNLOAD_AVAILABLE` are
eligible; `AUTH_REQUIRED` (e.g. Kaggle) is only usable after a **connection** exists
(see §11); `ROBOTS_DISALLOWED` / `SOURCE_UNREACHABLE` are never approvable.

Audits are cached in `source_audits` with a TTL (`AUDIT_CACHE_TTL_SECONDS`, 1h) so we
never hammer sources repeatedly.

---

## 4. Source adapters (`app/discovery/adapters/`)

**Purpose:** all source-specific logic lives behind one interface; the engine core
never knows what source it is talking to. Three real adapters exist today:
`huggingface.py`, `eu.py` (EU Open Data Portal), and `kaggle.py` (authenticated via
stored Connections).

```python
class DatasetSourceAdapter(ABC):
    async def search(requirements) -> list[AdapterSearchResult]
    async def get_metadata(dataset_id) -> AdapterMetadata
    async def get_preview(dataset_id, limit) -> (columns, rows)
    async def get_files(dataset_id) -> list[dict]          # machine-readable files
    async def download(dataset_id, dest, max_bytes) -> dict # streamed, hashed, capped
```

**Hugging Face adapter** (`huggingface.py`) — the first complete adapter. Uses only
documented public endpoints (no scraping, no credentials):

| Endpoint | Used for |
|---|---|
| `GET /api/datasets?search=…` | catalog search |
| `GET /api/datasets/{id}` | dataset metadata |
| `datasets-server /rows /parquet /splits /statistics` | preview rows, converted files, column stats |
| `GET /api/datasets/{id}/tree/main` | original files in the repo |
| `GET /datasets/{id}/resolve/main/{path}` | official file download |

**Deep search** (`_build_queries`): the parsed requirements are expanded into up to 6
query formulations — keywords+location, keywords alone (titles rarely contain city
names), domain + synonyms (`weather → climate, temperature, rainfall, precipitation`),
and each location variant paired with the domain. Each query pulls 40 results; merges
are deduplicated by dataset id. Reasoning: single-query search is the #1 cause of
"low confidence, nothing relevant" — recall first, then let ranking sort it out.

**Resilience** (`_request_with_retry`): transient upstream failures (429/5xx) are
retried with short backoff. This matters because HF's datasets-server itself
periodically 504s — surfacing that as a hard error made datasets look broken.

**EU Open Data adapter** (`eu.py`): data.gov's CKAN API was retired (404s on every
variant — verified live), so the second adapter targets the **data.europa.eu hub
search API** (`/api/hub/search/search`, public, no auth). Its fields are multilingual
dicts (`{"en": ..., "fr": ...}`) normalized to English text; format labels arrive as
str or nested dicts (both handled). Distributions carry `download_url`/`access_url`
— tabular ones (CSV/JSON/Parquet) are previewed via direct capped download and are
what `download()` streams; landing-page-only datasets fail with an honest error.

**Kaggle adapter** (`kaggle.py`): fully authenticated via the user's stored Connection
(`username:key`, decrypted server-side only — the adapter refuses to run without one,
raising `API_KEY_REQUIRED`). Uses Kaggle API v1 with basic auth: catalog search
(`/datasets/list`), metadata (`/datasets/list/{owner}/{slug}`), and download
(`/datasets/download/{owner}/{slug}`), which returns a ZIP — the largest tabular file
inside is extracted and returned as the primary data file (CSV/Parquet/TSV). Preview
is served the same way (Kaggle has no server-side preview API). Per-source search
failures are isolated: one failing source never starves the whole search.

**Progressive degradation:** both adapters run their query ladder in order and stop
early once enough specific matches accumulate — so precise queries win, but the broad
rungs guarantee that almost any query with a recognized domain returns results.

**Preview fallback chain** (`_preview_fallback`): when the rows endpoint fails —
1. Converted parquet files (datasets-server `/parquet`), read with **pandas + pyarrow**
2. Original tabular files from the repo tree (`.csv/.parquet/.tsv/.jsonl/.json`),
   smallest first, read with pandas (CSV via `nrows=limit`)

Both paths are capped at 64 MB and produce JSON-safe cells via `_clean_value()`,
which handles numpy scalars, **numpy arrays / lists / dicts inside cells** (a real
dataset, `climate_fever`, crashed the first version here), NaN/NaT, and timestamps.
If everything fails, the error says exactly what to do — including "select the
dataset to download it instead", because preview failure ≠ download failure.

---

## 5. Normalization (`app/datasets/normalize.py`)

**Purpose:** every source speaks its own dialect; the engine speaks one. Adapter
results become the internal `DatasetMetadata` shape (name, license, owner, tags,
dates, row/column counts, format, size, columns, availability flags).

**The `raw_metadata` rule:** the source's original payload is always preserved
untouched alongside the normalized fields. Anything the normalized schema doesn't
model is never thrown away — provenance and debugging depend on it.

---

## 6. Deduplication (`app/datasets/dedupe.py`)

**Purpose:** the same dataset reachable twice (e.g. from two query formulations, or
two sources) should appear once.

**Conservative by design:** a candidate is dropped only on exact identity —
`(source, source_dataset_id)` or canonical URL (normalized). **Name similarity never
merges anything.** Reasoning: a false merge silently hides a real, possibly better
dataset; a duplicate that survives costs one table row. Asymmetric risk → strict rule.

---

## 7. Ranking (`app/datasets/rank.py`)

**Purpose:** order candidates by fit, and make every score *explainable* — the UI
shows all six components, not one opaque number.

```
score = 0.30·keyword_match + 0.20·date_match + 0.20·domain_match
      + 0.10·completeness + 0.10·size_score + 0.10·feature_quality
```

| Component | Logic | Reasoning |
|---|---|---|
| `keyword_match` | fraction of keywords **and location variants** found in name/description/tags | A "chennai" mention in the title is the strongest possible signal |
| `date_match` | overlap of the dataset's year coverage with the requested window | 0.3 (not 0) when coverage is *unknown* — most catalog datasets publish no dates; zeroing them hid relevant results |
| `domain_match` | domain/tag overlap, falling back to text mentions | Rewards on-topic datasets even without structured domains |
| `completeness` | how much of the normalized metadata the source provided | Better-documented datasets are easier to evaluate and use |
| `size_score` | log-scale between 500 and 5M rows; hard penalties outside user constraints | Log scale: 1k vs 10k rows matters more than 1M vs 2M |
| `feature_quality` | share of typed columns, penalty for ID-like columns, penalty for <4 columns | Rewards feature-rich tables, down-ranks key-value dumps |

Ties are broken by the total; components are clamped to [0,1] and returned alongside
the score so the frontend can render the breakdown.

---

## 8. Preview & dataset detail

The candidate detail page assembles: overview, **real preview rows** (§4), schema
(columns + semantic types), provenance (source, source id, URL, retrieval time,
license, access method), score breakdown, and files. Everything shown comes from a
live API call — nothing is cached as fake.

---

## 9. Selection gates (`app/api/datasets.py` → `select`)

Two guards run before any download:

1. **Quota** — `MAX_STORED_DATASETS` (default 25). Over the limit → structured
   `DATASET_LIMIT_REACHED` error; the UI links to dataset management. Reasoning:
   unbounded downloads would fill disk and invite scraping-spree behavior.
2. **Credentials** — if the source `requires_auth` and no verified **connection**
   exists → `API_KEY_REQUIRED`; the UI links to Profile → Connections. Reasoning:
   sources like Kaggle/data.gov.in require keys by design; gating here means the
   engine never fakes an attempt it cannot make.

---

## 10. Download & validation (`app/datasets/download.py`)

**Download** (adapter `download()`): streamed via httpx (`aiter_bytes`), hard-capped
at `MAX_DATASET_DOWNLOAD_BYTES` (512 MB) — the cap is enforced *during* streaming, so
an oversized file is aborted mid-flight, not downloaded then rejected. SHA-256 is
computed while streaming (single pass).

**Validation** (`validate_dataframe` + `load_dataframe`): the file is parsed with
pandas/pyarrow and checked honestly — failures raise structured errors, never silently
pass: file readable, non-empty, has columns, encoding (UTF-8/Latin-1), CSV ragged-line
ratio (>50% unparseable rows → `DATASET_MALFORMED`), duplicate columns, all-NaN rows.
Reasoning: "never hide actual parsing problems" — a dataset that survives validation
is one EDA/training can trust.

**File discovery:** `get_files()` merges datasets-server parquet conversions with the
repo's original tabular files (§4), so datasets HF never converted are still
downloadable — the download reports the real format, which drives the right parser.

---

## 11. Connections — protected secrets (`app/api/connections.py`, `app/security/secrets.py`)

**Purpose:** sources that require credentials become usable *by the user's own keys*,
stored as protected secrets — never exposed to the browser.

- **Encryption at rest:** Fernet symmetric encryption (`cryptography` package). The
  key comes from `SECRET_ENCRYPTION_KEY`; the dev profile derives a stable key from
  local settings so secrets survive restarts without manual setup (documented as a
  dev convenience).
- **Never returned:** the API stores `secret_encrypted` + a masked hint (`abcd…wxyz`).
  Decryption happens only server-side, via `get_connection_secret()`, when an adapter
  actually needs the credential.
- **Validation before storage:** HF tokens are checked against `whoami-v2`, Kaggle
  `username:key` pairs against a real authenticated API call, data.gov.in keys against
  the catalog endpoint. A rejected key is refused with the source's own reason.

---

## 12. Storage (`app/storage.py`)

**Purpose:** downloaded bytes live outside the DB, addressable and deletable.

- **S3/MinIO** via **boto3** when `USE_S3_STORAGE=true` (bucket auto-created).
- **Local filesystem** fallback (`data/artifacts/<dataset_id>/<file>`) — the dev
  profile, so the whole platform runs without Docker.
- Every artifact: storage key, backend, size, SHA-256. Deletion removes the artifact
  *and* the empty dataset directory (best-effort; a storage failure never blocks
  record deletion).

**Why not store data in Postgres:** datasets are blobs up to hundreds of MB; the DB
keeps metadata and reports (queryable), storage keeps bytes.

---

## 13. Jobs (`app/workers/`)

**Purpose:** nothing long-running blocks an HTTP request, and every stage is visible.

- **Celery + Redis** for the full stack; an **eager dispatcher** (daemon threads +
  per-thread DB sessions) for the dev profile. Identical task functions either way.
- **Chaining:** select → download → (on success) EDA → (on success) preprocessing.
  Each stage has its own `Job` row so the UI can render the pipeline.
- **State:** `QUEUED → RUNNING (stage, progress, detail) → COMPLETED | FAILED` with
  structured `error_code`/`error_message`. On startup (eager profile) in-flight jobs
  from a dead process are marked `STALE_JOB` so nothing is blocked by a ghost.
- Run files for the RL engine are written by the orchestrator itself under
  `data/automl-artifacts/runs/<run_id>/` (its own persistence; our DB mirrors status).

---

## 14. Automatic EDA (`app/datasets/eda.py`)

**Purpose:** read-only profiling of the *validated file* — the source data is never
modified. Every number the UI shows is computed by pandas/NumPy over real rows.

| Concept | How | Why it matters |
|---|---|---|
| Shape & dtypes | `df.shape`, dtype mapping | Baseline |
| Missingness | per-column count + ratio | >40% triggers a quality warning |
| Duplicates | `df.duplicated()` | Silent leakage + waste in training |
| Unique ratios | `nunique / rows` | Feeds semantic typing and ID detection |
| Distributions | 20-bin histograms (numeric), top-10 values (categorical) | Rendered as charts |
| Outliers | IQR rule (`Q1−1.5·IQR`, `Q3+1.5·IQR`) | Flagged per column, never altered here |
| Correlations | Pearson matrix, top-25 by variance for readability | Feature relationships at a glance |
| Potential IDs | name pattern (`*_id`, `key`, …) + near-unique values | Must be excluded from features |
| Constant columns | `nunique ≤ 1` | Zero signal |
| Class balance | top-class share ≥ 0.9 → imbalanced | Warns before naive training |
| **Target candidates** | binary numeric → classification (0.85); low-card int → classification (0.55); categorical 2–50 → classification (0.6); continuous numeric → regression (0.45); >50% missing → excluded; ID-like → excluded | Every candidate carries confidence + rationale so the user can *confirm* rather than trust a black box |

The confirmed choice (user gate: target + task) is written back onto the dataset.

---

## 15. Preprocessing (`app/datasets/preprocess.py`)

**Purpose:** produce an ML-ready copy of the data — the raw file is preserved, the
processed copy is a separate registered artifact — with every step recorded.

| Step | What | Why |
|---|---|---|
| Deduplicate rows | `drop_duplicates` | Exact dupes bias validation |
| Remove constant columns | `nunique ≤ 1` | Zero variance = zero signal |
| Exclude identifier-like columns | name pattern + `unique_ratio > 0.9` | IDs memorize rows; also leak |
| Impute missing | numeric → median; categorical → mode | Median resists outliers; mode preserves category space |
| Encode categoricals | one-hot ≤12 uniques; ordinal codes otherwise | One-hot explodes on high-cardinality text-ish columns; ordinal keeps the table compact (the model pipeline can do better later) |
| Outliers | IQR count reported, values untouched | Silently winsorizing destroys information; the AutoML pipeline decides handling |
| Scaling | **deferred** | Fitting a scaler is a model-pipeline concern; doing it here would leak and lock the artifact |

The report (input/output shape, missing cells before/after, what was imputed/encoded,
outlier counts, ordered step list) is stored as `PreprocessReport` and rendered as the
Preprocessing section — including a **Download preprocessed CSV** export.

---

## 16. Library choices, summarized

| Library | Role | Why this one |
|---|---|---|
| **httpx** | all external HTTP (audits, HF adapter) | Async, timeouts, streaming for capped downloads, respx-mockable in tests |
| **pandas + NumPy** | validation, EDA, preprocessing, previews | The data-engineering standard; native CSV/Parquet IO |
| **pyarrow** | Parquet read/write | Columnar, fast, what HF conversions produce |
| **boto3** | S3/MinIO artifacts | Industry-standard S3 client; MinIO is API-compatible |
| **cryptography (Fernet)** | secret encryption at rest | Authenticated encryption; keys never leave the backend |
| **Celery + Redis** | background jobs (full stack) | Durable, retryable, survives restarts; eager fallback keeps dev simple |
| **SQLAlchemy 2 + Alembic** | persistence | Declarative models, JSON/JSONB columns for reports, migrations |
| **FastAPI + Pydantic** | API + schemas | Typed contracts shared with the frontend, auto OpenAPI docs |
| **urllib.robotparser** | robots.txt evaluation | Stdlib, zero deps; correct parsing of the standard |
| **scikit-learn / XGBoost / LightGBM** | the RL engine's training stack | Brought in with the teammate's `rl_automl` package (thread-capped — see README) |
| **Torch** (lazy, optional) | PPO policy training | Only imported if a trained policy is requested; not needed for runs today |

## 17. Failure philosophy

Every failure the engine can produce is structured: a machine code
(`DATASET_MALFORMED`, `PREVIEW_UNAVAILABLE`, `API_KEY_REQUIRED`, `DATASET_LIMIT_REACHED`,
`ROBOTS_DISALLOWED`, `SOURCE_UNREACHABLE`, `STALE_JOB`, …), a human message, and — in
the UI — *what happened, why, and what the user can do*. Raw stack traces never leave
the backend. And when a capability genuinely doesn't exist (no preview, no files, no
adapter yet), the product says so instead of pretending.
