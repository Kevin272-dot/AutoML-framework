# Current UI / Repository Audit

Reference document captured before the Phases 1–7 rebuild
(see `.commandcode/plans/intelligent-automl-phase1-7.md` for the approved plan).

Snapshot date: 2026-09-18 · Branch: `main` @ `ee29ee3`

---

## 1. Repository overview

| Item | State |
|---|---|
| Project type | Next.js frontend **only** |
| Framework | Next.js 16.3.4 (App Router) + React 19.2.8 + TypeScript (strict) |
| Package manager | npm |
| Styling | **No CSS framework.** All styles are a single inline `<style>` string (`GLOBAL_CSS`) in `src/app/layout.tsx` |
| State/data fetching | None (plain `fetch` in `src/lib/lib.ts`) |
| Backend | **None.** No Python, no FastAPI, no database, no Redis/Celery, no object storage |
| Tests | None |
| CI/CD | None |
| Auth | None |

## 2. Route inventory

| Route | File | State |
|---|---|---|
| `/` | `src/app/page.tsx` | Working. Natural-language search form (query + optional start/end year) with 3 example prompts. Navigates to `/search`. |
| `/search` | `src/app/search/page.tsx` | Working. Calls `POST /api/search`, renders result cards (name, description, domain badge, rows/columns/dates/missing/format, "suitability %"). Card grid, not a data table. |
| `/datasets/[id]` | `src/app/datasets/[id]/page.tsx` | **Empty (0 bytes).** |
| `/automl/configure/[id]` | `src/app/automl/configure/[id]/page.tsx` | **Empty (0 bytes).** |
| `/automl/training/[jobId]` | `src/app/automl/training/[jobId]/page.tsx` | **Empty (0 bytes).** |
| `/results/[jobId]` | `src/app/results/[jobId]/page.tsx` | **Empty (0 bytes).** |
| `layout.tsx` | `src/app/layout.tsx` | Sticky glassmorphic navbar ("IDPAutoML"), inline CSS design tokens. |

## 3. API client (`src/lib/lib.ts`)

A typed fetch wrapper pointing at `http://127.0.0.1:8000` (`NEXT_PUBLIC_API_URL`), mirroring a
Pydantic backend named `backend_app.py` **that does not exist in this repository**.

Endpoints referenced (all currently unreachable):

- `POST /api/search` → `SearchResponse`
- `GET /api/datasets/{id}` / `GET /api/datasets/{id}/preview` / `POST /api/datasets/select`
- `POST /api/automl/run`, `GET /api/jobs/{jobId}`, `GET /api/results/{jobId}`, `POST /api/predict`

Types defined: `DatasetOut`, `SearchResponse`, `DatasetPreview`, `JobStatusOut`,
`LeaderboardEntry`, `ResultsOut`, `PredictResponse`.

The wrapper pattern (typed generic `request<T>`, error-on-non-OK) is sound and was the one
salvageable piece — it is superseded by `src/lib/api-client.ts` + `src/lib/api-types.ts`
in the rebuild.

## 4. What the existing flow does (and lacks)

Current implied flow: `search → pick dataset → configure → train → results`.

It **completely lacks** the product's defining workflow:

- No requirement parsing (no structured extraction of domain/location/dates/task)
- No resource discovery, resource audit, or **user source approval** gate
- No source adapters, no dataset metadata normalization/dedup/ranking with explainable scores
- No dataset detail workspace (schema/statistics/quality/files/provenance)
- No real data retrieval, validation, EDA
- No app shell (sidebar/topbar), dashboard, projects, or job-state model
- Search results are an unexplained single "suitability %" with no score breakdown

## 5. Visual style assessment (why the redesign)

Current style conflicts with the target "professional ML workspace":

- Neon gradient primary buttons (`accent → accent2` 135°) with glow/translate hover
- Glassmorphic sticky navbar (`backdrop-filter: blur`)
- Oversized hero (2.2rem) and card-heavy marketing feel
- Card grid for datasets instead of a dense sortable data table
- Inline CSS string: no tokens, no component system, hard to maintain

Target style (per spec §39–41): restrained dark-neutral surfaces, single restrained accent,
subtle 1px borders, high information density, professional tables, no glow/gradients.

## 6. Disposition

| Existing piece | Decision |
|---|---|
| `package.json` / Next.js / TS setup | **Keep**; add Tailwind v4, lucide-react, @tanstack/react-table, @tanstack/react-query, recharts |
| `src/app/page.tsx` | **Rebuild** as Discovery entry (same core UX idea, new design system) |
| `src/app/search/page.tsx` | **Rebuild** into the multi-step discovery wizard + results data table |
| 4 empty route files | **Replace** with real implementations / honest "not available yet" states |
| `src/lib/lib.ts` | **Supersede** with `api-client.ts` + `api-types.ts` matching the new FastAPI OpenAPI contract |
| Inline `GLOBAL_CSS` | **Delete**; replaced by `globals.css` design tokens + Tailwind |
| Neon theme | **Delete** |
