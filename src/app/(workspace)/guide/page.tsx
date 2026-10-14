"use client";

import Link from "next/link";
import { BadgeCheck, Compass, Database, Download, GitBranch, KeyRound, LineChart, Microscope, ShieldCheck } from "lucide-react";

const STEPS = [
  {
    n: "01",
    title: "Describe your ML problem",
    body: "Open Dataset Discovery and type plain language: ‘Find datasets about crop yield in India between 2020 and 2025’. The parser extracts domains, locations, date ranges, task type, and constraints — no forms, no dropdowns.",
    icon: Compass,
  },
  {
    n: "02",
    title: "Approve the sources (Gate 1)",
    body: "The engine discovers candidate data sources and audits each one live: is it reachable, does it have a documented API, what does robots.txt say, are terms published? You approve exactly which sources may be searched.",
    icon: ShieldCheck,
  },
  {
    n: "03",
    title: "Search, rank, inspect",
    body: "Approved sources are searched with expanded queries. Results are deduplicated and ranked by a six-component score you can take apart. Open any candidate for live preview rows, schema, provenance, and the score breakdown.",
    icon: LineChart,
  },
  {
    n: "04",
    title: "Select the dataset (Gate 2)",
    body: "Selecting downloads the actual file, validates it (format, encoding, malformed rows), stores it with a SHA-256 hash, runs automatic EDA, and builds a preprocessed ML-ready copy — all as a visible job chain.",
    icon: Database,
  },
  {
    n: "05",
    title: "Confirm the target",
    body: "The EDA suggests target candidates with confidence and rationale. Pick the prediction target and task (classification or regression) — or accept the suggestion.",
    icon: BadgeCheck,
  },
  {
    n: "06",
    title: "Approve pipelines (Gate 3) and train",
    body: "The RL planner proposes ranked training pipelines from a 17-model registry. Approving starts training immediately: live progress, then a leaderboard with validation/test scores and a model-comparison chart.",
    icon: GitBranch,
  },
  {
    n: "07",
    title: "Take the model anywhere",
    body: "Download the verified model.zip (self-contained inference package) or predict directly — from the built-in form or the REST API. Every model is servable the moment the run completes.",
    icon: Download,
  },
] as const;

const INTEGRATION = [
  {
    title: "Predict via the built-in API",
    body: "The backend serves trained models directly. Get the feature schema, then POST rows.",
    code: `curl http://localhost:8000/api/automl/runs/{run_id}/features

curl -X POST http://localhost:8000/api/automl/runs/{run_id}/predict \\
  -H "Content-Type: application/json" \\
  -d '{"records": [{"tenure_months": 24, "support_calls": 3}]}'`,
  },
  {
    title: "Embed the model package",
    body: "model.zip is self-contained: model + preprocessor + schema + generated loader. It runs anywhere Python runs.",
    code: `from model_package.inference.model_loader import load_model

model = load_model("model_package")
model.predict([{ "tenure_months": 24, "support_calls": 3 }])`,
  },
  {
    title: "Drive the platform as an API",
    body: "Everything the UI does is REST. Generate a typed client from the OpenAPI schema and automate the whole flow.",
    code: `npx openapi-typescript http://localhost:8000/openapi.json -o api.ts

# then: POST /api/discovery/requests → /audit → /approve-sources
# → /search → /datasets/select → /api/automl/runs → /approve`,
  },
  {
    title: "Import the RL engine as a library",
    body: "Python apps can use the engine directly with their own data — no discovery, no UI.",
    code: `from rl_automl.orchestrator import AutoMLOrchestrator, PlanRequest

orch = AutoMLOrchestrator()
run = orch.plan(PlanRequest(problem_statement="...", frame=df, target="y"))
run = orch.approve(run, selection="all")
run = orch.execute(run, frame=df)`,
  },
] as const;

export default function GuidePage() {
  return (
    <div className="mx-auto max-w-4xl space-y-10">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">How to use Intelligent AutoML</h1>
        <p className="mt-1 max-w-2xl text-sm text-muted">
          From “I need a dataset” to a trained, servable model — every stage explained,
          including every way to plug the result into your own application.
        </p>
      </div>

      {/* Workflow */}
      <section>
        <h2 className="flex items-center gap-2 text-base font-semibold">
          <Compass className="size-4 text-accent" /> The workflow
        </h2>
        <div className="mt-4 space-y-3">
          {STEPS.map(({ n, title, body, icon: Icon }) => (
            <div key={n} className="flex gap-4 rounded-lg border border-border bg-surface p-4">
              <div className="flex flex-col items-center">
                <span className="num text-xs font-semibold text-accent">{n}</span>
                <span className="mt-1 h-full w-px bg-border" aria-hidden />
              </div>
              <div className="min-w-0 pb-1">
                <div className="flex items-center gap-2">
                  <Icon className="size-3.5 text-faint" />
                  <h3 className="text-sm font-semibold text-text">{title}</h3>
                </div>
                <p className="mt-1.5 text-sm leading-relaxed text-muted">{body}</p>
              </div>
            </div>
          ))}
        </div>
        <p className="mt-4 flex items-center gap-2 text-xs text-faint">
          <ShieldCheck className="size-3.5" />
          Gates 02, 04, and 06 are hard stops: the engine structurally cannot skip them.
        </p>
      </section>

      {/* Integration */}
      <section>
        <h2 className="flex items-center gap-2 text-base font-semibold">
          <Microscope className="size-4 text-accent" /> Integrating into your application
        </h2>
        <p className="mt-1.5 text-sm text-muted">
          Four ways, from zero-code to library-level. Quick reference below — the{" "}
          <Link href="/docs/integration" className="text-accent hover:underline">
            full integration guide
          </Link>{" "}
          has literally every step: downloading requirements, error codes, batch
          predictions, wrapping the model in your own FastAPI service, and typed clients
          for any language.
        </p>
        <div className="mt-4 space-y-4">
          {INTEGRATION.map(({ title, body, code }) => (
            <div key={title} className="rounded-lg border border-border bg-surface">
              <div className="border-b border-border px-4 py-2.5">
                <h3 className="text-sm font-semibold text-text">{title}</h3>
                <p className="mt-0.5 text-xs text-muted">{body}</p>
              </div>
              <pre className="overflow-x-auto p-4 font-mono text-[11px] leading-relaxed text-muted">{code}</pre>
            </div>
          ))}
        </div>
      </section>

      {/* Credentials */}
      <section>
        <h2 className="flex items-center gap-2 text-base font-semibold">
          <KeyRound className="size-4 text-accent" /> Sources that need credentials
        </h2>
        <div className="mt-3 rounded-lg border border-border bg-surface p-4 text-sm text-muted">
          <p>
            Kaggle, data.gov.in, and Hugging Face tokens require keys. Add them under{" "}
            <strong className="text-text">Profile → Connections</strong>: each key is
            validated against the source&apos;s real API, then encrypted at rest (Fernet).
            Only a masked hint is ever visible, and decryption happens exclusively
            server-side when an adapter actually needs the credential. Connected sources
            become approvable and fully searchable.
          </p>
        </div>
      </section>

      {/* Docs */}
      <section>
        <h2 className="text-base font-semibold">Full documentation</h2>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 text-sm">
          {[
            { href: "/docs/user-guide", label: "User guide", desc: "Step-by-step usage with all examples" },
            { href: "/docs/integration", label: "Integration guide", desc: "Every instruction to run the model in an app — API, model.zip, requirements, wrappers" },
            { href: "/docs/data-extraction", label: "Data extraction", desc: "Every stage of the engine, concept by concept" },
            { href: "/docs/architecture", label: "Architecture", desc: "How the whole system fits together" },
          ].map(({ href, label, desc }) => (
            <Link
              key={href}
              href={href}
              className="rounded-lg border border-border bg-surface p-4 transition-colors hover:border-accent/40"
            >
              <div className="text-sm font-medium text-text">{label}</div>
              <div className="mt-1 text-xs text-muted">{desc}</div>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
