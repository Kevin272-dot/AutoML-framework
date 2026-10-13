import Link from "next/link";
import { auth } from "@clerk/nextjs/server";
import {
  ArrowRight,
  BadgeCheck,
  Compass,
  Database,
  GitBranch,
  KeyRound,
  LineChart,
  ListChecks,
  ShieldCheck,
} from "lucide-react";

const PIPELINE = [
  { label: "Describe the problem", detail: "Plain language in, structured requirements out" },
  { label: "Discover sources", detail: "Hugging Face, EU Open Data, and more" },
  { label: "Audit every source", detail: "Reachability, official APIs, robots.txt, terms" },
  { label: "Approve sources", detail: "Gate 1 — nothing is searched without you", gate: true },
  { label: "Search and rank", detail: "Expanded queries, dedup, explainable scores" },
  { label: "Select a dataset", detail: "Gate 2 — real previews before any download", gate: true },
  { label: "EDA and preprocessing", detail: "Automatic profiling and an ML-ready copy" },
  { label: "Approve pipelines", detail: "Gate 3 — the RL planner proposes, you decide", gate: true },
  { label: "Train and compare", detail: "Leaderboard, verified model packages" },
] as const;

const FEATURES = [
  {
    icon: Compass,
    title: "Discovery before training",
    body:
      "Traditional AutoML assumes you already have a dataset. This platform starts one step earlier: it finds candidate data sources for your problem, verifies them, and ranks real datasets with a score you can take apart.",
  },
  {
    icon: ListChecks,
    title: "Explainable ranking",
    body:
      "Every relevance score breaks into six published components — keyword, date coverage, domain, completeness, size, feature quality. No opaque number decides which data you train on.",
  },
  {
    icon: Database,
    title: "Real data, real provenance",
    body:
      "Previews are live rows from the source API. Downloads are validated (format, encoding, malformed rows), stored with SHA-256 hashes, and every dataset keeps its source, license, and retrieval trail.",
  },
  {
    icon: LineChart,
    title: "Automatic EDA to ML-ready",
    body:
      "The moment a dataset lands: profiling, distributions, correlations, class balance, target candidates with confidence. Then deduplication, imputation, and encoding produce a preprocessed copy — every step recorded.",
  },
  {
    icon: GitBranch,
    title: "RL-driven pipeline search",
    body:
      "A PPO agent with a 17-model registry proposes training pipelines ranked by a learned surrogate. It proposes, you approve — then the platform trains, compares, and packages the winner.",
  },
  {
    icon: KeyRound,
    title: "Your keys, protected",
    body:
      "Connect Hugging Face, Kaggle, or government portals with API keys that are validated against the source and encrypted at rest. They never touch the browser.",
  },
];

export default async function LandingPage() {
  const { userId } = await auth();
  const primaryHref = userId ? "/dashboard" : "/sign-up";
  const primaryLabel = userId ? "Open workspace" : "Start free";

  return (
    <main>
      {/* Hero — asymmetric: pitch left, the actual product flow right */}
      <section className="relative overflow-hidden border-b border-border">
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.35]"
          aria-hidden
          style={{
            background:
              "radial-gradient(60rem 30rem at 85% -10%, var(--accent-muted), transparent 60%)",
          }}
        />
        <div className="relative mx-auto grid w-full max-w-6xl gap-12 px-6 py-20 lg:grid-cols-[1.1fr_1fr] lg:py-28">
          <div>
            <p className="text-xs font-medium uppercase tracking-[0.18em] text-accent">
              Dataset discovery + AutoML
            </p>
            <h1 className="mt-4 text-4xl font-semibold leading-[1.05] sm:text-5xl lg:text-[3.4rem]">
              The right data first.
              <br />
              <span className="text-muted">Then the right model.</span>
            </h1>
            <p className="mt-6 max-w-xl text-base leading-relaxed text-muted">
              Describe your ML problem in plain language. Intelligent AutoML finds data
              sources, audits them, and ranks real datasets — with you approving every
              step — then searches training pipelines with an RL agent and hands you a
              verified model package.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link
                href={primaryHref}
                className="inline-flex items-center gap-2 rounded-md border border-accent/60 bg-accent/15 px-5 py-2.5 text-sm font-medium text-accent transition-colors hover:bg-accent/25"
              >
                {primaryLabel}
                <ArrowRight className="size-4" />
              </Link>
              <Link
                href="/discovery"
                className="inline-flex items-center gap-2 rounded-md border border-border bg-surface px-5 py-2.5 text-sm font-medium text-text transition-colors hover:border-border-strong"
              >
                <Compass className="size-4" />
                Discover datasets
              </Link>
            </div>
            <div className="mt-8 flex flex-wrap items-center gap-x-8 gap-y-3 border-t border-border pt-6">
              {[
                ["3", "approval gates"],
                ["2", "live data sources"],
                ["17", "models in the registry"],
                ["6", "published ranking factors"],
              ].map(([n, label]) => (
                <div key={label} className="flex items-baseline gap-1.5">
                  <span className="num text-lg font-semibold text-text">{n}</span>
                  <span className="text-xs text-muted">{label}</span>
                </div>
              ))}
            </div>
            <p className="mt-5 flex items-center gap-2 text-xs text-faint">
              <ShieldCheck className="size-3.5" />
              Only documented APIs. Every artifact carries provenance.
            </p>
          </div>

          {/* Proof object: the gated pipeline, rendered as the product renders it */}
          <div className="rounded-xl border border-border bg-surface p-5 shadow-[var(--shadow-raised)]">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <span className="text-xs font-medium uppercase tracking-wider text-muted">
                Discovery request
              </span>
              <span className="rounded border border-warning/40 bg-warning/10 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-warning">
                awaiting approval
              </span>
            </div>
            <div className="space-y-2 py-3">
              {PIPELINE.slice(0, 5).map((step, i) => (
                <div
                  key={step.label}
                  className={[
                    "flex items-start gap-3 rounded-md border px-3 py-2",
                    "gate" in step && step.gate
                      ? "border-accent/50 bg-accent/5"
                      : "border-transparent bg-elevated/60",
                  ].join(" ")}
                >
                  <span className="num mt-0.5 text-[10px] text-faint">{i + 1}</span>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 text-xs font-medium text-text">
                      {step.label}
                      {"gate" in step && step.gate ? (
                        <span className="rounded border border-accent/40 bg-accent/10 px-1 py-px text-[9px] font-medium uppercase tracking-wide text-accent">
                          your call
                        </span>
                      ) : null}
                    </div>
                    <div className="mt-0.5 text-[11px] text-faint">{step.detail}</div>
                  </div>
                </div>
              ))}
            </div>
            <p className="border-t border-border pt-3 text-[11px] text-faint">
              3 of 9 stages need your explicit approval. The engine cannot skip them.
            </p>
          </div>
        </div>
      </section>

      {/* How it works — the full pipeline */}
      <section id="pipeline" className="border-b border-border">
        <div className="mx-auto w-full max-w-6xl px-6 py-20">
          <h2 className="text-2xl font-semibold tracking-tight">
            A pipeline that stops for you, three times
          </h2>
          <p className="mt-3 max-w-2xl text-sm leading-relaxed text-muted">
            AutoML tools love to automate judgment calls. This one automates the labor and
            leaves the judgment to you: which sources may be searched, which dataset gets
            downloaded, and which model pipelines get trained.
          </p>
          <ol className="mt-10 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {PIPELINE.map((step, i) => (
              <li
                key={step.label}
                className={[
                  "rounded-lg border p-4",
                  "gate" in step && step.gate
                    ? "border-accent/40 bg-accent/5"
                    : "border-border bg-surface",
                ].join(" ")}
              >
                <div className="flex items-center gap-2">
                  <span className="num text-xs text-faint">{String(i + 1).padStart(2, "0")}</span>
                  <span className="text-sm font-medium text-text">{step.label}</span>
                  {"gate" in step && step.gate ? (
                    <BadgeCheck className="size-3.5 text-accent" aria-label="approval gate" />
                  ) : null}
                </div>
                <p className="mt-1.5 text-xs leading-relaxed text-muted">{step.detail}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* Features — flat, divider-separated, no card nesting */}
      <section id="features" className="border-b border-border">
        <div className="mx-auto w-full max-w-6xl px-6 py-20">
          <h2 className="text-2xl font-semibold tracking-tight">Built like a workbench</h2>
          <div className="mt-10 grid gap-x-12 gap-y-10 md:grid-cols-2">
            {FEATURES.map(({ icon: Icon, title, body }) => (
              <div key={title} className="border-t border-border pt-6">
                <div className="flex items-center gap-2.5">
                  <Icon className="size-4 text-accent" />
                  <h3 className="text-sm font-semibold text-text">{title}</h3>
                </div>
                <p className="mt-2.5 text-sm leading-relaxed text-muted">{body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section>
        <div className="mx-auto flex w-full max-w-6xl flex-col items-start gap-6 px-6 py-20 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">
              Stop hunting for datasets. Start describing problems.
            </h2>
            <p className="mt-2 max-w-xl text-sm text-muted">
              One workspace: discovery, auditing, EDA, preprocessing, and RL-driven
              training — with provenance on every artifact.
            </p>
          </div>
          <Link
            href={primaryHref}
            className="inline-flex shrink-0 items-center gap-2 rounded-md border border-accent/60 bg-accent/15 px-6 py-3 text-sm font-medium text-accent transition-colors hover:bg-accent/25"
          >
            {primaryLabel}
            <ArrowRight className="size-4" />
          </Link>
        </div>
      </section>
    </main>
  );
}
