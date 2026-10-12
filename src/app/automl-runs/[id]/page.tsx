"use client";

import { use, useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { approveAutoMLRun, executeAutoMLRun, getAutoMLRun, getAutoMLRunJob } from "@/lib/api-client";
import type { AutoMLRunOut } from "@/lib/api-types";
import { Badge, StatusBadge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardHeader, MetricCard, ProgressBar, Spinner } from "@/components/ui/Card";
import { ErrorState } from "@/components/ui/States";
import { formatNumber, formatScore } from "@/lib/utils";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function RecommendationsPanel({ run, onDone, onExecute }: { run: AutoMLRunOut; onDone: () => void; onExecute: () => void }) {
  const approvedModels = new Set(run.approved);
  const alreadyApproved = approvedModels.size > 0;
  const [selected, setSelected] = useState<Set<number>>(
    () => new Set(run.recommendations.filter((r) => !approvedModels.has(r.model)).map((r) => r.rank))
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const toggle = (rank: number) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(rank)) next.delete(rank);
      else next.add(rank);
      return next;
    });

  const approve = async (selection: "all" | number[]) => {
    setBusy(true);
    setError(null);
    try {
      await approveAutoMLRun(run.run_id, selection);
      onDone();
    } catch (err) {
      setError(err as Error);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader
        title={alreadyApproved ? "Pipelines approved — ready to train" : "Recommended pipelines"}
        subtitle={
          alreadyApproved
            ? `Approved: ${run.approved.join(", ")}. Nothing is trained until you execute.`
            : "The RL/surrogate planner proposes these pipelines. Approve the ones we may train."
        }
        actions={
          !alreadyApproved ? (
            <Button size="sm" variant="secondary" onClick={() => void approve("all")} disabled={busy}>
              Approve all
            </Button>
          ) : (
            <Button size="sm" onClick={onExecute}>
              Execute now
            </Button>
          )
        }
      />
      <div className="space-y-2 p-4">
        {run.recommendations.map((rec) => {
          const isApproved = approvedModels.has(rec.model);
          return (
            <div
              key={rec.rank}
              className={[
                "rounded-md border px-3 py-2.5",
                isApproved
                  ? "border-success/40 bg-success/5"
                  : selected.has(rec.rank)
                    ? "border-accent/60 bg-accent/5"
                    : "border-border",
              ].join(" ")}
            >
              <div className="flex items-center justify-between gap-3">
                <label className="flex min-w-0 cursor-pointer items-center gap-3">
                  {!isApproved ? (
                    <input
                      type="checkbox"
                      checked={selected.has(rec.rank)}
                      onChange={() => toggle(rec.rank)}
                      className="size-3.5 accent-[var(--color-accent)]"
                      aria-label={`Approve ${rec.model}`}
                    />
                  ) : (
                    <span className="text-xs font-medium text-success">approved</span>
                  )}
                  <span className="num text-xs text-faint">#{rec.rank}</span>
                  <span className="text-sm font-medium text-text">{rec.model}</span>
                </label>
                <div className="flex shrink-0 items-center gap-2">
                  <Badge tone={rec.expected_cost === "high" ? "warning" : "neutral"}>
                    {rec.expected_cost} cost
                  </Badge>
                  {rec.expected_performance !== null ? (
                    <span className="num text-xs text-muted">
                      est. f1 {formatScore(rec.expected_performance, 3)}
                    </span>
                  ) : null}
                </div>
              </div>
              {rec.preprocessing.length > 0 ? (
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {rec.preprocessing.map((p) => (
                    <span key={p} className="rounded bg-elevated px-1.5 py-0.5 text-[10px] text-muted">
                      {p}
                    </span>
                  ))}
                </div>
              ) : null}
              {rec.reason ? <p className="mt-1.5 text-xs leading-relaxed text-faint">{rec.reason}</p> : null}
            </div>
          );
        })}

        {error ? <ErrorState what={error.message} whatToDo="Retry the approval." /> : null}

        {!alreadyApproved ? (
          <div className="flex items-center justify-between rounded-lg border border-border bg-surface px-4 py-3">
            <p className="text-xs text-muted">
              <span className="num font-medium text-text">{selected.size}</span> pipeline(s) selected
            </p>
            <Button onClick={() => void approve(Array.from(selected))} disabled={selected.size === 0 || busy}>
              {busy ? <Spinner className="border-t-text" /> : null}
              Approve selected pipelines
            </Button>
          </div>
        ) : null}
      </div>
    </Card>
  );
}

function Leaderboard({ run }: { run: AutoMLRunOut }) {
  if (run.leaderboard.length === 0) return null;
  const metric = run.leaderboard.find((e) => e.primary_metric)?.primary_metric ?? run.metric ?? "score";
  return (
    <Card>
      <CardHeader
        title="Leaderboard"
        subtitle={`Primary metric: ${metric}${run.best_model ? ` · best: ${run.best_model}` : ""}`}
      />
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-elevated/60 text-[11px] uppercase tracking-wider text-muted">
              <th className="px-3 py-2 text-left">Model</th>
              <th className="px-3 py-2 text-left">Status</th>
              <th className="px-3 py-2 text-left">Validation</th>
              <th className="px-3 py-2 text-left">Test</th>
              <th className="px-3 py-2 text-left">Train time</th>
              <th className="px-3 py-2 text-left">Inference</th>
            </tr>
          </thead>
          <tbody>
            {[...run.leaderboard]
              .sort((a, b) => (b.test_score ?? b.validation_score ?? -Infinity) - (a.test_score ?? a.validation_score ?? -Infinity))
              .map((entry, i) => (
                <tr key={`${entry.model}-${i}`} className="border-b border-border/50 last:border-0">
                  <td className="px-3 py-2">
                    <span className={i === 0 && entry.status === "success" ? "font-medium text-accent" : ""}>
                      {entry.model}
                    </span>
                  </td>
                  <td className="px-3 py-2">
                    <Badge tone={entry.status === "success" ? "success" : "danger"}>{entry.status}</Badge>
                  </td>
                  <td className="num px-3 py-2">{formatScore(entry.validation_score)}</td>
                  <td className="num px-3 py-2">{formatScore(entry.test_score)}</td>
                  <td className="num px-3 py-2 text-muted">{entry.training_time_s.toFixed(2)}s</td>
                  <td className="num px-3 py-2 text-muted">{entry.inference_time_ms.toFixed(1)}ms</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

export default function AutoMLRunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [executing, setExecuting] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const { data: run, isLoading, isError, error: loadError, refetch } = useQuery({
    queryKey: ["automl-run", id],
    queryFn: () => getAutoMLRun(id),
    refetchInterval: (q) => {
      const status = q.state.data?.status;
      if (["executing", "evaluating", "comparing", "selecting", "packaging", "queued"].includes(status ?? "")) {
        return 2000;
      }
      return false;
    },
  });

  const { data: job } = useQuery({
    queryKey: ["automl-run", id, "job"],
    queryFn: () => getAutoMLRunJob(id),
    enabled: !!run && ["queued", "executing", "evaluating", "comparing", "selecting", "packaging"].includes(run.status),
    refetchInterval: 1500,
  });

  // Refetch the run when the job finishes.
  useEffect(() => {
    if (job?.status === "COMPLETED" || job?.status === "FAILED") void refetch();
  }, [job?.status, refetch]);

  const doExecute = async () => {
    setExecuting(true);
    setError(null);
    try {
      await executeAutoMLRun(id);
      await refetch();
    } catch (err) {
      setError(err as Error);
    } finally {
      setExecuting(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 pt-20 text-sm text-muted">
        <Spinner /> Loading run…
      </div>
    );
  }
  if (isError || !run) {
    return <ErrorState what={(loadError as Error)?.message ?? "Run not found."} whatToDo="Open it from AutoML Runs." />;
  }

  const isRunning = ["queued", "executing", "evaluating", "comparing", "selecting", "packaging"].includes(run.status);
  const awaiting = run.status === "awaiting_approval";
  const complete = run.status === "complete";

  return (
    <div className="mx-auto max-w-5xl space-y-4">
      <div className="rounded-lg border border-border bg-surface p-5">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-lg font-semibold">AutoML Run</h1>
            <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs text-muted">
              <StatusBadge status={run.status.replace(/_/g, " ").toUpperCase()} />
              {run.task ? <Badge tone="accent">{run.task}</Badge> : null}
              {run.metric ? <Badge tone="neutral">metric: {run.metric}</Badge> : null}
              {run.target ? <span className="font-mono">target: {run.target}</span> : null}
            </div>
            <p className="mt-2 max-w-2xl text-xs text-muted">{run.problem_statement}</p>
            {run.planner_note ? <p className="mt-1 text-xs text-faint">Planner: {run.planner_note}</p> : null}
          </div>
          {awaiting && run.approved.length > 0 ? (
            <Button onClick={() => void doExecute()} disabled={executing}>
              {executing ? <Spinner className="border-t-text" /> : null}
              Execute approved pipelines
            </Button>
          ) : null}
        </div>
        {run.error ? (
          <ErrorState className="mt-3" title="Run failed" what={run.error} whatToDo="Adjust the pipeline selection and execute again, or start a new run." />
        ) : null}
      </div>

      {isRunning && job ? (
        <Card className="p-4">
          <div className="mb-2 flex items-center justify-between text-sm">
            <span className="flex items-center gap-2">
              {job.status === "RUNNING" ? <Spinner /> : null}
              {job.detail ?? "Working…"}
            </span>
            <StatusBadge status={job.status} />
          </div>
          <ProgressBar value={job.progress} />
          <p className="mt-2 text-xs text-faint">Stage: {job.stage?.toLowerCase() ?? "queued"}</p>
        </Card>
      ) : null}

      {complete && run.best_model ? (
        <div className="grid gap-3 md:grid-cols-3">
          <MetricCard label="Best model" value={<span className="text-base">{run.best_model}</span>} />
          <MetricCard label="Best test score" value={<span className="num">{formatScore(run.best_test_score)}</span>} hint={run.metric ?? undefined} />
          <MetricCard label="Pipelines trained" value={<span className="num">{run.leaderboard.length}</span>} />
        </div>
      ) : null}

      {awaiting ? <RecommendationsPanel run={run} onDone={() => void refetch()} onExecute={() => void doExecute()} /> : null}

      <Leaderboard run={run} />

      {run.artifacts.model_zip || run.artifacts.results_zip ? (
        <Card>
          <CardHeader
            title="Artifacts"
            subtitle={
              run.artifacts.verified
                ? "Verified packages produced by the run"
                : "Packages produced by the run"
            }
          />
          <div className="flex gap-2 p-4">
            {run.artifacts.model_zip ? (
              <a
                href={`${API_URL}/api/automl/runs/${run.run_id}/artifacts/model`}
                className="rounded-md border border-border bg-elevated px-3 py-2 text-xs font-medium text-text hover:border-accent/40"
              >
                Download model.zip
              </a>
            ) : null}
            {run.artifacts.results_zip ? (
              <a
                href={`${API_URL}/api/automl/runs/${run.run_id}/artifacts/results`}
                className="rounded-md border border-border bg-elevated px-3 py-2 text-xs font-medium text-text hover:border-accent/40"
              >
                Download results.zip
              </a>
            ) : null}
          </div>
        </Card>
      ) : null}

      <p className="text-xs text-faint">
        Trained {formatNumber(run.leaderboard.length)} pipeline(s) · run {run.run_id}
      </p>
    </div>
  );
}
