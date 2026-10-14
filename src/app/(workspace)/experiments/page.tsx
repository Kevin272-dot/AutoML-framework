"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { listAutoMLRuns } from "@/lib/api-client";
import type { AutoMLRunOut } from "@/lib/api-types";
import { Badge, StatusBadge } from "@/components/ui/Badge";
import { ButtonLink } from "@/components/ui/Button";
import { Card, CardHeader, Spinner } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/States";
import { formatDate, formatScore } from "@/lib/utils";

/** An "experiment" is one discovery request taken through to trained pipelines.
 * Today that maps 1:1 to an AutoML run; the page groups them by dataset. */
export default function ExperimentsPage() {
  const { data: runs, isLoading, isError, error } = useQuery({
    queryKey: ["automl-runs"],
    queryFn: listAutoMLRuns,
    refetchInterval: 10_000,
  });

  const grouped = new Map<string, AutoMLRunOut[]>();
  for (const run of runs ?? []) {
    const existing = grouped.get(run.dataset_id) ?? [];
    grouped.set(run.dataset_id, [...existing, run]);
  }
  const experiments = Array.from(grouped.entries());

  return (
    <div className="mx-auto max-w-5xl space-y-4">
      <div>
        <h1 className="text-lg font-semibold tracking-tight">Experiments</h1>
        <p className="mt-0.5 text-sm text-muted">
          Datasets taken through the AutoML pipeline, grouped with all their runs.
        </p>
      </div>

      {isLoading ? (
        <div className="flex items-center gap-2 py-16 text-sm text-muted">
          <Spinner /> Loading experiments…
        </div>
      ) : isError ? (
        <EmptyState title="Could not load experiments" description={(error as Error)?.message} />
      ) : experiments.length === 0 ? (
        <EmptyState
          title="No experiments yet"
          description="Confirm a target on a downloaded dataset and start an AutoML run — it appears here grouped by dataset."
          action={{ href: "/datasets", label: "Open datasets" }}
        />
      ) : (
        <div className="space-y-4">
          {experiments.map(([datasetId, datasetRuns]) => {
            const completed = datasetRuns.filter((r) => r.status === "complete");
            const best = completed
              .map((r) => r.best_test_score ?? -Infinity)
              .reduce((a, b) => Math.max(a, b), -Infinity);
            const bestModel = completed.find((r) => r.best_test_score === best)?.best_model;
            return (
              <Card key={datasetId}>
                <CardHeader
                  title={`Dataset ${datasetId.slice(0, 8)}…`}
                  subtitle={`${datasetRuns.length} run(s) · ${completed.length} completed`}
                  actions={
                    <div className="flex items-center gap-2 text-xs">
                      {bestModel ? (
                        <span className="text-muted">
                          best: <span className="font-medium text-text">{bestModel}</span>
                          {isFinite(best) ? <span className="num"> ({formatScore(best)})</span> : null}
                        </span>
                      ) : null}
                      <ButtonLink href={`/datasets/${datasetId}`} size="sm" variant="secondary">
                        Dataset
                      </ButtonLink>
                    </div>
                  }
                />
                <div className="divide-y divide-border">
                  {datasetRuns.map((run) => (
                    <Link
                      key={run.run_id}
                      href={`/automl-runs/${run.run_id}`}
                      className="flex items-center justify-between gap-4 px-4 py-2.5 hover:bg-hover/50"
                    >
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 text-xs">
                          <span className="font-mono text-faint">{run.run_id}</span>
                          {run.best_model ? <Badge tone="accent">{run.best_model}</Badge> : null}
                          {run.target ? <span className="text-faint">target: {run.target}</span> : null}
                        </div>
                        <div className="mt-0.5 text-[11px] text-faint">{formatDate(run.created_at)}</div>
                      </div>
                      <div className="flex shrink-0 items-center gap-3">
                        <span className="num text-xs text-muted">
                          {run.leaderboard.length} pipeline(s)
                        </span>
                        <StatusBadge status={run.status.replace(/_/g, " ").toUpperCase()} />
                      </div>
                    </Link>
                  ))}
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
