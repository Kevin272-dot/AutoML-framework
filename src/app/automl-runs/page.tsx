"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { listAutoMLRuns } from "@/lib/api-client";
import { Badge, StatusBadge } from "@/components/ui/Badge";
import { ButtonLink } from "@/components/ui/Button";
import { Card, CardHeader, Spinner } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/States";
import { formatDate, formatScore } from "@/lib/utils";

export default function AutoMLRunsPage() {
  const { data: runs, isLoading, isError, error } = useQuery({
    queryKey: ["automl-runs"],
    queryFn: listAutoMLRuns,
    refetchInterval: 5000,
  });

  return (
    <div className="mx-auto max-w-5xl space-y-4">
      <div>
        <h1 className="text-lg font-semibold tracking-tight">AutoML Runs</h1>
        <p className="mt-0.5 text-sm text-muted">
          RL-driven pipeline search runs over your confirmed datasets.
        </p>
      </div>

      {isLoading ? (
        <div className="flex items-center gap-2 py-16 text-sm text-muted">
          <Spinner /> Loading runs…
        </div>
      ) : isError ? (
        <EmptyState title="Could not load runs" description={(error as Error)?.message} />
      ) : !runs || runs.length === 0 ? (
        <EmptyState
          title="No AutoML runs yet"
          description="Confirm a target on a downloaded dataset, then start a run — the planner will propose pipelines for your approval."
          action={{ href: "/datasets", label: "Open datasets" }}
        />
      ) : (
        <Card>
          <CardHeader title={`${runs.length} run(s)`} />
          <div className="divide-y divide-border">
            {runs.map((run) => (
              <Link
                key={run.run_id}
                href={`/automl-runs/${run.run_id}`}
                className="flex items-center justify-between gap-4 px-4 py-3 hover:bg-hover/50"
              >
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium text-text">
                    {run.problem_statement || run.run_id}
                  </div>
                  <div className="mt-0.5 flex items-center gap-2 text-xs text-faint">
                    {run.task ? <Badge tone="accent">{run.task}</Badge> : null}
                    {run.target ? <span className="font-mono">target: {run.target}</span> : null}
                    <span>{formatDate(run.created_at)}</span>
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-3 text-xs">
                  {run.best_model ? (
                    <span className="text-muted">
                      best: <span className="font-medium text-text">{run.best_model}</span>
                      {run.best_test_score !== null ? (
                        <span className="num"> ({formatScore(run.best_test_score)})</span>
                      ) : null}
                    </span>
                  ) : null}
                  <StatusBadge status={run.status.replace(/_/g, " ").toUpperCase()} />
                </div>
              </Link>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
