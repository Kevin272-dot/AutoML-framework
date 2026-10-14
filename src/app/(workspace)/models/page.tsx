"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { Download, Microscope } from "lucide-react";
import { getRunFeatures, listAutoMLRuns } from "@/lib/api-client";
import { Badge, StatusBadge } from "@/components/ui/Badge";
import { ButtonLink } from "@/components/ui/Button";
import { Card, CardHeader, MetricCard, Spinner } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/States";
import { formatDate, formatScore } from "@/lib/utils";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function ModelsPage() {
  const { data: runs, isLoading, isError, error } = useQuery({
    queryKey: ["automl-runs"],
    queryFn: listAutoMLRuns,
    refetchInterval: 10_000,
  });

  const trained = (runs ?? []).filter(
    (r) => r.status === "complete" && r.best_model
  );
  const packagable = trained.filter((r) => r.artifacts.model_zip);

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 py-16 text-sm text-muted">
        <Spinner /> Loading models…
      </div>
    );
  }
  if (isError) {
    return <EmptyState title="Could not load models" description={(error as Error)?.message} />;
  }

  return (
    <div className="mx-auto max-w-5xl space-y-4">
      <div>
        <h1 className="text-lg font-semibold tracking-tight">Models</h1>
        <p className="mt-0.5 text-sm text-muted">
          Trained models from AutoML runs. Each one is packaged, verified, and servable —
          download the package or predict directly from this page.
        </p>
      </div>

      {trained.length === 0 ? (
        <EmptyState
          title="No trained models yet"
          description="Complete an AutoML run (discover a dataset, confirm the target, approve pipelines) and the best model appears here."
          action={{ href: "/discovery", label: "Start dataset discovery" }}
        />
      ) : (
        <>
          <div className="grid gap-3 md:grid-cols-3">
            <MetricCard label="Trained models" value={<span className="num">{trained.length}</span>} />
            <MetricCard
              label="Verified packages"
              value={<span className="num">{packagable.filter((r) => r.artifacts.verified).length}</span>}
              hint="model.zip verified by the engine"
            />
            <MetricCard
              label="Servable"
              value={<span className="num">{packagable.length}</span>}
              hint="predict via API or the built-in form"
            />
          </div>

          <Card>
            <CardHeader title="All models" />
            <div className="divide-y divide-border">
              {trained.map((run) => (
                <div key={run.run_id} className="flex items-center justify-between gap-4 px-4 py-3">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-sm font-medium text-text">{run.best_model}</span>
                      {run.task ? <Badge tone="accent">{run.task}</Badge> : null}
                      {run.artifacts.verified ? <Badge tone="success">verified</Badge> : null}
                    </div>
                    <div className="mt-0.5 truncate text-xs text-faint">
                      target: {run.target ?? "—"} · metric: {run.metric ?? "—"} · {formatDate(run.updated_at)}
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <span className="num text-xs text-muted">
                      test {formatScore(run.best_test_score)}
                    </span>
                    <ButtonLink href={`/models/${run.run_id}`} size="sm">
                      <Microscope className="size-3.5" /> Predict
                    </ButtonLink>
                    {run.artifacts.model_zip ? (
                      <a
                        href={`${API_URL}/api/automl/runs/${run.run_id}/artifacts/model`}
                        className="inline-flex items-center gap-1.5 rounded-md border border-border bg-elevated px-2.5 py-1.5 text-xs font-medium text-text hover:border-accent/40"
                        title="Download model.zip"
                      >
                        <Download className="size-3.5" /> model.zip
                      </a>
                    ) : null}
                    <Link
                      href={`/automl-runs/${run.run_id}`}
                      className="rounded-md border border-border bg-elevated px-2.5 py-1.5 text-xs text-muted hover:text-text"
                    >
                      Run
                    </Link>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
