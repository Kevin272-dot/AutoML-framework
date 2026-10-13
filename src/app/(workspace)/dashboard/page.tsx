"use client";

import { useQuery } from "@tanstack/react-query";
import { Compass, Database, Play, Plus } from "lucide-react";
import { getDashboardStats } from "@/lib/api-client";
import { ButtonLink } from "@/components/ui/Button";
import { Card, CardHeader, MetricCard, Spinner } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/Badge";
import { EmptyState } from "@/components/ui/States";
import { formatNumber } from "@/lib/utils";

export default function DashboardPage() {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["dashboard", "stats"],
    queryFn: getDashboardStats,
    refetchInterval: 10_000,
  });

  return (
    <div className="mx-auto max-w-6xl space-y-5">
      <div>
        <h1 className="text-lg font-semibold tracking-tight">Workspace</h1>
        <p className="mt-0.5 text-sm text-muted">
          Discover data for your ML problem, then run the automated pipeline — all in one place.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        <ButtonLink href="/discovery">
          <Compass className="size-4" /> Discover dataset
        </ButtonLink>
        <ButtonLink href="/datasets" variant="secondary">
          <Database className="size-4" /> Datasets
        </ButtonLink>
        <ButtonLink href="/automl-runs" variant="secondary">
          <Play className="size-4" /> New AutoML run
        </ButtonLink>
      </div>

      {isLoading ? (
        <div className="flex items-center gap-2 py-16 text-sm text-muted">
          <Spinner /> Loading workspace…
        </div>
      ) : isError || !data ? (
        <EmptyState
          title="Backend unreachable"
          description={(error as Error)?.message ?? "Start the FastAPI backend to populate the dashboard."}
          action={{ href: "/discovery", label: "Dataset Discovery" }}
        />
      ) : (
        <>
          <div className="grid gap-3 md:grid-cols-4">
            <MetricCard label="Datasets" value={<span className="num">{formatNumber(data.total_datasets)}</span>} />
            <MetricCard
              label="Discovery requests"
              value={<span className="num">{formatNumber(data.total_discovery_requests)}</span>}
            />
            <MetricCard label="Registered sources" value={<span className="num">{formatNumber(data.total_sources)}</span>} />
            <MetricCard label="Active jobs" value={<span className="num">{formatNumber(data.active_jobs)}</span>} />
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader
                title="Active jobs"
                subtitle="Background work in progress"
                actions={
                  data.active_job_list.length === 0 ? (
                    <span className="text-xs text-faint">none</span>
                  ) : undefined
                }
              />
              {data.active_job_list.length === 0 ? (
                <p className="px-4 py-6 text-center text-xs text-faint">No jobs are running right now.</p>
              ) : (
                <ul className="divide-y divide-border text-sm">
                  {data.active_job_list.map((job) => (
                    <li key={job.id} className="flex items-center justify-between px-4 py-2.5">
                      <span className="text-xs text-muted">{job.kind.replaceAll("_", " ").toLowerCase()}</span>
                      <span className="flex items-center gap-2">
                        <span className="text-xs text-faint">{job.stage?.toLowerCase()}</span>
                        <StatusBadge status={job.status} />
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </Card>

            <Card>
              <CardHeader title="Recent discovery requests" />
              {data.recent_requests.length === 0 ? (
                <div className="px-4 py-6 text-center">
                  <p className="text-xs text-faint">No discovery requests yet.</p>
                  <ButtonLink href="/discovery" size="sm" variant="secondary" className="mt-3">
                    <Plus className="size-3.5" /> Describe an ML problem
                  </ButtonLink>
                </div>
              ) : (
                <ul className="divide-y divide-border text-sm">
                  {data.recent_requests.map((req) => (
                    <li key={req.id} className="flex items-center justify-between gap-4 px-4 py-2.5">
                      <a href={`/discovery/${req.id}`} className="min-w-0 flex-1 truncate text-xs text-text hover:text-accent">
                        &ldquo;{req.query}&rdquo;
                      </a>
                      <StatusBadge status={req.status} />
                    </li>
                  ))}
                </ul>
              )}
            </Card>

            <Card className="lg:col-span-2">
              <CardHeader title="Recent datasets" subtitle="Selected and downloaded datasets" />
              {data.recent_datasets.length === 0 ? (
                <div className="px-4 py-6 text-center">
                  <p className="text-xs text-faint">No datasets selected yet.</p>
                  <ButtonLink href="/discovery" size="sm" variant="secondary" className="mt-3">
                    <Compass className="size-3.5" /> Start with dataset discovery
                  </ButtonLink>
                </div>
              ) : (
                <ul className="divide-y divide-border text-sm">
                  {data.recent_datasets.map((ds) => (
                    <li key={ds.id} className="flex items-center justify-between gap-4 px-4 py-2.5">
                      <a href={`/datasets/${ds.id}`} className="min-w-0 flex-1 truncate text-xs text-text hover:text-accent">
                        {ds.name}
                      </a>
                      <span className="num shrink-0 text-xs text-faint">
                        {formatNumber(ds.row_count)} rows · {ds.source ?? "—"}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
