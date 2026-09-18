"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { listDatasets } from "@/lib/api-client";
import { Badge } from "@/components/ui/Badge";
import { ButtonLink } from "@/components/ui/Button";
import { Card, CardHeader, Spinner } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/States";
import { formatNumber } from "@/lib/utils";

export default function DatasetsPage() {
  const { data: datasets, isLoading, isError, error } = useQuery({
    queryKey: ["datasets"],
    queryFn: listDatasets,
  });

  return (
    <div className="mx-auto max-w-5xl space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Datasets</h1>
          <p className="mt-0.5 text-sm text-muted">Datasets you have selected and downloaded.</p>
        </div>
        <ButtonLink href="/discovery" variant="secondary" size="sm">
          Discover more
        </ButtonLink>
      </div>

      {isLoading ? (
        <div className="flex items-center gap-2 py-16 text-sm text-muted">
          <Spinner /> Loading datasets…
        </div>
      ) : isError ? (
        <EmptyState title="Could not load datasets" description={(error as Error)?.message} />
      ) : !datasets || datasets.length === 0 ? (
        <EmptyState
          title="No datasets yet"
          description="Describe your ML problem in Dataset Discovery, approve sources, and select a dataset to get started."
          action={{ href: "/discovery", label: "Start dataset discovery" }}
        />
      ) : (
        <Card>
          <CardHeader title={`${datasets.length} dataset(s)`} />
          <div className="divide-y divide-border">
            {datasets.map((ds) => (
              <Link
                key={ds.id}
                href={`/datasets/${ds.id}`}
                className="flex items-center justify-between gap-4 px-4 py-3 hover:bg-hover/50"
              >
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium text-text">{ds.name}</div>
                  <div className="mt-0.5 flex items-center gap-2 text-xs text-faint">
                    {ds.source ? <Badge tone="neutral">{ds.source}</Badge> : null}
                    {ds.file_format ? <span>{ds.file_format}</span> : null}
                    {ds.selected_target ? (
                      <Badge tone="success">
                        target: {ds.selected_target}
                      </Badge>
                    ) : null}
                  </div>
                </div>
                <span className="num shrink-0 text-xs text-muted">
                  {formatNumber(ds.row_count)} rows · {ds.column_count ?? "?"} cols
                </span>
              </Link>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
