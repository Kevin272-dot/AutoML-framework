"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import { ArrowUpDown, Eye, Info } from "lucide-react";
import type { RankedCandidateOut, ScoreComponents } from "@/lib/api-types";
import { formatNumber, formatPct } from "@/lib/utils";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";

const COMPONENT_LABELS: Record<keyof ScoreComponents, { label: string; weight: number }> = {
  keyword_match: { label: "Keyword match", weight: 0.3 },
  date_match: { label: "Date match", weight: 0.2 },
  domain_match: { label: "Domain match", weight: 0.2 },
  completeness: { label: "Completeness", weight: 0.1 },
  size_score: { label: "Size", weight: 0.1 },
  feature_quality: { label: "Feature quality", weight: 0.1 },
};

export function ScoreBreakdown({ components, overall }: { components: ScoreComponents; overall: number }) {
  return (
    <div className="w-64 rounded-lg border border-border bg-elevated p-3 text-xs shadow-lg">
      <div className="mb-2 flex items-baseline justify-between border-b border-border pb-2">
        <span className="font-medium text-text">Overall relevance</span>
        <span className="num text-sm font-semibold text-accent">{formatPct(overall, 1)}</span>
      </div>
      {Object.entries(COMPONENT_LABELS).map(([key, meta]) => {
        const value = components[key as keyof ScoreComponents] ?? 0;
        return (
          <div key={key} className="mb-1.5">
            <div className="flex justify-between text-muted">
              <span>
                {meta.label} <span className="text-faint">({Math.round(meta.weight * 100)}%)</span>
              </span>
              <span className="num">{formatPct(value)}</span>
            </div>
            <div className="mt-0.5 h-1 overflow-hidden rounded bg-bg">
              <div className="h-full rounded bg-accent/70" style={{ width: `${value * 100}%` }} />
            </div>
          </div>
        );
      })}
      <p className="mt-2 border-t border-border pt-2 text-[10px] leading-relaxed text-faint">
        Score components are normalized 0–1 and combined with fixed weights. Higher keyword/domain/date
        overlap and richer, better-sized data rank first.
      </p>
    </div>
  );
}

const columnHelper = createColumnHelper<RankedCandidateOut>();

export function ResultsTable({ results }: { results: RankedCandidateOut[] }) {
  const router = useRouter();
  const [sorting, setSorting] = useState<SortingState>([{ id: "overall_score", desc: true }]);
  const [hovered, setHovered] = useState<string | null>(null);

  const columns = useMemo(
    () => [
      columnHelper.accessor("name", {
        header: "Dataset",
        cell: (info) => (
          <div className="max-w-[280px]">
            <div className="truncate font-medium text-text">{info.getValue()}</div>
            <div className="truncate text-xs text-faint">{info.row.original.source_dataset_id}</div>
          </div>
        ),
      }),
      columnHelper.accessor("source", {
        header: "Source",
        cell: (info) => <Badge tone="neutral">{info.getValue()}</Badge>,
      }),
      columnHelper.accessor("overall_score", {
        header: "Relevance",
        cell: (info) => (
          <div className="relative flex items-center gap-1.5">
            <span className="num font-medium text-accent">{formatPct(info.getValue(), 1)}</span>
            <button
              className="text-faint hover:text-text"
              aria-label="Explain score"
              onClick={() => setHovered(hovered === info.row.original.id ? null : info.row.original.id)}
            >
              <Info className="size-3.5" />
            </button>
            {hovered === info.row.original.id ? (
              <div className="absolute left-6 top-5 z-20">
                <ScoreBreakdown components={info.row.original.score_components} overall={info.row.original.overall_score} />
              </div>
            ) : null}
          </div>
        ),
      }),
      columnHelper.accessor("row_count", {
        header: "Rows",
        cell: (info) => <span className="num">{formatNumber(info.getValue())}</span>,
      }),
      columnHelper.accessor("column_count", {
        header: "Columns",
        cell: (info) => <span className="num">{info.getValue() ?? "—"}</span>,
      }),
      columnHelper.accessor((row) => (row.date_start ? `${row.date_start ?? "—"} – ${row.date_end ?? "—"}` : "—"), {
        id: "date",
        header: "Date",
        cell: (info) => <span className="text-xs text-muted">{info.getValue()}</span>,
      }),
      columnHelper.accessor("file_format", {
        header: "Format",
        cell: (info) => (info.getValue() ? <Badge tone="neutral">{info.getValue()!}</Badge> : <span className="text-faint">—</span>),
      }),
      columnHelper.accessor("license", {
        header: "License",
        cell: (info) => <span className="text-xs text-muted">{info.getValue() ?? "unknown"}</span>,
      }),
      columnHelper.display({
        id: "actions",
        header: "Actions",
        cell: (info) => (
          <div className="flex gap-1.5">
            <Link
              href={`/datasets/candidates/${info.row.original.id}`}
              className="inline-flex items-center gap-1 rounded-md border border-border bg-elevated px-2 py-1 text-xs text-text hover:border-accent/40"
            >
              <Eye className="size-3" /> Details
            </Link>
            <Button
              size="sm"
              disabled={!info.row.original.download_available}
              onClick={() => router.push(`/datasets/candidates/${info.row.original.id}`)}
              title={info.row.original.download_available ? "Select this dataset" : "No downloadable files"}
            >
              Select
            </Button>
          </div>
        ),
      }),
    ],
    [hovered, router]
  );

  const table = useReactTable({
    data: results,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: { pagination: { pageSize: 15 } },
  });

  return (
    <div className="overflow-x-auto rounded-lg border border-border bg-surface">
      <table className="w-full text-sm">
        <thead>
          {table.getHeaderGroups().map((hg) => (
            <tr key={hg.id} className="border-b border-border bg-elevated/60">
              {hg.headers.map((header) => (
                <th key={header.id} className="whitespace-nowrap px-3 py-2.5 text-left text-[11px] font-medium uppercase tracking-wider text-muted">
                  {header.isPlaceholder ? null : (
                    <button
                      className="inline-flex items-center gap-1 hover:text-text"
                      onClick={header.column.getToggleSortingHandler()}
                      disabled={!header.column.getCanSort()}
                    >
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {header.column.getCanSort() ? <ArrowUpDown className="size-3 opacity-50" /> : null}
                    </button>
                  )}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id} className="border-b border-border/60 last:border-0 hover:bg-hover/50">
              {row.getVisibleCells().map((cell) => (
                <td key={cell.id} className="px-3 py-2.5 align-middle">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="flex items-center justify-between border-t border-border px-3 py-2 text-xs text-muted">
        <span className="num">
          {table.getState().pagination.pageIndex * table.getState().pagination.pageSize + 1}–
          {Math.min((table.getState().pagination.pageIndex + 1) * table.getState().pagination.pageSize, results.length)}{" "}
          of {results.length}
        </span>
        <div className="flex gap-2">
          <Button variant="secondary" size="sm" onClick={() => table.previousPage()} disabled={!table.getCanPreviousPage()}>
            Previous
          </Button>
          <Button variant="secondary" size="sm" onClick={() => table.nextPage()} disabled={!table.getCanNextPage()}>
            Next
          </Button>
        </div>
      </div>
    </div>
  );
}
