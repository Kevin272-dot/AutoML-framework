"use client";

import { use, useState } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { ExternalLink, Info } from "lucide-react";
import { getCandidate, getCandidatePreview, selectDataset } from "@/lib/api-client";
import type { ScoreComponents } from "@/lib/api-types";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardHeader, Spinner } from "@/components/ui/Card";
import { ErrorState, NotAvailableYet } from "@/components/ui/States";
import { ScoreBreakdown } from "@/components/discovery/ResultsTable";
import { formatBytes, formatNumber, formatPct } from "@/lib/utils";

const TABS = ["Overview", "Preview", "Schema", "Files", "Provenance"] as const;

export default function CandidateDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const [tab, setTab] = useState<(typeof TABS)[number]>("Overview");
  const [showScore, setShowScore] = useState(false);
  const [selecting, setSelecting] = useState(false);
  const [selectError, setSelectError] = useState<Error | null>(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);

  const { data: candidate, isLoading, isError, error } = useQuery({
    queryKey: ["candidate", id],
    queryFn: () => getCandidate(id),
  });

  const { data: preview, isLoading: previewLoading, error: previewError, refetch: refetchPreview } = useQuery({
    queryKey: ["candidate", id, "preview"],
    queryFn: () => getCandidatePreview(id),
    enabled: tab === "Preview" && !!candidate?.preview_available,
    retry: 1,
  });

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 pt-20 text-sm text-muted">
        <Spinner /> Loading dataset…
      </div>
    );
  }
  if (isError || !candidate) {
    return <ErrorState what={(error as Error)?.message ?? "Dataset candidate not found."} whatToDo="Return to the discovery results." />;
  }

  const doSelect = async () => {
    setSelecting(true);
    setSelectError(null);
    try {
      const res = await selectDataset(candidate.id);
      setSelectedDatasetId(res.dataset_id);
      router.push(`/datasets/${res.dataset_id}`);
    } catch (err) {
      setSelectError(err as Error);
    } finally {
      setSelecting(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl space-y-4">
      {/* Header */}
      <div className="rounded-lg border border-border bg-surface p-5">
        <div className="flex items-start justify-between gap-6">
          <div className="min-w-0">
            <h1 className="text-lg font-semibold">{candidate.name}</h1>
            <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs text-muted">
              <Badge tone="accent">{candidate.source}</Badge>
              {candidate.license ? <Badge tone="neutral">license: {candidate.license}</Badge> : null}
              {candidate.file_format ? <Badge tone="neutral">{candidate.file_format}</Badge> : null}
              <span className="num">
                {formatNumber(candidate.row_count)} rows · {candidate.column_count ?? "?"} columns
                {candidate.file_size_bytes ? ` · ${formatBytes(candidate.file_size_bytes)}` : ""}
              </span>
            </div>
            {candidate.description ? (
              <p className="mt-2.5 max-w-3xl text-sm leading-relaxed text-muted">{candidate.description}</p>
            ) : null}
          </div>

          <div className="flex shrink-0 flex-col items-end gap-2">
            <button
              className="flex items-center gap-1 text-sm text-muted hover:text-text"
              onClick={() => setShowScore((s) => !s)}
            >
              <span className="num font-semibold text-accent">
                {candidate.overall_score !== null ? formatPct(candidate.overall_score, 1) : "—"}
              </span>
              relevance
              <Info className="size-3.5" />
            </button>
            {showScore && candidate.score_components ? (
              <ScoreBreakdown components={candidate.score_components as ScoreComponents} overall={candidate.overall_score ?? 0} />
            ) : null}
            <Button onClick={() => void doSelect()} disabled={!candidate.download_available || selecting}>
              {selecting ? <Spinner className="border-t-text" /> : null}
              {candidate.download_available ? "Select dataset" : "No downloadable files"}
            </Button>
            {selectError ? <span className="max-w-52 text-right text-xs text-danger">{selectError.message}</span> : null}
            {selectedDatasetId ? <span className="text-xs text-success">Selected — opening dataset…</span> : null}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-border">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={[
              "rounded-t-md border-b-2 px-4 py-2 text-sm transition-colors",
              tab === t ? "border-accent font-medium text-text" : "border-transparent text-muted hover:text-text",
            ].join(" ")}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === "Overview" ? (
        <div className="grid gap-4 md:grid-cols-3">
          <Card className="p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-faint">Coverage</h3>
            <dl className="mt-2 space-y-1.5 text-sm">
              <div className="flex justify-between"><dt className="text-muted">Date start</dt><dd>{candidate.date_start ?? "—"}</dd></div>
              <div className="flex justify-between"><dt className="text-muted">Date end</dt><dd>{candidate.date_end ?? "—"}</dd></div>
              <div className="flex justify-between"><dt className="text-muted">Rows</dt><dd className="num">{formatNumber(candidate.row_count)}</dd></div>
              <div className="flex justify-between"><dt className="text-muted">Columns</dt><dd className="num">{candidate.column_count ?? "—"}</dd></div>
              <div className="flex justify-between"><dt className="text-muted">Format</dt><dd>{candidate.file_format ?? "—"}</dd></div>
              <div className="flex justify-between"><dt className="text-muted">Size</dt><dd className="num">{formatBytes(candidate.file_size_bytes)}</dd></div>
            </dl>
          </Card>
          <Card className="p-4 md:col-span-2">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-faint">Tags</h3>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {candidate.tags.length ? (
                candidate.tags.slice(0, 24).map((t) => <Badge key={t} tone="neutral">{t}</Badge>)
              ) : (
                <span className="text-sm text-faint">No tags published by the source.</span>
              )}
            </div>
          </Card>
        </div>
      ) : null}

      {tab === "Preview" ? (
        previewLoading ? (
          <div className="flex items-center gap-2 py-10 text-sm text-muted"><Spinner /> Loading real sample rows from the source…</div>
        ) : previewError ? (
          <ErrorState
            what={(previewError as Error).message}
            why="The source's preview service may be temporarily unavailable."
            whatToDo="Retry, or use the dataset's source page. Selecting the dataset downloads real files regardless of preview."
            onRetry={() => void refetchPreview()}
          />
        ) : preview && preview.rows.length > 0 ? (
          <div className="overflow-x-auto rounded-lg border border-border bg-surface">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border bg-elevated/60">
                  {preview.columns.map((c) => (
                    <th key={c} className="whitespace-nowrap px-3 py-2 text-left font-medium text-muted">{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.rows.slice(0, 50).map((row, i) => (
                  <tr key={i} className="border-b border-border/50 last:border-0">
                    {preview.columns.map((c) => (
                      <td key={c} className="max-w-[220px] truncate px-3 py-1.5 font-mono text-[11px] text-text/90">
                        {row[c] === null || row[c] === undefined ? <span className="text-faint">null</span> : String(row[c])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="border-t border-border px-3 py-2 text-xs text-faint">
              First {Math.min(50, preview.rows.length)} real rows served by the source API
              {preview.row_count_total ? ` · ${formatNumber(preview.row_count_total)} total rows` : ""}.
            </div>
          </div>
        ) : (
          <NotAvailableYet feature="Preview" description="The source does not expose a preview for this dataset configuration." />
        )
      ) : null}

      {tab === "Schema" ? (
        candidate.columns.length > 0 ? (
          <div className="overflow-x-auto rounded-lg border border-border bg-surface">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-elevated/60 text-[11px] uppercase tracking-wider text-muted">
                  <th className="px-3 py-2 text-left">Column</th>
                  <th className="px-3 py-2 text-left">Type</th>
                  <th className="px-3 py-2 text-left">Semantic type</th>
                  <th className="px-3 py-2 text-left">Examples</th>
                </tr>
              </thead>
              <tbody>
                {candidate.columns.map((c) => (
                  <tr key={c.name} className="border-b border-border/50 last:border-0">
                    <td className="px-3 py-2 font-mono text-xs">{c.name}</td>
                    <td className="px-3 py-2 text-xs text-muted">{c.dtype}</td>
                    <td className="px-3 py-2"><Badge tone="neutral">{c.semantic_type}</Badge></td>
                    <td className="px-3 py-2 text-xs text-faint">
                      {c.sample_values?.slice(0, 3).map((v) => String(v)).join(", ") || "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <NotAvailableYet
            feature="Schema"
            description="Column-level schema becomes available after the dataset is selected and downloaded, when the real file is validated."
          />
        )
      ) : null}

      {tab === "Files" ? (
        <NotAvailableYet
          feature="File listing"
          description="Data files are retrieved directly from the source's official download endpoint when you select this dataset (documented machine-readable files only)."
        />
      ) : null}

      {tab === "Provenance" ? (
        <Card>
          <CardHeader title="Provenance" subtitle="Where this dataset came from" />
          <dl className="divide-y divide-border text-sm">
            {[
              ["Source", candidate.source],
              ["Source dataset ID", candidate.source_dataset_id],
              ["URL", candidate.url],
              ["Retrieved at", new Date(candidate.retrieved_at).toLocaleString()],
              ["License", candidate.license ?? "unknown — check the source page"],
              ["Access method", "Official public API (no scraping)"],
            ].map(([k, v]) => (
              <div key={k as string} className="flex items-center justify-between gap-4 px-4 py-2.5">
                <dt className="text-muted">{k}</dt>
                <dd className="max-w-[60%] truncate text-right">
                  {String(v).startsWith("http") ? (
                    <a href={String(v)} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-accent hover:underline">
                      {String(v)} <ExternalLink className="size-3" />
                    </a>
                  ) : (
                    String(v)
                  )}
                </dd>
              </div>
            ))}
          </dl>
        </Card>
      ) : null}
    </div>
  );
}
