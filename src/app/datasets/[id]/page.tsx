"use client";

import { use, useState } from "react";
import { useRouter } from "next/navigation";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getDataset, getDatasetFiles, getDatasetJobs, getEda, confirmTarget, deleteDataset, getPreprocessReport, startPreprocessing } from "@/lib/api-client";
import type { ColumnStats, PreprocessReportOut } from "@/lib/api-types";
import { Badge, StatusBadge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardHeader, MetricCard, ProgressBar, Spinner } from "@/components/ui/Card";
import { ErrorState, NotAvailableYet } from "@/components/ui/States";
import { ConfirmDeleteButton } from "@/components/ui/ConfirmDeleteButton";
import { formatBytes, formatNumber, formatPct } from "@/lib/utils";

const CHART_COLORS = ["var(--accent)", "var(--success)", "var(--warning)", "var(--danger)", "#9d7bd8", "#5bc0d4"];

function DownloadProgress({ datasetId }: { datasetId: string }) {
  const { data: jobs } = useQuery({
    queryKey: ["dataset", datasetId, "jobs"],
    queryFn: () => getDatasetJobs(datasetId),
    refetchInterval: (q) => {
      const running = q.state.data?.some((j) => j.status === "RUNNING" || j.status === "QUEUED");
      return running ? 1500 : false;
    },
  });
  const jobList = jobs ?? [];
  const download = jobList.find((j) => j.kind === "DATASET_DOWNLOAD");
  const eda = jobList.find((j) => j.kind === "EDA");
  if (!download) return null;

  const overall =
    download.status === "COMPLETED"
      ? eda?.status === "COMPLETED"
        ? 1
        : 0.85
      : 0.1 + download.progress * 0.7;

  return (
    <Card className="p-4">
      <div className="mb-2 flex items-center justify-between text-sm">
        <span className="font-medium">Pipeline progress</span>
        <div className="flex gap-2">
          <StatusBadge status={download.status} />
          {eda ? <StatusBadge status={eda.status} /> : null}
        </div>
      </div>
      <ProgressBar value={overall} />
      <div className="mt-2 flex items-center justify-between text-xs text-muted">
        <span>
          {download.status === "FAILED"
            ? ""
            : download.stage === "DOWNLOADING"
              ? "Downloading from the approved source…"
              : download.stage === "VALIDATING"
                ? "Validating the downloaded file…"
                : download.stage === "STORING"
                  ? "Storing artifact…"
                  : download.status === "COMPLETED" && eda?.status === "RUNNING"
                    ? "Running exploratory data analysis…"
                    : download.status === "COMPLETED"
                      ? "Done."
                      : "Waiting…"}
        </span>
        <span className="num">{Math.round(overall * 100)}%</span>
      </div>
      {download.status === "FAILED" ? (
        <ErrorState
          className="mt-3"
          title="Download failed"
          what={download.error_message ?? "The dataset could not be retrieved."}
          why={download.error_code ?? undefined}
          whatToDo="Try selecting the dataset again, or pick a different dataset."
        />
      ) : null}
      {eda?.status === "FAILED" ? (
        <ErrorState
          className="mt-3"
          title="EDA failed"
          what={eda.error_message ?? "Exploratory analysis failed."}
          why={eda.error_code ?? undefined}
          whatToDo="Check the dataset file validity or re-select the dataset."
        />
      ) : null}
    </Card>
  );
}

function Histogram({ stats }: { stats: ColumnStats }) {
  if (!stats.histogram?.length) return null;
  const data = stats.histogram.map(([bin, count]) => ({
    bin: Number.isInteger(bin) ? String(bin) : bin.toPrecision(3),
    count,
  }));
  return (
    <div className="h-28">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 4 }}>
          <XAxis dataKey="bin" tick={{ fontSize: 9, fill: "var(--muted)" }} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 9, fill: "var(--muted)" }} width={30} />
          <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--text)", fontSize: 11 }} />
          <Bar dataKey="count" fill="var(--accent)" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function TopValues({ stats }: { stats: ColumnStats }) {
  if (!stats.top_values?.length) return null;
  const data = stats.top_values.map(([value, count]) => ({
    value: value.length > 12 ? `${value.slice(0, 11)}…` : value,
    count,
  }));
  return (
    <div className="h-28">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
          <XAxis type="number" tick={{ fontSize: 9, fill: "var(--muted)" }} />
          <YAxis type="category" dataKey="value" tick={{ fontSize: 9, fill: "var(--muted)" }} width={70} />
          <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--text)", fontSize: 11 }} />
          <Bar dataKey="count" fill="var(--success)" radius={[0, 3, 3, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function TargetConfirmation({ datasetId }: { datasetId: string }) {
  const [selected, setSelected] = useState<string>("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [done, setDone] = useState(false);

  const { data: eda } = useQuery({ queryKey: ["dataset", datasetId, "eda"], queryFn: () => getEda(datasetId) });
  if (!eda) return null;
  if (eda.target_candidates.length === 0) {
    return (
      <NotAvailableYet
        feature="Target suggestion"
        description="No confident target candidates were found in this dataset. AutoML target configuration arrives in the next phase."
      />
    );
  }

  const candidates = eda.target_candidates;
  const effectiveTarget = selected || eda.suggested_target || candidates[0].column;
  const effectiveTask = candidates.find((c) => c.column === effectiveTarget)?.task ?? eda.suggested_task ?? "classification";

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      await confirmTarget(
        datasetId,
        effectiveTarget,
        effectiveTask === "regression" ? "regression" : "classification"
      );
      setDone(true);
    } catch (err) {
      setError(err as Error);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card>
      <CardHeader
        title="Confirm the prediction target"
        subtitle="Detected from column types and distributions. AutoML will use this in the next phase."
      />
      <div className="p-4">
        <div className="space-y-2">
          {candidates.map((c) => (
            <label
              key={c.column}
              className={[
                "flex cursor-pointer items-center justify-between rounded-md border px-3 py-2 text-sm",
                effectiveTarget === c.column ? "border-accent/60 bg-accent/5" : "border-border hover:border-border-strong",
              ].join(" ")}
            >
              <span className="flex items-center gap-3">
                <input
                  type="radio"
                  name="target"
                  checked={effectiveTarget === c.column}
                  onChange={() => {
                    setSelected(c.column);
                    setDone(false);
                  }}
                  className="size-3.5 accent-[var(--color-accent)]"
                />
                <span className="font-mono text-xs text-text">{c.column}</span>
                <Badge tone="neutral">{c.task}</Badge>
              </span>
              <span className="text-xs text-faint">
                {Math.round(c.confidence * 100)}% · {c.rationale}
              </span>
            </label>
          ))}
        </div>
        <div className="mt-3 flex items-center justify-between">
          <span className="text-xs text-muted">
            Task: <span className="text-text">{effectiveTask}</span>
          </span>
          <Button size="sm" onClick={() => void save()} disabled={saving || done}>
            {saving ? <Spinner className="border-t-text" /> : null}
            {done ? "Target confirmed" : "Confirm target"}
          </Button>
        </div>
        {done ? (
          <p className="mt-2 text-xs text-success">
            Target saved. The AutoML training pipeline (Optuna tuning, leaderboard, SHAP) arrives in the next phase.
          </p>
        ) : null}
        {error ? (
          <ErrorState className="mt-2" what={error.message} whatToDo="Pick a column that exists in the dataset and retry." />
        ) : null}
      </div>
    </Card>
  );
}

function PreprocessSection({ datasetId }: { datasetId: string }) {
  const queryClient = useQueryClient();
  const [starting, setStarting] = useState(false);

  const { data: report, error, isLoading, refetch } = useQuery({
    queryKey: ["dataset", datasetId, "preprocess"],
    queryFn: () => getPreprocessReport(datasetId),
    retry: false,
  });

  const start = async () => {
    setStarting(true);
    try {
      await startPreprocessing(datasetId);
      await refetch();
      await queryClient.invalidateQueries({ queryKey: ["dataset", datasetId, "jobs"] });
    } finally {
      setStarting(false);
    }
  };

  if (isLoading) {
    return (
      <Card className="p-4 text-sm text-muted">
        <div className="flex items-center gap-2"><Spinner /> Checking preprocessing…</div>
      </Card>
    );
  }
  if (error || !report) {
    return (
      <Card className="p-4">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h3 className="text-sm font-medium text-text">Preprocessing</h3>
            <p className="mt-0.5 text-xs text-muted">
              Build an ML-ready copy: dedupe, impute missing values, encode categoricals, exclude IDs.
            </p>
          </div>
          <Button size="sm" onClick={() => void start()} disabled={starting}>
            {starting ? <Spinner className="border-t-text" /> : null}
            Run preprocessing
          </Button>
        </div>
      </Card>
    );
  }

  const missingBefore = report.missing_cells_before;
  const missingAfter = report.missing_cells_after;

  return (
    <Card>
      <CardHeader title="Preprocessing" subtitle="What was applied to produce the ML-ready copy" />
      <div className="space-y-4 p-4">
        <div className="grid gap-3 sm:grid-cols-4">
          <MetricCard
            label="Shape"
            value={
              <span className="num">
                {report.input_shape[0].toLocaleString()}×{report.input_shape[1]} → {report.output_shape[0].toLocaleString()}×
                {report.output_shape[1]}
              </span>
            }
            hint="before → after"
          />
          <MetricCard
            label="Missing cells"
            value={<span className="num">{formatNumber(missingBefore)} → {formatNumber(missingAfter)}</span>}
          />
          <MetricCard label="Duplicates removed" value={<span className="num">{formatNumber(report.rows_dropped_duplicates)}</span>} />
          <MetricCard
            label="Columns removed"
            value={<span className="num">{report.constant_columns_removed.length + report.identifier_columns_excluded.length}</span>}
            hint="constant + identifier-like"
          />
        </div>

        <div className="grid gap-3 md:grid-cols-3 text-xs">
          {Object.keys(report.imputed).length > 0 ? (
            <div className="rounded-md border border-border p-3">
              <h4 className="font-medium text-text">Imputed (median / mode)</h4>
              <ul className="mt-1 space-y-0.5 text-muted">
                {Object.entries(report.imputed).slice(0, 8).map(([col, strategy]) => (
                  <li key={col} className="truncate font-mono">{col} → {strategy}</li>
                ))}
              </ul>
            </div>
          ) : null}
          {Object.keys(report.one_hot_encoded).length + Object.keys(report.ordinal_encoded).length > 0 ? (
            <div className="rounded-md border border-border p-3">
              <h4 className="font-medium text-text">Encoded</h4>
              <ul className="mt-1 space-y-0.5 text-muted">
                {Object.entries(report.one_hot_encoded).map(([col, n]) => (
                  <li key={col} className="truncate font-mono">{col} → one-hot ({n})</li>
                ))}
                {Object.entries(report.ordinal_encoded).map(([col, n]) => (
                  <li key={col} className="truncate font-mono">{col} → ordinal ({n})</li>
                ))}
              </ul>
            </div>
          ) : null}
          {Object.keys(report.outliers).length > 0 ? (
            <div className="rounded-md border border-border p-3">
              <h4 className="font-medium text-text">Outliers flagged (IQR)</h4>
              <ul className="mt-1 space-y-0.5 text-muted">
                {Object.entries(report.outliers).slice(0, 8).map(([col, n]) => (
                  <li key={col} className="truncate font-mono">{col}: {formatNumber(n)}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>

        <div>
          <h4 className="text-xs font-medium uppercase tracking-wider text-faint">Applied steps</h4>
          <ol className="mt-1 list-inside list-decimal space-y-0.5 text-xs text-muted">
            {report.steps.map((step, i) => (
              <li key={i}>{step}</li>
            ))}
          </ol>
        </div>
      </div>
    </Card>
  );
}

export default function DatasetWorkspacePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<Error | null>(null);

  const { data: dataset, isLoading, isError, error } = useQuery({
    queryKey: ["dataset", id],
    queryFn: () => getDataset(id),
    refetchInterval: 5000,
  });
  const { data: files } = useQuery({
    queryKey: ["dataset", id, "files"],
    queryFn: () => getDatasetFiles(id),
    enabled: !!dataset,
  });
  const { data: eda, error: edaError } = useQuery({
    queryKey: ["dataset", id, "eda"],
    queryFn: () => getEda(id),
    enabled: !!dataset,
    retry: false,
    refetchInterval: (q) => (q.state.error ? false : 10_000),
  });

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 pt-20 text-sm text-muted">
        <Spinner /> Loading dataset…
      </div>
    );
  }
  if (isError || !dataset) {
    return <ErrorState what={(error as Error)?.message ?? "Dataset not found."} whatToDo="Open it from the Datasets page." />;
  }

  const doDelete = async () => {
    setDeleting(true);
    setDeleteError(null);
    try {
      await deleteDataset(id);
      router.push("/datasets");
    } catch (err) {
      setDeleteError(err as Error);
      setDeleting(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl space-y-4">
      <div className="rounded-lg border border-border bg-surface p-5">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-lg font-semibold">{dataset.name}</h1>
            <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs text-muted">
              {dataset.source ? <Badge tone="accent">{dataset.source}</Badge> : null}
              {dataset.file_format ? <Badge tone="neutral">{dataset.file_format}</Badge> : null}
              {dataset.license ? <Badge tone="neutral">license: {dataset.license}</Badge> : null}
              <span className="num">
                {formatNumber(dataset.row_count)} rows · {dataset.column_count ?? "?"} columns
              </span>
            </div>
          </div>
          <div className="flex flex-col items-end gap-2">
            <div className="flex gap-2">
              <a
                href={`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/datasets/${id}/export?variant=raw`}
                className="inline-flex items-center gap-1.5 rounded-md border border-border bg-elevated px-2.5 py-1.5 text-xs font-medium text-text hover:border-accent/40"
                title="Download the raw data as CSV"
              >
                Download CSV
              </a>
              {files?.some((f) => f.file_name === "processed.parquet") ? (
                <a
                  href={`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/datasets/${id}/export?variant=processed`}
                  className="inline-flex items-center gap-1.5 rounded-md border border-accent/60 bg-accent/15 px-2.5 py-1.5 text-xs font-medium text-accent hover:bg-accent/25"
                  title="Download the preprocessed data as CSV"
                >
                  Download preprocessed CSV
                </a>
              ) : null}
            </div>
            {dataset.selected_target ? (
              <Badge tone="success">
                target: {dataset.selected_target} ({dataset.selected_task})
              </Badge>
            ) : null}
            <ConfirmDeleteButton
              size="md"
              disabled={deleting}
              onConfirm={() => doDelete()}
            />
            {deleteError ? <span className="max-w-52 text-right text-xs text-danger">{deleteError.message}</span> : null}
          </div>
        </div>
      </div>

      <DownloadProgress datasetId={id} />

      {eda ? (
        <>
          <div className="grid gap-3 md:grid-cols-4">
            <MetricCard
              label="Shape"
              value={
                <span className="num">
                  {eda.shape[0].toLocaleString()} × {eda.shape[1]}
                </span>
              }
              hint="rows × columns"
            />
            <MetricCard label="Duplicate rows" value={<span className="num">{formatNumber(eda.duplicate_rows)}</span>} />
            <MetricCard
              label="Missing overall"
              value={formatPct(
                eda.column_stats.reduce((acc, c) => acc + c.missing_count, 0) /
                  Math.max(1, eda.shape[0] * eda.shape[1]),
                1
              )}
            />
            <MetricCard
              label="Suggested target"
              value={<span className="font-mono text-sm">{eda.suggested_target ?? "—"}</span>}
              hint={eda.suggested_task ? `task: ${eda.suggested_task}` : undefined}
            />
          </div>

          {eda.quality_warnings.length > 0 ? (
            <Card className="border-warning/30 bg-warning/5 p-4">
              <h3 className="text-sm font-semibold text-warning">Quality warnings</h3>
              <ul className="mt-1.5 list-inside list-disc space-y-0.5 text-xs text-muted">
                {eda.quality_warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </Card>
          ) : null}

          {eda.class_balance ? (
            <Card>
              <CardHeader
                title="Class balance"
                subtitle={`${eda.class_balance.n_classes} classes${eda.class_balance.imbalanced ? " · imbalanced" : ""}`}
              />
              <div className="h-44 p-4">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={Object.entries(eda.class_balance.counts).map(([k, v]) => ({ name: k, count: v }))}
                    margin={{ top: 4, right: 8, bottom: 0, left: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="name" tick={{ fontSize: 10, fill: "var(--muted)" }} />
                    <YAxis tick={{ fontSize: 10, fill: "var(--muted)" }} width={40} />
                    <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--text)", fontSize: 11 }} />
                    <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                      {Object.keys(eda.class_balance.counts).map((_, i) => (
                        <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          ) : null}

          <Card>
            <CardHeader title="Columns" subtitle="Per-column statistics from the validated file" />
            <div className="divide-y divide-border">
              {eda.column_stats.map((c) => (
                <div key={c.name} className="grid grid-cols-[minmax(140px,1fr)_2fr] items-center gap-4 px-4 py-3">
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="truncate font-mono text-xs text-text">{c.name}</span>
                      {c.is_potential_id ? <Badge tone="warning">id?</Badge> : null}
                      {c.is_constant ? <Badge tone="danger">constant</Badge> : null}
                    </div>
                    <div className="mt-0.5 flex gap-2 text-[10px] text-faint">
                      <span>{c.dtype}</span>
                      <span>· {formatPct(c.missing_ratio, 1)} missing</span>
                      <span>· {formatNumber(c.unique_count)} unique</span>
                      {c.outlier_count ? <span>· {formatNumber(c.outlier_count)} outliers</span> : null}
                    </div>
                  </div>
                  <div>{c.histogram ? <Histogram stats={c} /> : c.top_values ? <TopValues stats={c} /> : null}</div>
                </div>
              ))}
            </div>
          </Card>

          <PreprocessSection datasetId={id} />

          <TargetConfirmation datasetId={id} />
        </>
      ) : edaError ? (
        <NotAvailableYet
          feature="EDA report"
          description="The exploratory data analysis report appears here once the dataset download and validation complete."
        />
      ) : (
        <div className="flex items-center gap-2 text-sm text-muted">
          <Spinner /> Waiting for EDA…
        </div>
      )}

      {files && files.length > 0 ? (
        <Card>
          <CardHeader title="Files" subtitle="Validated artifacts in storage" />
          <div className="divide-y divide-border text-sm">
            {files.map((f) => (
              <div key={f.id} className="flex items-center justify-between px-4 py-2.5">
                <span className="font-mono text-xs">{f.file_name}</span>
                <span className="flex items-center gap-3 text-xs text-muted">
                  {f.file_format}
                  <span className="num">{formatBytes(f.file_size_bytes)}</span>
                  {f.validated ? <Badge tone="success">validated</Badge> : <Badge tone="warning">unvalidated</Badge>}
                </span>
              </div>
            ))}
          </div>
        </Card>
      ) : null}
    </div>
  );
}
