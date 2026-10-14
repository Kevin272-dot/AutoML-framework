"use client";

import { use, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getAutoMLRun, getRunFeatures, predictWithModel } from "@/lib/api-client";
import type { PredictOut } from "@/lib/api-types";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardHeader, Spinner } from "@/components/ui/Card";
import { ErrorState, NotAvailableYet } from "@/components/ui/States";
import { formatScore } from "@/lib/utils";

function PredictForm({ runId }: { runId: string }) {
  const { data: features, isLoading, isError, error } = useQuery({
    queryKey: ["model", runId, "features"],
    queryFn: () => getRunFeatures(runId),
  });

  const [values, setValues] = useState<Record<string, string>>({});
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<PredictOut | null>(null);
  const [predictError, setPredictError] = useState<Error | null>(null);

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 py-10 text-sm text-muted">
        <Spinner /> Loading feature schema…
      </div>
    );
  }
  if (isError || !features) {
    return (
      <NotAvailableYet
        feature="Prediction form"
        description={(error as Error)?.message ?? "This model has no packaged feature schema."}
      />
    );
  }

  const submit = async () => {
    setPredicting(true);
    setPredictError(null);
    try {
      const records = [Object.fromEntries(Object.entries(values).map(([k, v]) => [k, v]))];
      const res = await predictWithModel(runId, records);
      setResult(res);
    } catch (err) {
      setPredictError(err as Error);
    } finally {
      setPredicting(false);
    }
  };

  const filled = features.features.every((f) => (values[f.name] ?? "") !== "");

  return (
    <Card>
      <CardHeader
        title="Predict"
        subtitle={`Send one row — the model was trained to predict ${features.target ?? "the target"} (${features.task_type})`}
      />
      <div className="space-y-3 p-4">
        <div className="grid gap-3 sm:grid-cols-2">
          {features.features.map((f) => (
            <div key={f.name}>
              <label className="mb-1 block text-xs font-medium text-muted">
                <span className="font-mono">{f.name}</span>
                <span className="ml-1 text-faint">({f.kind})</span>
              </label>
              <input
                value={values[f.name] ?? ""}
                onChange={(e) => setValues((prev) => ({ ...prev, [f.name]: e.target.value }))}
                inputMode={f.kind === "numeric" ? "decimal" : undefined}
                className="w-full rounded-md border border-border bg-elevated px-3 py-2 text-sm text-text placeholder:text-faint focus:border-accent focus:outline-none"
                placeholder={f.kind === "numeric" ? "e.g. 42" : "value"}
              />
            </div>
          ))}
        </div>

        <Button onClick={() => void submit()} disabled={predicting || !filled}>
          {predicting ? <Spinner className="border-t-text" /> : null}
          Predict
        </Button>

        {predictError ? (
          <ErrorState what={predictError.message} whatToDo="Check that every field matches the training data format." />
        ) : null}

        {result ? (
          <div className="rounded-lg border border-success/40 bg-success/5 p-4">
            <div className="flex items-baseline gap-2">
              <span className="text-xs font-medium uppercase tracking-wider text-muted">Prediction</span>
              {result.classes && result.predictions[0] !== null ? (
                <span className="text-[10px] text-faint">
                  classes: {result.classes.join(", ")}
                </span>
              ) : null}
            </div>
            <p className="mt-1 text-2xl font-semibold text-text">
              {result.predictions[0] === null ? "—" : String(result.predictions[0])}
            </p>
            {result.probabilities && result.classes ? (
              <div className="mt-3 space-y-1.5">
                {result.classes.map((cls, i) => {
                  const p = result.probabilities?.[0]?.[i] ?? 0;
                  return (
                    <div key={cls} className="flex items-center gap-2">
                      <span className="w-24 truncate font-mono text-xs text-muted">{cls}</span>
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-elevated">
                        <div className="h-full rounded-full bg-accent" style={{ width: `${p * 100}%` }} />
                      </div>
                      <span className="num w-12 text-right text-xs text-muted">{(p * 100).toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>
            ) : null}
            {result.count > 1 ? (
              <p className="mt-2 text-xs text-muted">
                +{result.count - 1} more prediction(s) in the response.
              </p>
            ) : null}
          </div>
        ) : null}
      </div>
    </Card>
  );
}

export default function ModelPredictPage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = use(params);

  const { data: run, isLoading, isError, error } = useQuery({
    queryKey: ["automl-run", runId],
    queryFn: () => getAutoMLRun(runId),
  });

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 pt-20 text-sm text-muted">
        <Spinner /> Loading model…
      </div>
    );
  }
  if (isError || !run) {
    return <ErrorState what={(error as Error)?.message ?? "Model not found."} whatToDo="Open it from the Models page." />;
  }

  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <div className="rounded-lg border border-border bg-surface p-5">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-lg font-semibold">{run.best_model}</h1>
            <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs text-muted">
              {run.task ? <Badge tone="accent">{run.task}</Badge> : null}
              <span className="num">test {formatScore(run.best_test_score)}</span>
              <span>· target: {run.target ?? "—"}</span>
            </div>
          </div>
          <a
            href={`/automl-runs/${run.run_id}`}
            className="rounded-md border border-border bg-elevated px-2.5 py-1.5 text-xs text-muted hover:text-text"
          >
            View run
          </a>
        </div>
      </div>

      <PredictForm runId={run.run_id} />

      <Card>
        <CardHeader title="Use this model in your own app" subtitle="Two ways — no platform code required" />
        <div className="space-y-4 p-4 text-xs">
          <div>
            <h4 className="font-medium text-text">1. Call this API directly</h4>
            <pre className="mt-1.5 overflow-x-auto rounded-md bg-elevated p-3 leading-relaxed">
{`curl -X POST ${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/automl/runs/${run.run_id}/predict \\
  -H "Content-Type: application/json" \\
  -d '{"records": [{ ...feature values... }]}'`}
            </pre>
          </div>
          <div>
            <h4 className="font-medium text-text">2. Download model.zip and embed it</h4>
            <pre className="mt-1.5 overflow-x-auto rounded-md bg-elevated p-3 leading-relaxed">
{`from model_package.inference.model_loader import load_model
model = load_model("model_package")
model.predict([{ ...feature values... }])`}
            </pre>
          </div>
        </div>
      </Card>
    </div>
  );
}
