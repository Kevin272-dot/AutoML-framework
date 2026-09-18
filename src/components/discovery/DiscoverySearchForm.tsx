"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { Search } from "lucide-react";
import { createDiscoveryRequest, auditRequestSources, ApiError } from "@/lib/api-client";
import { Button } from "@/components/ui/Button";
import { Spinner } from "@/components/ui/Card";
import { ErrorState } from "@/components/ui/States";

const EXAMPLES = [
  "Find datasets about crop yield in India between 2020 and 2025",
  "Telecom customer churn classification data",
  "Air quality measurements for Europe from 2021 to 2024",
  "Housing prices in California with at least 5000 rows",
];

export function DiscoverySearchForm({ compact = false }: { compact?: boolean }) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const submit = async (q: string) => {
    if (!q.trim() || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const req = await createDiscoveryRequest(q.trim());
      // Kick off source discovery + audit immediately; the wizard page tracks status.
      await auditRequestSources(req.id);
      router.push(`/discovery/${req.id}`);
    } catch (err) {
      setError(err as Error);
      setSubmitting(false);
    }
  };

  return (
    <div className={compact ? "" : "mx-auto max-w-3xl"}>
      {!compact ? (
        <div className="mb-8 text-center">
          <h1 className="text-2xl font-semibold tracking-tight">What data should you use?</h1>
          <p className="mt-2 text-sm text-muted">
            Describe your ML problem in plain language. We discover potential data sources, audit them,
            and <span className="text-text">you approve</span> which ones we search — then rank real
            datasets for your problem.
          </p>
        </div>
      ) : null}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void submit(query);
        }}
        className="flex gap-2"
      >
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-faint" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder='e.g. "Find datasets about crop yield in India between 2020 and 2025"'
            className="w-full rounded-md border border-border bg-surface py-2.5 pl-9 pr-3 text-sm text-text placeholder:text-faint focus:border-accent focus:outline-none"
            aria-label="Describe your ML problem"
          />
        </div>
        <Button type="submit" disabled={submitting || !query.trim()}>
          {submitting ? <Spinner className="border-t-text" /> : null}
          Discover datasets
        </Button>
      </form>

      {error ? (
        <ErrorState
          className="mt-4"
          what={(error as ApiError).message || "The discovery request could not be created."}
          why={error instanceof ApiError && error.code === "BACKEND_UNREACHABLE" ? "The backend service is not reachable." : undefined}
          whatToDo="Make sure the FastAPI backend is running on port 8000, then retry."
          onRetry={() => void submit(query)}
        />
      ) : null}

      {!compact ? (
        <div className="mt-8">
          <p className="mb-2 text-xs font-medium uppercase tracking-wider text-faint">Example prompts</p>
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                onClick={() => void submit(ex)}
                disabled={submitting}
                className="rounded-md border border-border bg-surface px-3 py-1.5 text-xs text-muted transition-colors hover:border-accent/40 hover:text-text disabled:opacity-50"
              >
                {ex}
              </button>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
