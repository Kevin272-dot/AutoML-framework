"use client";

import { use, useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getDiscoveryRequest, getJob, getSearchResults, startSearch } from "@/lib/api-client";
import type { RankedCandidateOut } from "@/lib/api-types";
import { Stepper } from "@/components/ui/Stepper";
import { StatusBadge } from "@/components/ui/Badge";
import { Spinner } from "@/components/ui/Card";
import { ErrorState } from "@/components/ui/States";
import { SourceApprovalPanel } from "@/components/discovery/SourceApprovalPanel";
import { ResultsTable } from "@/components/discovery/ResultsTable";

function RequirementsSummary({ req }: { req: Awaited<ReturnType<typeof getDiscoveryRequest>> }) {
  const p = req.parsed_requirements;
  const chips: [string, string][] = [];
  if (p.domain.length) chips.push(["Domain", p.domain.join(", ")]);
  if (p.location.length) chips.push(["Location", p.location.join(", ")]);
  if (p.date_start || p.date_end)
    chips.push(["Date range", `${p.date_start ?? "…"} → ${p.date_end ?? "…"}`]);
  if (p.task) chips.push(["Task", p.task]);
  if (p.keywords.length) chips.push(["Keywords", p.keywords.join(", ")]);
  if (p.min_rows || p.max_rows)
    chips.push(["Rows", `${p.min_rows ?? "any"} – ${p.max_rows ?? "any"}`]);
  if (p.preferred_format) chips.push(["Format", p.preferred_format]);

  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold">Understanding your request</h2>
        <span className="text-[10px] uppercase tracking-wider text-faint">parser: {p.parser}</span>
      </div>
      <p className="mt-1 text-xs text-muted">&ldquo;{req.query}&rdquo;</p>
      <div className="mt-3 flex flex-wrap gap-1.5">
        {chips.length === 0 ? (
          <span className="text-xs text-faint">No specific constraints detected — all sources will be considered.</span>
        ) : (
          chips.map(([label, value]) => (
            <span key={label} className="rounded border border-border bg-elevated px-2 py-1 text-xs">
              <span className="text-faint">{label}: </span>
              <span className="text-text">{value}</span>
            </span>
          ))
        )}
      </div>
    </div>
  );
}

export default function DiscoveryWizardPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [searchJobId, setSearchJobId] = useState<string | null>(null);
  const [approved, setApproved] = useState(false);
  const [searchError, setSearchError] = useState<Error | null>(null);

  const { data: request, isLoading, isError, error, refetch } = useQuery({
    queryKey: ["discovery", id],
    queryFn: () => getDiscoveryRequest(id),
    refetchInterval: (q) => {
      const status = q.state.data?.status;
      if (status === "DATASET_SEARCHING") return 2000;
      return 10000;
    },
  });

  // Poll the search job while it runs.
  const { data: job } = useQuery({
    queryKey: ["discovery", id, "search-job", searchJobId],
    queryFn: () => getJob(searchJobId!),
    enabled: !!searchJobId,
    refetchInterval: 1500,
  });

  // When the search job completes, invalidate the request so status moves to DATASETS_READY.
  useEffect(() => {
    if (job?.status === "COMPLETED") {
      void refetch();
    }
  }, [job?.status, refetch]);

  // Auto-refetch results when the request becomes ready.
  const { data: results } = useQuery({
    queryKey: ["discovery", id, "results"],
    queryFn: () => getSearchResults(id),
    enabled: request?.status === "DATASETS_READY" || !!request?.status.match(/DATASET_SELECTED|DATASET_READY|WAITING_FOR_DATASET_SELECTION/),
  });

  const beginSearch = async (approvedIds: string[]) => {
    setApproved(true);
    setSearchError(null);
    try {
      const res = await startSearch(id);
      setSearchJobId(res.job_id);
      void approvedIds;
      await refetch();
    } catch (err) {
      setSearchError(err as Error);
      setApproved(false);
    }
  };

  const retrySearch = async () => {
    setSearchError(null);
    try {
      const res = await startSearch(id);
      setSearchJobId(res.job_id);
      await refetch();
    } catch (err) {
      setSearchError(err as Error);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 pt-20 text-sm text-muted">
        <Spinner /> Loading discovery request…
      </div>
    );
  }
  if (isError || !request) {
    return (
      <ErrorState
        title="Discovery request not found"
        what={(error as Error)?.message ?? "This discovery request does not exist."}
        whatToDo="Start a new discovery from the Dataset Discovery page."
      />
    );
  }
  if (request.status === "FAILED") {
    return (
      <ErrorState
        title="Discovery failed"
        what={request.error_message ?? "The discovery pipeline failed."}
        why={request.error_code ?? undefined}
        whatToDo="Try a different query or retry the search."
        onRetry={() => void refetch()}
      />
    );
  }

  const searching = request.status === "DATASET_SEARCHING" || (approved && !!searchJobId && searchJobId !== "" && (job?.status ?? "QUEUED") !== "COMPLETED");
  const showResults = request.status === "DATASETS_READY" || request.status === "WAITING_FOR_DATASET_SELECTION" ||
    request.status === "DATASET_SELECTED" || request.status === "DATASET_DOWNLOADING" || request.status === "DATASET_READY";

  return (
    <div className="mx-auto max-w-5xl space-y-5">
      <Stepper status={request.status} />
      <RequirementsSummary req={request} />

      {request.status === "WAITING_FOR_SOURCE_APPROVAL" ? (
        <SourceApprovalPanel requestId={id} onApproved={(ids) => void beginSearch(ids)} />
      ) : null}

      {searching ? (
        <div className="rounded-lg border border-border bg-surface p-6">
          <div className="flex items-center gap-2 text-sm">
            <Spinner /> Searching only your approved sources…
          </div>
          {job?.detail ? <p className="mt-2 text-xs text-muted">{job.detail}</p> : null}
          {job?.status === "FAILED" ? (
            <ErrorState
              className="mt-3"
              what={job.error_message ?? "The dataset search failed."}
              why={job.error_code ?? undefined}
              whatToDo="Retry the search, or adjust the query."
              onRetry={() => void retrySearch()}
            />
          ) : null}
        </div>
      ) : null}

      {searchError ? (
        <ErrorState
          what={searchError.message}
          whatToDo="Retry the search."
          onRetry={() => void retrySearch()}
        />
      ) : null}

      {showResults && results ? (
        <div>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-base font-semibold">
              {results.length} dataset{results.length === 1 ? "" : "s"} found
              {results.length > 0 ? " across your approved sources" : ""}
            </h2>
            <StatusBadge status={request.status} />
          </div>
          {results.length === 0 ? (
            <div className="rounded-lg border border-dashed border-border bg-surface px-6 py-10 text-center text-sm text-muted">
              No datasets matched. Try a broader query or approve more sources.
            </div>
          ) : (
            <ResultsTable results={results as RankedCandidateOut[]} />
          )}
        </div>
      ) : null}
    </div>
  );
}
