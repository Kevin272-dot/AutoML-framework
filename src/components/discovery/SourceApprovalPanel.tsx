"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ExternalLink, ShieldCheck, ShieldAlert, ShieldQuestion } from "lucide-react";
import { approveSources, getRequestSources } from "@/lib/api-client";
import type { SourceWithAuditOut } from "@/lib/api-types";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { CheckIcon, ErrorState } from "@/components/ui/States";
import { Spinner } from "@/components/ui/Card";

function CapabilityRow({ label, ok }: { label: string; ok: boolean | null }) {
  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-muted">{label}</span>
      <CheckIcon ok={ok} />
    </div>
  );
}

function PolicyIcon({ status }: { status: string }) {
  if (status === "RESTRICTED") return <ShieldAlert className="size-3.5 text-danger" />;
  if (status === "KNOWN") return <ShieldCheck className="size-3.5 text-success" />;
  return <ShieldQuestion className="size-3.5 text-warning" />;
}

function isEligible(s: SourceWithAuditOut): boolean {
  return (
    !!s.audit &&
    ["API_AVAILABLE", "DOWNLOAD_AVAILABLE", "ELIGIBLE"].includes(s.audit.technical_status) &&
    !s.requires_auth
  );
}

const TECH_LABELS: Record<string, string> = {
  API_AVAILABLE: "Available via official API",
  DOWNLOAD_AVAILABLE: "Available via official download",
  AUTH_REQUIRED: "Requires source credentials (not yet supported)",
  ROBOTS_DISALLOWED: "Automated access disallowed by robots.txt",
  SOURCE_UNREACHABLE: "Source unreachable",
  UNSUPPORTED: "No usable machine-readable access",
};

function SourceCard({
  source,
  selected,
  onToggle,
}: {
  source: SourceWithAuditOut;
  selected: boolean;
  onToggle: () => void;
}) {
  const audit = source.audit;
  const eligible = isEligible(source);
  return (
    <label
      className={[
        "flex cursor-pointer flex-col rounded-lg border bg-surface p-4 transition-colors",
        selected ? "border-accent/60" : "border-border hover:border-border-strong",
        !eligible && "cursor-not-allowed opacity-60",
      ].join(" ")}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-2.5">
          <input
            type="checkbox"
            checked={selected}
            disabled={!eligible}
            onChange={onToggle}
            className="mt-1 size-3.5 accent-[var(--color-accent)]"
            aria-label={`Approve ${source.name}`}
          />
          <div>
            <div className="text-sm font-semibold text-text">{source.name}</div>
            <div className="mt-0.5 flex items-center gap-2 text-xs text-muted">
              <Badge tone="neutral">{source.source_type.replaceAll("_", " ").toLowerCase()}</Badge>
              <a
                href={source.base_url}
                target="_blank"
                rel="noreferrer"
                onClick={(e) => e.preventDefault()}
                className="inline-flex items-center gap-0.5 hover:text-accent"
              >
                {source.base_url.replace(/^https?:\/\//, "")} <ExternalLink className="size-3" />
              </a>
            </div>
          </div>
        </div>
        <span
          className={[
            "rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide",
            eligible ? "border-success/40 bg-success/10 text-success" : "border-danger/40 bg-danger/10 text-danger",
          ].join(" ")}
        >
          {eligible ? "Eligible" : "Unavailable"}
        </span>
      </div>

      {audit ? (
        <div className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 border-t border-border pt-3">
          <CapabilityRow label="API" ok={audit.api_available} />
          <CapabilityRow label="Search" ok={audit.search_available} />
          <CapabilityRow label="Metadata" ok={audit.metadata_available} />
          <CapabilityRow label="Preview" ok={audit.preview_available} />
          <CapabilityRow label="Download" ok={audit.download_available} />
          <CapabilityRow label="Robots" ok={audit.robots_allowed} />
        </div>
      ) : (
        <div className="mt-3 border-t border-border pt-3 text-xs text-faint">Audit pending…</div>
      )}

      {audit ? (
        <div className="mt-3 space-y-1 border-t border-border pt-3 text-xs">
          <div className="flex items-center gap-1.5 text-muted">
            <PolicyIcon status={audit.policy_status} />
            <span>
              Technical access: <span className="text-text">{TECH_LABELS[audit.technical_status] ?? audit.technical_status}</span>
            </span>
          </div>
          <div className="text-faint">
            Policy information: <span className="text-muted">{audit.policy_status.toLowerCase()}</span>
            {" · "}Terms: {audit.terms_status.toLowerCase()}
            {" · "}License: {audit.license_status.toLowerCase()}
          </div>
          {audit.notes.length > 0 ? (
            <ul className="list-inside list-disc text-faint">
              {audit.notes.slice(0, 2).map((n, i) => (
                <li key={i}>{n}</li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
    </label>
  );
}

export function SourceApprovalPanel({
  requestId,
  onApproved,
}: {
  requestId: string;
  onApproved: (approvedIds: string[]) => void;
}) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [approving, setApproving] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const { data: sources, isLoading, isError, error: queryError, refetch } = useQuery({
    queryKey: ["discovery", requestId, "sources"],
    queryFn: () => getRequestSources(requestId),
  });

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 py-10 text-sm text-muted">
        <Spinner /> Auditing sources…
      </div>
    );
  }
  if (isError || !sources) {
    return (
      <ErrorState
        what={(queryError as Error)?.message ?? "Could not load audited sources."}
        whatToDo="Retry, or start a new discovery request."
        onRetry={() => void refetch()}
      />
    );
  }

  const eligibleSources = sources.filter(isEligible);
  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const approve = async () => {
    setApproving(true);
    setError(null);
    try {
      const result = await approveSources(requestId, Array.from(selected));
      onApproved(result.approved_source_ids);
    } catch (err) {
      setError(err as Error);
    } finally {
      setApproving(false);
    }
  };

  return (
    <div>
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold">Choose the sources we may search</h2>
          <p className="mt-0.5 text-xs text-muted">
            {sources.length} source(s) discovered and audited. Only the sources you approve here will be
            queried. Nothing has been searched yet.
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setSelected(new Set(eligibleSources.map((s) => s.id)))}
          >
            Select all eligible
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setSelected(new Set())}>
            Clear
          </Button>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        {sources.map((s) => (
          <SourceCard
            key={s.id}
            source={s}
            selected={selected.has(s.id)}
            onToggle={() => toggle(s.id)}
          />
        ))}
      </div>

      {error ? (
        <ErrorState
          className="mt-4"
          what={(error as Error).message || "Source approval failed."}
          whatToDo="Check the selected sources and retry."
          onRetry={() => void approve()}
        />
      ) : null}

      <div className="mt-4 flex items-center justify-between rounded-lg border border-border bg-surface px-4 py-3">
        <p className="text-xs text-muted">
          <span className="num font-medium text-text">{selected.size}</span> source(s) selected
          {selected.size === 0 ? " — select at least one to continue" : ""}
        </p>
        <Button onClick={() => void approve()} disabled={selected.size === 0 || approving}>
          {approving ? <Spinner className="border-t-text" /> : null}
          Approve selected sources
        </Button>
      </div>
    </div>
  );
}
