"use client";

import { cn } from "@/lib/utils";
import { Check } from "lucide-react";

const STEPS = [
  "Understanding request",
  "Discovering sources",
  "Auditing sources",
  "Awaiting your approval",
  "Searching datasets",
  "Ranking datasets",
];

/** Maps a DiscoveryRequest status onto the visible wizard step. */
export function stepIndexForStatus(status: string): number {
  switch (status) {
    case "REQUEST_CREATED":
      return 0;
    case "REQUIREMENTS_PARSED":
    case "RESOURCES_DISCOVERED":
    case "RESOURCES_AUDITED":
      return 2;
    case "WAITING_FOR_SOURCE_APPROVAL":
      return 3;
    case "SOURCES_APPROVED":
    case "DATASET_SEARCHING":
      return 4;
    case "DATASETS_READY":
    case "WAITING_FOR_DATASET_SELECTION":
    case "DATASET_SELECTED":
    case "DATASET_DOWNLOADING":
    case "DATASET_READY":
      return 5;
    default:
      return 2;
  }
}

export function Stepper({ status, className }: { status: string; className?: string }) {
  const current = stepIndexForStatus(status);
  return (
    <ol className={cn("flex flex-wrap items-center gap-x-1 gap-y-2 text-xs", className)} aria-label="Discovery progress">
      {STEPS.map((label, i) => {
        const done = i < current;
        const active = i === current;
        return (
          <li key={label} className="flex items-center gap-1">
            {i > 0 ? <span className="mx-1 text-faint">→</span> : null}
            <span
              className={cn(
                "flex items-center gap-1.5 rounded border px-2 py-1",
                done && "border-success/30 bg-success/5 text-success",
                active && "border-accent/50 bg-accent/10 font-medium text-accent",
                !done && !active && "border-border text-faint"
              )}
            >
              {done ? <Check className="size-3" /> : <span className="num text-[10px] opacity-60">{i + 1}</span>}
              {label}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
