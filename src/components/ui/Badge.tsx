import { cn } from "@/lib/utils";

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: React.ReactNode;
  tone?: "neutral" | "accent" | "success" | "warning" | "danger";
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[11px] font-medium uppercase tracking-wide",
        tone === "neutral" && "border-border bg-elevated text-muted",
        tone === "accent" && "border-accent/40 bg-accent/10 text-accent",
        tone === "success" && "border-success/40 bg-success/10 text-success",
        tone === "warning" && "border-warning/40 bg-warning/10 text-warning",
        tone === "danger" && "border-danger/40 bg-danger/10 text-danger",
        className
      )}
    >
      {children}
    </span>
  );
}

const STATUS_TONES: Record<string, "neutral" | "accent" | "success" | "warning" | "danger"> = {
  COMPLETED: "success",
  DATASET_READY: "success",
  RUNNING: "accent",
  SEARCHING: "accent",
  DATASET_SEARCHING: "accent",
  QUEUED: "neutral",
  PENDING: "neutral",
  FAILED: "danger",
  ERROR: "danger",
  WAITING_FOR_SOURCE_APPROVAL: "warning",
  WAITING_FOR_DATASET_SELECTION: "warning",
};

export function StatusBadge({ status, className }: { status: string; className?: string }) {
  const tone = STATUS_TONES[status] ?? "neutral";
  return (
    <Badge tone={tone} className={cn("normal-case tracking-normal", className)}>
      {status.replaceAll("_", " ").toLowerCase()}
    </Badge>
  );
}
