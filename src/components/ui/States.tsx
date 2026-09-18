import { AlertTriangle, Check, CircleSlash, Inbox } from "lucide-react";
import { cn } from "@/lib/utils";
import { ButtonLink } from "./Button";

export function EmptyState({
  title,
  description,
  action,
  className,
}: {
  title: string;
  description?: string;
  action?: { href: string; label: string };
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col items-center justify-center rounded-lg border border-dashed border-border bg-surface px-6 py-16 text-center", className)}>
      <Inbox className="size-8 text-faint" />
      <h3 className="mt-3 text-sm font-semibold text-text">{title}</h3>
      {description ? <p className="mt-1 max-w-sm text-xs text-muted">{description}</p> : null}
      {action ? (
        <ButtonLink href={action.href} variant="secondary" size="sm" className="mt-4">
          {action.label}
        </ButtonLink>
      ) : null}
    </div>
  );
}

export function ErrorState({
  title = "Something went wrong",
  what,
  why,
  whatToDo,
  onRetry,
  className,
}: {
  title?: string;
  what: string;
  why?: string;
  whatToDo?: string;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <div className={cn("rounded-lg border border-danger/40 bg-danger/5 px-5 py-4", className)} role="alert">
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 size-4 shrink-0 text-danger" />
        <div className="text-sm">
          <h3 className="font-semibold text-text">{title}</h3>
          <p className="mt-1 text-text/90">{what}</p>
          {why ? (
            <p className="mt-1 text-xs text-muted">
              <span className="font-medium text-muted">Why:</span> {why}
            </p>
          ) : null}
          {whatToDo ? (
            <p className="mt-0.5 text-xs text-muted">
              <span className="font-medium text-muted">What you can do:</span> {whatToDo}
            </p>
          ) : null}
          {onRetry ? (
            <button
              onClick={onRetry}
              className="mt-2 rounded-md border border-border bg-elevated px-2.5 py-1.5 text-xs font-medium text-text hover:bg-hover"
            >
              Retry
            </button>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export function NotAvailableYet({ feature, description }: { feature: string; description: string }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border bg-surface px-6 py-20 text-center">
      <CircleSlash className="size-8 text-faint" />
      <h3 className="mt-3 text-sm font-semibold text-text">{feature} is not available yet</h3>
      <p className="mt-1 max-w-md text-xs text-muted">{description}</p>
    </div>
  );
}

export function CheckIcon({ ok }: { ok: boolean | null }) {
  if (ok === null) return <span className="text-xs text-faint">?</span>;
  if (ok) return <Check className="size-3.5 text-success" aria-label="yes" />;
  return <span className="text-xs text-danger" aria-label="no">✕</span>;
}
