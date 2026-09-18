"use client";

import { useEffect, useRef, useState } from "react";
import { Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";

/** Two-step delete button: first click arms it, second click confirms.
 * Auto-disarms after 4 seconds of inactivity. */
export function ConfirmDeleteButton({
  onConfirm,
  label = "Delete",
  confirmLabel = "Confirm delete",
  size = "sm",
  disabled,
  className,
}: {
  onConfirm: () => void | Promise<void>;
  label?: string;
  confirmLabel?: string;
  size?: "sm" | "md";
  disabled?: boolean;
  className?: string;
}) {
  const [armed, setArmed] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => {
    if (timer.current) clearTimeout(timer.current);
  }, []);

  const click = async () => {
    if (!armed) {
      setArmed(true);
      timer.current = setTimeout(() => setArmed(false), 4000);
      return;
    }
    if (timer.current) clearTimeout(timer.current);
    setArmed(false);
    await onConfirm();
  };

  return (
    <button
      onClick={() => void click()}
      disabled={disabled}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border font-medium transition-colors",
        "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-danger",
        "disabled:cursor-not-allowed disabled:opacity-50",
        size === "sm" ? "px-2.5 py-1.5 text-xs" : "px-3.5 py-2 text-sm",
        armed
          ? "border-danger bg-danger/20 text-danger"
          : "border-border bg-elevated text-muted hover:border-danger/50 hover:text-danger",
        className
      )}
      aria-label={armed ? confirmLabel : label}
    >
      <Trash2 className="size-3.5" />
      {armed ? confirmLabel : label}
    </button>
  );
}
