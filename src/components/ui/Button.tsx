import Link from "next/link";
import { cn } from "@/lib/utils";

export function Button({
  children,
  variant = "primary",
  size = "md",
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md";
}) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-1.5 rounded-md border font-medium transition-colors",
        "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent",
        "disabled:cursor-not-allowed disabled:opacity-50",
        size === "sm" ? "px-2.5 py-1.5 text-xs" : "px-3.5 py-2 text-sm",
        variant === "primary" && "border-accent/60 bg-accent/15 text-accent hover:bg-accent/25",
        variant === "secondary" && "border-border bg-elevated text-text hover:border-border-strong hover:bg-hover",
        variant === "ghost" && "border-transparent text-muted hover:bg-hover hover:text-text",
        variant === "danger" && "border-danger/50 bg-danger/10 text-danger hover:bg-danger/20",
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}

export function ButtonLink({
  children,
  href,
  variant = "primary",
  size = "md",
  className,
}: {
  children: React.ReactNode;
  href: string;
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md";
  className?: string;
}) {
  return (
    <Link
      href={href}
      className={cn(
        "inline-flex items-center justify-center gap-1.5 rounded-md border font-medium transition-colors",
        size === "sm" ? "px-2.5 py-1.5 text-xs" : "px-3.5 py-2 text-sm",
        variant === "primary" && "border-accent/60 bg-accent/15 text-accent hover:bg-accent/25",
        variant === "secondary" && "border-border bg-elevated text-text hover:border-border-strong hover:bg-hover",
        variant === "ghost" && "border-transparent text-muted hover:bg-hover hover:text-text",
        className
      )}
    >
      {children}
    </Link>
  );
}
