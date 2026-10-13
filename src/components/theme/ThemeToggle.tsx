"use client";

import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme } from "@/components/theme/ThemeProvider";
import { cn } from "@/lib/utils";

/** Quick theme cycle: light -> dark -> system. Shown in the topbar. */
export function ThemeToggle({ className }: { className?: string }) {
  const { theme, resolvedTheme, setTheme } = useTheme();

  const cycle = () => {
    if (theme === "light") setTheme("dark");
    else if (theme === "dark") setTheme("system");
    else setTheme("light");
  };

  const label =
    theme === "system"
      ? `System (${resolvedTheme}) — click for light`
      : theme === "dark"
        ? "Dark — click for system"
        : "Bright — click for dark";

  return (
    <button
      onClick={cycle}
      title={label}
      aria-label={label}
      className={cn(
        "relative flex size-7 items-center justify-center rounded-md border border-border bg-surface text-muted",
        "transition-colors hover:border-border-strong hover:text-text",
        className
      )}
    >
      <Sun
        className={cn(
          "absolute size-3.5 transition-all duration-200",
          theme === "light" ? "rotate-0 scale-100 opacity-100" : "-rotate-90 scale-0 opacity-0"
        )}
      />
      <Moon
        className={cn(
          "absolute size-3.5 transition-all duration-200",
          theme === "dark" ? "rotate-0 scale-100 opacity-100" : "rotate-90 scale-0 opacity-0"
        )}
      />
      <Monitor
        className={cn(
          "absolute size-3.5 transition-all duration-200",
          theme === "system" ? "scale-100 opacity-100" : "scale-0 opacity-0"
        )}
      />
    </button>
  );
}
