"use client";

import { useQuery } from "@tanstack/react-query";
import { Activity } from "lucide-react";
import { Show, SignInButton, SignUpButton, UserButton } from "@clerk/nextjs";
import Link from "next/link";
import { getDashboardStats } from "@/lib/api-client";
import { useClerkEnabled } from "@/app/providers";
import { StatusBadge } from "@/components/ui/Badge";

export function TopBar() {
  const clerkEnabled = useClerkEnabled();
  const { data } = useQuery({
    queryKey: ["dashboard", "stats"],
    queryFn: getDashboardStats,
    refetchInterval: 10_000,
  });

  const activeJobs = data?.active_job_list ?? [];

  return (
    <header className="sticky top-0 z-40 flex h-12 items-center justify-between border-b border-border bg-bg/95 px-4 backdrop-blur">
      <div className="flex items-center gap-2 text-sm">
        <span className="text-[11px] font-medium uppercase tracking-wider text-faint">Project</span>
        <span className="font-medium text-text">Default Project</span>
      </div>

      <div className="flex items-center gap-3">
        {activeJobs.length > 0 ? (
          <div className="flex items-center gap-2 rounded-md border border-border bg-surface px-2.5 py-1 text-xs">
            <Activity className="size-3.5 animate-pulse text-accent" />
            <span className="num text-muted">
              {activeJobs.length} active job{activeJobs.length === 1 ? "" : "s"}
            </span>
            {activeJobs.slice(0, 2).map((job) => (
              <StatusBadge key={job.id} status={job.status} />
            ))}
          </div>
        ) : (
          <span className="hidden items-center gap-1.5 rounded-md border border-border bg-surface px-2.5 py-1 text-xs text-faint sm:flex">
            <Activity className="size-3.5" /> No active jobs
          </span>
        )}

        {clerkEnabled ? (
          <>
            <Show when="signed-in">
              <UserButton />
            </Show>
            <Show when="signed-out">
              <SignInButton mode="redirect" fallbackRedirectUrl="/dashboard">
                <button className="rounded-md px-2.5 py-1.5 text-xs font-medium text-muted hover:bg-hover hover:text-text">
                  Sign in
                </button>
              </SignInButton>
              <SignUpButton mode="redirect" fallbackRedirectUrl="/dashboard">
                <button className="rounded-md border border-accent/60 bg-accent/15 px-2.5 py-1.5 text-xs font-medium text-accent hover:bg-accent/25">
                  Sign up
                </button>
              </SignUpButton>
            </Show>
          </>
        ) : (
          <Link
            href="/profile"
            className="flex size-7 items-center justify-center rounded-full bg-elevated text-xs font-semibold text-muted"
            title="Profile & Preferences"
          >
            LU
          </Link>
        )}
      </div>
    </header>
  );
}
