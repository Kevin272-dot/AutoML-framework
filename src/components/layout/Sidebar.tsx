"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import {
  Beaker,
  Bot,
  Boxes,
  ChevronsLeft,
  Compass,
  Database,
  LayoutDashboard,
  Settings,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/discovery", label: "Dataset Discovery", icon: Compass },
  { href: "/datasets", label: "Datasets", icon: Database },
  { href: "/experiments", label: "Experiments", icon: Beaker },
  { href: "/automl-runs", label: "AutoML Runs", icon: Bot },
  { href: "/models", label: "Models", icon: Boxes },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={cn(
        "sticky top-0 flex h-screen shrink-0 flex-col border-r border-border bg-surface transition-[width] duration-150",
        collapsed ? "w-14" : "w-56"
      )}
    >
      <div className={cn("flex h-12 items-center border-b border-border", collapsed ? "justify-center px-0" : "justify-between px-4")}>
        {!collapsed ? (
          <Link href="/dashboard" className="flex items-center gap-2">
            <span className="flex size-6 items-center justify-center rounded bg-accent/15 text-xs font-bold text-accent">IA</span>
            <span className="text-sm font-semibold tracking-tight">Intelligent AutoML</span>
          </Link>
        ) : (
          <span className="flex size-6 items-center justify-center rounded bg-accent/15 text-xs font-bold text-accent">IA</span>
        )}
      </div>

      <nav className="flex-1 space-y-0.5 overflow-y-auto p-2" aria-label="Main navigation">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || pathname.startsWith(`${href}/`);
          return (
            <Link
              key={href}
              href={href}
              title={collapsed ? label : undefined}
              className={cn(
                "flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm transition-colors",
                active
                  ? "bg-accent/10 font-medium text-accent"
                  : "text-muted hover:bg-hover hover:text-text",
                collapsed && "justify-center px-0"
              )}
            >
              <Icon className="size-4 shrink-0" />
              {!collapsed ? label : null}
            </Link>
          );
        })}
      </nav>

      <button
        onClick={() => setCollapsed((c) => !c)}
        className="flex items-center justify-center border-t border-border py-2.5 text-faint hover:bg-hover hover:text-text"
        aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        <ChevronsLeft className={cn("size-4 transition-transform", collapsed && "rotate-180")} />
      </button>
    </aside>
  );
}
