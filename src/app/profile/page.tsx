"use client";

import { UserProfile } from "@clerk/nextjs";
import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme, type ThemePreference } from "@/components/theme/ThemeProvider";
import { useClerkEnabled } from "@/app/providers";
import { Card, CardHeader } from "@/components/ui/Card";
import { cn } from "@/lib/utils";

const THEME_OPTIONS: { value: ThemePreference; label: string; description: string; icon: typeof Sun }[] = [
  { value: "light", label: "Bright", description: "Default appearance", icon: Sun },
  { value: "dark", label: "Dark", description: "Low-light workspace", icon: Moon },
  { value: "system", label: "System", description: "Match your OS setting", icon: Monitor },
];

function PreferencesCard() {
  const { theme, setTheme } = useTheme();
  return (
    <Card>
      <CardHeader title="Preferences" subtitle="Appearance and workspace settings" />
      <div className="p-4">
        <h4 className="text-xs font-medium uppercase tracking-wider text-faint">Theme</h4>
        <div className="mt-2 grid gap-2 sm:grid-cols-3">
          {THEME_OPTIONS.map(({ value, label, description, icon: Icon }) => (
            <button
              key={value}
              onClick={() => setTheme(value)}
              className={cn(
                "flex items-start gap-2.5 rounded-lg border p-3 text-left transition-colors",
                theme === value ? "border-accent/60 bg-accent/5" : "border-border hover:border-border-strong"
              )}
              aria-pressed={theme === value}
            >
              <Icon className={cn("mt-0.5 size-4", theme === value ? "text-accent" : "text-faint")} />
              <span>
                <span className="block text-sm font-medium text-text">{label}</span>
                <span className="block text-xs text-faint">{description}</span>
              </span>
            </button>
          ))}
        </div>
        <p className="mt-2 text-xs text-faint">
          Bright is the default. Your choice is saved to this browser and applied before the page renders.
        </p>
      </div>
    </Card>
  );
}

export default function ProfilePage() {
  const clerkEnabled = useClerkEnabled();

  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <div>
        <h1 className="text-lg font-semibold tracking-tight">Profile</h1>
        <p className="mt-0.5 text-sm text-muted">Manage your account and workspace preferences.</p>
      </div>

      <PreferencesCard />

      {clerkEnabled ? (
        <Card className="overflow-hidden">
          <CardHeader title="Account" subtitle="Managed securely by Clerk" />
          <div className="flex justify-center p-4">
            <UserProfile routing="hash" />
          </div>
        </Card>
      ) : (
        <Card className="p-4">
          <h4 className="text-sm font-medium text-text">Account is not configured</h4>
          <p className="mt-1 text-xs text-muted">
            Add your Clerk keys to <code className="rounded bg-elevated px-1 py-0.5">.env.local</code> and restart the
            dev server to manage your account here.
          </p>
        </Card>
      )}
    </div>
  );
}
