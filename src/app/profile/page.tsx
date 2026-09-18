"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { KeyRound, ShieldCheck, ShieldOff } from "lucide-react";
import { UserProfile } from "@clerk/nextjs";
import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme, type ThemePreference } from "@/components/theme/ThemeProvider";
import { useClerkEnabled } from "@/app/providers";
import {
  deleteConnection,
  listConnections,
  listSources,
  saveConnection,
  ApiError,
} from "@/lib/api-client";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardHeader, Spinner } from "@/components/ui/Card";
import { ConfirmDeleteButton } from "@/components/ui/ConfirmDeleteButton";
import { ErrorState } from "@/components/ui/States";
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

const KEY_HINTS: Record<string, string> = {
  kaggle: "Format: username:key (from kaggle.com/settings → API)",
  "data-gov-in": "Your data.gov.in API key",
  huggingface: "A Hugging Face access token (hf_…)",
};

function ConnectionsCard() {
  const queryClient = useQueryClient();
  const [secretInputs, setSecretInputs] = useState<Record<string, string>>({});
  const [savingSource, setSavingSource] = useState<string | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const { data: sources } = useQuery({ queryKey: ["sources"], queryFn: listSources });
  const { data: connections, isLoading } = useQuery({ queryKey: ["connections"], queryFn: listConnections });

  const authSources = (sources ?? []).filter((s) => s.requires_auth);
  const bySource = new Map((connections ?? []).map((c) => [c.source_id, c]));

  const save = async (sourceId: string) => {
    const secret = secretInputs[sourceId]?.trim();
    if (!secret) return;
    setSavingSource(sourceId);
    setError(null);
    try {
      await saveConnection(sourceId, secret);
      setSecretInputs((prev) => ({ ...prev, [sourceId]: "" }));
      await queryClient.invalidateQueries({ queryKey: ["connections"] });
    } catch (err) {
      setError(err as Error);
    } finally {
      setSavingSource(null);
    }
  };

  const remove = async (sourceId: string) => {
    setError(null);
    try {
      await deleteConnection(sourceId);
      await queryClient.invalidateQueries({ queryKey: ["connections"] });
    } catch (err) {
      setError(err as Error);
    }
  };

  if (!authSources.length) return null;

  return (
    <Card>
      <CardHeader
        title="Connections"
        subtitle="API keys for sources that require credentials. Stored as protected, encrypted secrets — never exposed to the browser."
      />
      <div className="space-y-3 p-4">
        {isLoading ? (
          <div className="flex items-center gap-2 text-sm text-muted"><Spinner /> Loading connections…</div>
        ) : (
          authSources.map((source) => {
            const conn = bySource.get(source.id);
            return (
              <div key={source.id} className="rounded-lg border border-border p-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <KeyRound className="size-4 text-faint" />
                    <span className="text-sm font-medium text-text">{source.name}</span>
                    {conn ? (
                      <Badge tone={conn.validated ? "success" : "warning"}>
                        {conn.validated ? "validated" : "connected"}
                      </Badge>
                    ) : (
                      <Badge tone="neutral">not connected</Badge>
                    )}
                  </div>
                  {conn ? (
                    <span className="flex items-center gap-2 text-xs text-faint">
                      <ShieldCheck className="size-3.5 text-success" /> {conn.secret_hint}
                      <ConfirmDeleteButton label="Remove" confirmLabel="Confirm remove" onConfirm={() => void remove(source.id)} />
                    </span>
                  ) : (
                    <ShieldOff className="size-3.5 text-faint" />
                  )}
                </div>
                <div className="mt-2 flex gap-2">
                  <input
                    type="password"
                    value={secretInputs[source.id] ?? ""}
                    onChange={(e) => setSecretInputs((prev) => ({ ...prev, [source.id]: e.target.value }))}
                    placeholder={conn ? "Replace with a new key…" : (KEY_HINTS[source.slug] ?? "API key")}
                    className="flex-1 rounded-md border border-border bg-elevated px-3 py-1.5 text-xs text-text placeholder:text-faint focus:border-accent focus:outline-none"
                    aria-label={`API key for ${source.name}`}
                  />
                  <Button
                    size="sm"
                    onClick={() => void save(source.id)}
                    disabled={savingSource === source.id || !(secretInputs[source.id] ?? "").trim()}
                  >
                    {savingSource === source.id ? <Spinner className="border-t-text" /> : null}
                    Save key
                  </Button>
                </div>
              </div>
            );
          })
        )}
        {error ? (
          <ErrorState
            what={(error as ApiError).message || "Could not save the connection."}
            why={error instanceof ApiError && error.code === "SECRET_INVALID" ? "The source rejected this key." : undefined}
            whatToDo="Check the key and retry."
          />
        ) : null}
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

      <ConnectionsCard />

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
