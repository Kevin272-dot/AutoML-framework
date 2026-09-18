import { Card, CardHeader } from "@/components/ui/Card";

const SETTINGS = [
  ["Backend API URL", "http://localhost:8000", "Set via NEXT_PUBLIC_API_URL"],
  ["Job backend", "eager (in-process) — switch to celery when Docker/Redis is available", "Set via JOB_BACKEND=celery"],
  ["Artifact storage", "local filesystem — switch to S3/MinIO when Docker is available", "Set via USE_S3_STORAGE=true"],
  ["Database", "SQLite (dev profile) — switch to PostgreSQL via DATABASE_URL", "Set via DATABASE_URL"],
  ["Max download size", "512 MB", "Set via MAX_DATASET_DOWNLOAD_BYTES"],
];

export default function SettingsPage() {
  return (
    <div className="mx-auto max-w-3xl">
      <h1 className="text-lg font-semibold tracking-tight">Settings</h1>
      <p className="mt-0.5 mb-4 text-sm text-muted">
        Current environment configuration. All values are set via backend environment variables (see
        <code className="mx-1 rounded bg-elevated px-1 py-0.5 text-xs">backend/.env</code>).
      </p>
      <Card>
        <CardHeader title="Environment" subtitle="Effective configuration for this deployment" />
        <dl className="divide-y divide-border text-sm">
          {SETTINGS.map(([name, value, hint]) => (
            <div key={name} className="flex items-start justify-between gap-4 px-4 py-3">
              <div>
                <dt className="text-text">{name}</dt>
                <dd className="text-xs text-faint">{hint}</dd>
              </div>
              <code className="max-w-[45%] truncate rounded bg-elevated px-2 py-1 text-xs text-muted">{value}</code>
            </div>
          ))}
        </dl>
      </Card>
    </div>
  );
}
