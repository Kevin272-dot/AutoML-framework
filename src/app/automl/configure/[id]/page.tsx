"use client";
// src/app/automl/configure/[id]/page.tsx -- AutoML Configuration
import { useEffect, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import { previewDataset, runAutoML, DatasetPreview } from "@/lib/lib";

export default function ConfigurePage() {
  const router = useRouter();
  const params = useParams();
  const id = params.id as string;

  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    previewDataset(id)
      .then((p) => {
        setPreview(p);
        setTargetColumn(p.columns[p.columns.length - 1]);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [id]);

  const handleStart = async () => {
    setStarting(true);
    setError(null);
    try {
      const job = await runAutoML(id, targetColumn);
      router.push(`/automl/training/${job.job_id}`);
    } catch (err) {
      setError((err as Error).message);
      setStarting(false);
    }
  };

  if (loading) return <main className="container"><p style={{ color: "var(--muted)" }}>Loading dataset…</p></main>;
  if (!preview) return <main className="container"><div className="error-box">{error}</div></main>;

  return (
    <main className="container">
      <div className="steps" style={{ marginBottom: "24px" }}>
        <span className="step">1. Search</span>
        <span className="step">2. Preview</span>
        <span className="step active">3. Configure</span>
        <span className="step">4. Train</span>
        <span className="step">5. Results</span>
      </div>

      <h1>Configure AutoML Run</h1>
      <p className="subtitle">Confirm the target column. Task type (classification vs. regression) and preprocessing are detected automatically.</p>

      {error && <div className="error-box">{error}</div>}

      <div className="card" style={{ maxWidth: "500px" }}>
        <label style={{ display: "block", marginBottom: "8px", fontWeight: 600, fontSize: ".9rem" }}>
          Target column
        </label>
        <select value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)}>
          {preview.columns.map((col) => (
            <option key={col} value={col}>{col}</option>
          ))}
        </select>
        <p style={{ color: "var(--muted)", fontSize: ".82rem", marginTop: "10px" }}>
          Missing values will be imputed, categorical features encoded, and numeric
          features scaled automatically before training.
        </p>
      </div>

      <button className="btn btn-primary" style={{ marginTop: "20px" }} onClick={handleStart} disabled={starting}>
        {starting ? "Starting…" : "Run AutoML"}
      </button>
    </main>
  );
}
