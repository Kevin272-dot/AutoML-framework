"use client";
// src/app/datasets/[id]/page.tsx -- Dataset Preview
import { useEffect, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import { previewDataset, selectDataset, DatasetPreview } from "@/lib/lib";

export default function DatasetPreviewPage() {
  const router = useRouter();
  const params = useParams();
  const id = params.id as string;

  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selecting, setSelecting] = useState(false);

  useEffect(() => {
    previewDataset(id)
      .then(setPreview)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [id]);

  const handleUseDataset = async () => {
    setSelecting(true);
    try {
      await selectDataset(id);
      router.push(`/automl/configure/${id}`);
    } catch (err) {
      setError((err as Error).message);
      setSelecting(false);
    }
  };

  if (loading) return <main className="container"><p style={{ color: "var(--muted)" }}>Loading preview…</p></main>;
  if (error) return <main className="container"><div className="error-box">{error}</div></main>;
  if (!preview) return null;

  return (
    <main className="container">
      <div className="steps" style={{ marginBottom: "24px" }}>
        <span className="step">1. Search</span>
        <span className="step active">2. Preview</span>
        <span className="step">3. Configure</span>
        <span className="step">4. Train</span>
        <span className="step">5. Results</span>
      </div>

      <h1>{preview.name}</h1>
      <div className="meta-row" style={{ marginBottom: "24px" }}>
        <span>{preview.shape.rows.toLocaleString()} rows</span>
        <span>{preview.shape.columns} columns</span>
      </div>

      <h2>Columns &amp; Types</h2>
      <table>
        <thead><tr><th>Column</th><th>Type</th><th>Missing %</th></tr></thead>
        <tbody>
          {preview.columns.map((col) => (
            <tr key={col}>
              <td>{col}</td>
              <td>{preview.dtypes[col]}</td>
              <td>{preview.missing_summary[col] ?? 0}%</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2 style={{ marginTop: "32px" }}>Sample Rows</h2>
      <div style={{ overflowX: "auto" }}>
        <table>
          <thead>
            <tr>{preview.columns.map((c) => <th key={c}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {preview.sample_rows.map((row, i) => (
              <tr key={i}>
                {preview.columns.map((c) => (
                  <td key={c}>{typeof row[c] === "number" ? (row[c] as number).toFixed(2) : String(row[c])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <button className="btn btn-primary" style={{ marginTop: "24px" }} onClick={handleUseDataset} disabled={selecting}>
        {selecting ? "Selecting…" : "Use This Dataset"}
      </button>
    </main>
  );
}
