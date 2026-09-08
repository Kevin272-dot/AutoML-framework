"use client";
// src/app/search/page.tsx -- Dataset Search results
import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { searchDatasets, DatasetOut } from "@/lib/lib";

export default function SearchPage() {
  const router = useRouter();
  const params = useSearchParams();
  const query = params.get("query") || "";
  const dateStart = params.get("date_start") || undefined;
  const dateEnd = params.get("date_end") || undefined;

  const [results, setResults] = useState<DatasetOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!query) return;
    setLoading(true);
    setError(null);
    searchDatasets(query, dateStart, dateEnd)
      .then((res) => setResults(res.results))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [query, dateStart, dateEnd]);

  return (
    <main className="container">
      <div className="steps" style={{ marginBottom: "24px" }}>
        <span className="step active">1. Search</span>
        <span className="step">2. Preview</span>
        <span className="step">3. Configure</span>
        <span className="step">4. Train</span>
        <span className="step">5. Results</span>
      </div>

      <h1>Results for &ldquo;{query}&rdquo;</h1>
      <p className="subtitle">Ranked by suitability score — keyword match, date coverage, domain, completeness, size, and feature quality.</p>

      {loading && <p style={{ color: "var(--muted)" }}>Searching dataset sources…</p>}
      {error && <div className="error-box">{error}</div>}

      {!loading && !error && results.length === 0 && (
        <p style={{ color: "var(--muted)" }}>No datasets matched. Try a broader query.</p>
      )}

      <div className="grid-2">
        {results.map((ds) => (
          <div key={ds.id} className="card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <span className="card-title">{ds.name}</span>
              <span className="badge badge-score">{ds.suitability_score}% suitable</span>
            </div>
            <p className="card-desc">{ds.description}</p>
            <span className="badge badge-domain">{ds.domain}</span>
            <div className="meta-row">
              <span>{ds.rows.toLocaleString()} rows</span>
              <span>{ds.columns} columns</span>
              <span>{ds.date_start} → {ds.date_end}</span>
              <span>{ds.missing_pct}% missing</span>
              <span>{ds.file_format}</span>
            </div>
            <button
              className="btn btn-primary"
              style={{ marginTop: "16px" }}
              onClick={() => router.push(`/datasets/${ds.id}`)}
            >
              Preview Dataset
            </button>
          </div>
        ))}
      </div>
    </main>
  );
}
