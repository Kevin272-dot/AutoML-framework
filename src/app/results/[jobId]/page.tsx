"use client";
// src/app/results/[jobId]/page.tsx -- Results Dashboard
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { getResults, predict, ResultsOut } from "@/lib/lib";

export default function ResultsPage() {
  const params = useParams();
  const jobId = params.jobId as string;

  const [results, setResults] = useState<ResultsOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [recordsInput, setRecordsInput] = useState("");
  const [predictions, setPredictions] = useState<(string | number)[] | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);

  useEffect(() => {
    getResults(jobId).then(setResults).catch((err) => setError(err.message));
  }, [jobId]);

  const handlePredict = async () => {
    setPredicting(true);
    setPredictError(null);
    setPredictions(null);
    try {
      const records = JSON.parse(recordsInput);
      const res = await predict(jobId, Array.isArray(records) ? records : [records]);
      setPredictions(res.predictions);
    } catch (err) {
      setPredictError((err as Error).message);
    } finally {
      setPredicting(false);
    }
  };

  if (error) return <main className="container"><div className="error-box">{error}</div></main>;
  if (!results) return <main className="container"><p style={{ color: "var(--muted)" }}>Loading results…</p></main>;

  const sortedLeaderboard = [...results.leaderboard].sort((a, b) => {
    const key = results.task === "classification" ? "f1" : "r2";
    return (b.metrics[key] ?? 0) - (a.metrics[key] ?? 0);
  });
  const maxImportance = Math.max(...Object.values(results.feature_importance), 0.0001);

  return (
    <main className="container">
      <div className="steps" style={{ marginBottom: "24px" }}>
        <span className="step">1. Search</span>
        <span className="step">2. Preview</span>
        <span className="step">3. Configure</span>
        <span className="step">4. Train</span>
        <span className="step active">5. Results</span>
      </div>

      <h1>Results</h1>
      <p className="subtitle">
        Task: <strong>{results.task}</strong> · Target: <strong>{results.target_column}</strong> · Best model: <strong style={{ color: "var(--accent2)" }}>{results.best_model}</strong>
      </p>

      <h2>Model Leaderboard</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            {Object.keys(sortedLeaderboard[0]?.metrics || {}).map((m) => <th key={m}>{m}</th>)}
            <th>Training time (s)</th>
          </tr>
        </thead>
        <tbody>
          {sortedLeaderboard.map((row) => (
            <tr key={row.model_name} className={row.model_name === results.best_model ? "best-row" : ""}>
              <td>{row.model_name}{row.model_name === results.best_model ? " 🏆" : ""}</td>
              {Object.values(row.metrics).map((v, i) => <td key={i}>{v}</td>)}
              <td>{row.training_time_sec}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2 style={{ marginTop: "32px" }}>Feature Importance</h2>
      {Object.keys(results.feature_importance).length === 0 ? (
        <p style={{ color: "var(--muted)" }}>Not available for the selected model type.</p>
      ) : (
        <div className="card">
          {Object.entries(results.feature_importance).map(([feature, importance]) => (
            <div key={feature} className="feature-bar-row">
              <span className="feature-label">{feature}</span>
              <div className="feature-bar-track">
                <div className="feature-bar-fill" style={{ width: `${(importance / maxImportance) * 100}%` }} />
              </div>
              <span>{importance}</span>
            </div>
          ))}
        </div>
      )}

      <h2 style={{ marginTop: "32px" }}>Try a Prediction</h2>
      <p className="subtitle">Paste a JSON record (or array of records) using the feature columns from the dataset preview.</p>
      <textarea
        rows={5}
        placeholder='{"vehicle_density": 0.5, "avg_speed": -0.2, "weather_index": 0.1, ...}'
        value={recordsInput}
        onChange={(e) => setRecordsInput(e.target.value)}
        style={{ width: "100%", fontFamily: "monospace", marginBottom: "12px" }}
      />
      <button className="btn btn-primary" onClick={handlePredict} disabled={predicting || !recordsInput.trim()}>
        {predicting ? "Predicting…" : "Predict"}
      </button>

      {predictError && <div className="error-box" style={{ marginTop: "16px" }}>{predictError}</div>}
      {predictions && (
        <div className="card" style={{ marginTop: "16px" }}>
          <strong>Predictions:</strong> {predictions.join(", ")}
        </div>
      )}
    </main>
  );
}
