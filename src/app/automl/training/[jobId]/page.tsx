"use client";
// src/app/automl/training/[jobId]/page.tsx -- Training Dashboard
import { useEffect, useRef, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import { getJobStatus, JobStatusOut } from "@/lib/lib";

const STAGES = ["Data Validation", "EDA", "Preprocessing", "Model Training", "Evaluation"];

export default function TrainingPage() {
  const router = useRouter();
  const params = useParams();
  const jobId = params.jobId as string;

  const [status, setStatus] = useState<JobStatusOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const s = await getJobStatus(jobId);
        setStatus(s);
        if (s.status === "completed") {
          clearInterval(pollRef.current!);
          setTimeout(() => router.push(`/results/${jobId}`), 800);
        } else if (s.status === "failed") {
          clearInterval(pollRef.current!);
        }
      } catch (err) {
        setError((err as Error).message);
      }
    };
    poll();
    pollRef.current = setInterval(poll, 2000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [jobId, router]);

  const stageIndex = status?.status === "queued" ? 0
    : status?.status === "running" ? 2
    : status?.status === "completed" ? STAGES.length
    : 0;
  const progressPct = Math.round((stageIndex / STAGES.length) * 100);

  return (
    <main className="container">
      <div className="steps" style={{ marginBottom: "24px" }}>
        <span className="step">1. Search</span>
        <span className="step">2. Preview</span>
        <span className="step">3. Configure</span>
        <span className="step active">4. Train</span>
        <span className="step">5. Results</span>
      </div>

      <h1>Training in Progress</h1>
      <p className="subtitle">Job ID: <code>{jobId}</code></p>

      {error && <div className="error-box">{error}</div>}

      <div className="card">
        <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
          {status?.status !== "completed" && status?.status !== "failed" && <span className="spinner" />}
          <span className={`status-pill status-${status?.status || "queued"}`}>
            {(status?.status || "queued").toUpperCase()}
          </span>
        </div>
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${progressPct}%` }} />
        </div>

        {STAGES.map((stage, i) => (
          <div key={stage} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", color: i < stageIndex ? "var(--text)" : "var(--muted)" }}>
            <span>{stage}</span>
            <span>{i < stageIndex ? "✓" : "…"}</span>
          </div>
        ))}

        {status?.status === "failed" && (
          <div className="error-box" style={{ marginTop: "16px" }}>
            Training failed: {status.error || "Unknown error"}
          </div>
        )}
      </div>
    </main>
  );
}
