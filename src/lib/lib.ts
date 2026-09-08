// src/lib/lib.ts
// Combines types + API client into one file. Mirrors backend_app.py's
// Pydantic schemas and endpoints exactly -- keep in sync if the API changes.

// ---------- Types ----------
export interface DatasetOut {
  id: string;
  name: string;
  description: string;
  source: string;
  url: string;
  domain: string;
  rows: number;
  columns: number;
  date_start: string;
  date_end: string;
  file_format: string;
  missing_pct: number;
  suitability_score: number;
}

export interface SearchResponse {
  query: string;
  results: DatasetOut[];
}

export interface DatasetPreview {
  id: string;
  name: string;
  columns: string[];
  dtypes: Record<string, string>;
  sample_rows: Record<string, string | number>[];
  shape: { rows: number; columns: number };
  missing_summary: Record<string, number>;
}

export interface JobStatusOut {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  task?: string | null;
  best_model?: string | null;
  error?: string | null;
}

export interface LeaderboardEntry {
  model_name: string;
  metrics: Record<string, number>;
  training_time_sec: number;
}

export interface ResultsOut {
  job_id: string;
  status: string;
  task: string;
  target_column: string;
  leaderboard: LeaderboardEntry[];
  best_model: string;
  feature_importance: Record<string, number>;
  eda_summary: Record<string, unknown>;
}

export interface PredictResponse {
  job_id: string;
  predictions: (string | number)[];
}

// ---------- API client ----------
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json();
}

export function searchDatasets(query: string, date_start?: string, date_end?: string) {
  return request<SearchResponse>("/api/search", {
    method: "POST",
    body: JSON.stringify({ query, date_start, date_end }),
  });
}

export function getDataset(id: string) {
  return request<DatasetOut>(`/api/datasets/${id}`);
}

export function previewDataset(id: string) {
  return request<DatasetPreview>(`/api/datasets/${id}/preview`);
}

export function selectDataset(dataset_id: string) {
  return request<{ dataset_id: string; status: string; next_step: string }>(
    "/api/datasets/select",
    { method: "POST", body: JSON.stringify({ dataset_id }) }
  );
}

export function runAutoML(dataset_id: string, target_column?: string) {
  return request<JobStatusOut>("/api/automl/run", {
    method: "POST",
    body: JSON.stringify({ dataset_id, target_column }),
  });
}

export function getJobStatus(jobId: string) {
  return request<JobStatusOut>(`/api/jobs/${jobId}`);
}

export function getResults(jobId: string) {
  return request<ResultsOut>(`/api/results/${jobId}`);
}

export function predict(job_id: string, records: Record<string, string | number>[]) {
  return request<PredictResponse>("/api/predict", {
    method: "POST",
    body: JSON.stringify({ job_id, records }),
  });
}
