/** Typed API client for the FastAPI backend. All external calls are server-side;
 * the browser only ever talks to this backend. */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export class ApiError extends Error {
  code: string;
  status: number;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    throw new ApiError(0, "BACKEND_UNREACHABLE", "Cannot reach the backend service. Is it running?");
  }
  if (!res.ok) {
    let code = "API_ERROR";
    let message = `Request failed with status ${res.status}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") {
        message = body.detail;
      } else if (body.detail && typeof body.detail === "object") {
        code = body.detail.code ?? code;
        message = body.detail.message ?? message;
      }
    } catch {
      /* keep defaults */
    }
    throw new ApiError(res.status, code, message);
  }
  return res.json() as Promise<T>;
}

// ---------- discovery ----------
export function createDiscoveryRequest(query: string) {
  return request<import("./api-types").DiscoveryRequestOut>("/api/discovery/requests", {
    method: "POST",
    body: JSON.stringify({ query }),
  });
}

export function getDiscoveryRequest(id: string) {
  return request<import("./api-types").DiscoveryRequestOut>(`/api/discovery/requests/${id}`);
}

export function auditRequestSources(id: string) {
  return request<import("./api-types").DiscoveryRequestOut>(`/api/discovery/requests/${id}/audit`, {
    method: "POST",
  });
}

export function getRequestSources(id: string) {
  return request<import("./api-types").SourceWithAuditOut[]>(`/api/discovery/requests/${id}/sources`);
}

export function approveSources(id: string, sourceIds: string[]) {
  return request<import("./api-types").ApproveSourcesOut>(`/api/discovery/requests/${id}/approve-sources`, {
    method: "POST",
    body: JSON.stringify({ source_ids: sourceIds }),
  });
}

export function startSearch(id: string) {
  return request<{ request_id: string; job_id: string; status: string }>(
    `/api/discovery/requests/${id}/search`,
    { method: "POST" }
  );
}

export function getSearchResults(id: string) {
  return request<import("./api-types").RankedCandidateOut[]>(`/api/discovery/requests/${id}/results`);
}

// ---------- datasets ----------
export function getCandidate(id: string) {
  return request<import("./api-types").CandidateDetail>(`/api/datasets/candidates/${id}`);
}

export function getCandidatePreview(id: string) {
  return request<import("./api-types").PreviewOut>(`/api/datasets/candidates/${id}/preview`);
}

export function selectDataset(candidateId: string) {
  return request<{ dataset_id: string; download_job_id: string; eda_job_id: string; status: string }>(
    "/api/datasets/select",
    { method: "POST", body: JSON.stringify({ candidate_id: candidateId }) }
  );
}

export function getDataset(id: string) {
  return request<import("./api-types").DatasetOut>(`/api/datasets/${id}`);
}

export function deleteDataset(id: string) {
  return request<{ dataset_id: string; status: string; artifacts_removed: number }>(
    `/api/datasets/${id}`,
    { method: "DELETE" }
  );
}

export function listDatasets() {
  return request<import("./api-types").DatasetOut[]>("/api/datasets");
}

export function getDatasetFiles(id: string) {
  return request<import("./api-types").DatasetFileInfo[]>(`/api/datasets/${id}/files`);
}

export function getDatasetJobs(id: string) {
  return request<import("./api-types").JobOut[]>(`/api/datasets/${id}/jobs`);
}

export function getEda(id: string) {
  return request<import("./api-types").EDAOut>(`/api/datasets/${id}/eda`);
}

export function confirmTarget(id: string, target: string, task: "classification" | "regression") {
  return request<{ dataset_id: string; selected_target: string; selected_task: string }>(
    `/api/datasets/${id}/confirm-target`,
    { method: "POST", body: JSON.stringify({ target, task }) }
  );
}

export function getPreprocessReport(id: string) {
  return request<import("./api-types").PreprocessReportOut>(`/api/datasets/${id}/preprocess`);
}

export function startPreprocessing(id: string) {
  return request<{ dataset_id: string; job_id?: string; status: string }>(
    `/api/datasets/${id}/preprocess`,
    { method: "POST" }
  );
}

// ---------- jobs / dashboard ----------
export function getJob(id: string) {
  return request<import("./api-types").JobOut>(`/api/jobs/${id}`);
}

export function getDashboardStats() {
  return request<import("./api-types").DashboardStatsOut>("/api/dashboard/stats");
}

export function listSources() {
  return request<import("./api-types").SourceSummary[]>("/api/sources");
}

export interface ConnectionOut {
  source_id: string;
  source_name: string;
  source_slug: string;
  status: string;
  validated: boolean;
  secret_hint: string;
  created_at: string;
}

export function listConnections() {
  return request<ConnectionOut[]>("/api/connections");
}

export function saveConnection(sourceId: string, secret: string) {
  return request<ConnectionOut>("/api/connections", {
    method: "POST",
    body: JSON.stringify({ source_id: sourceId, secret }),
  });
}

export function deleteConnection(sourceId: string) {
  return request<{ source_id: string; status: string }>(`/api/connections/${sourceId}`, {
    method: "DELETE",
  });
}
