/** TypeScript mirrors of the FastAPI Pydantic schemas (app/schemas.py). Keep in sync. */

export interface ParsedRequirements {
  keywords: string[];
  domain: string[];
  location: string[];
  date_start: string | null;
  date_end: string | null;
  task: "classification" | "regression" | null;
  target_hints: string[];
  required_features: string[];
  preferred_format: string | null;
  min_rows: number | null;
  max_rows: number | null;
  parser: string;
}

export interface SourceOut {
  id: string;
  slug: string;
  name: string;
  base_url: string;
  source_type: string;
  access_method: string;
  supports_search: boolean;
  supports_metadata: boolean;
  supports_preview: boolean;
  supports_download: boolean;
  requires_auth: boolean;
  status: string;
}

export interface AuditOut {
  id: string;
  source_id: string;
  reachable: boolean;
  api_available: boolean;
  search_available: boolean;
  metadata_available: boolean;
  preview_available: boolean;
  download_available: boolean;
  robots_status: string;
  robots_allowed: boolean | null;
  terms_status: string;
  license_status: string;
  authentication_required: boolean;
  recommended_access_method: string;
  technical_status: string;
  policy_status: string;
  notes: string[];
  audited_at: string | null;
}

export interface SourceWithAuditOut extends SourceOut {
  audit: AuditOut | null;
}

export interface DiscoveryRequestOut {
  id: string;
  query: string;
  parsed_requirements: ParsedRequirements;
  status: string;
  error_code: string | null;
  error_message: string | null;
  created_at: string;
}

export interface ApproveSourcesOut {
  request_id: string;
  status: string;
  approved_source_ids: string[];
  rejected_source_ids: string[];
  rejected_reasons: Record<string, string>;
}

export interface ScoreComponents {
  keyword_match: number;
  date_match: number;
  domain_match: number;
  completeness: number;
  size_score: number;
  feature_quality: number;
}

export interface RankedCandidateOut {
  id: string;
  source: string;
  source_dataset_id: string;
  name: string;
  description: string | null;
  url: string;
  license: string | null;
  owner: string | null;
  tags: string[];
  date_start: string | null;
  date_end: string | null;
  row_count: number | null;
  column_count: number | null;
  file_format: string | null;
  file_size_bytes: number | null;
  download_available: boolean;
  preview_available: boolean;
  overall_score: number;
  score_components: ScoreComponents;
}

export interface CandidateDetail {
  id: string;
  source: string;
  source_slug: string | null;
  source_dataset_id: string;
  name: string;
  description: string | null;
  url: string;
  license: string | null;
  owner: string | null;
  tags: string[];
  date_start: string | null;
  date_end: string | null;
  row_count: number | null;
  column_count: number | null;
  file_format: string | null;
  file_size_bytes: number | null;
  columns: CandidateColumn[];
  download_available: boolean;
  preview_available: boolean;
  overall_score: number | null;
  score_components: ScoreComponents | null;
  raw_metadata_keys: string[];
  retrieved_at: string;
}

export interface CandidateColumn {
  name: string;
  dtype: string;
  semantic_type: string;
  nullable: boolean;
  sample_values: unknown[];
}

export interface PreviewOut {
  candidate_id: string;
  columns: string[];
  dtypes: Record<string, string>;
  rows: Record<string, unknown>[];
  row_count_total: number | null;
}

export interface DatasetOut {
  id: string;
  name: string;
  description: string | null;
  source: string | null;
  source_dataset_id: string | null;
  source_url: string | null;
  license: string | null;
  row_count: number | null;
  column_count: number | null;
  file_format: string | null;
  selected_target: string | null;
  selected_task: string | null;
  created_at: string;
}

export interface DatasetFileInfo {
  id: string;
  file_name: string;
  file_format: string | null;
  file_size_bytes: number | null;
  file_hash_sha256: string | null;
  source_file_url: string | null;
  validated: boolean;
  validation_report: Record<string, unknown> | null;
}

export interface JobOut {
  id: string;
  kind: string;
  status: string;
  stage: string | null;
  progress: number;
  detail: string | null;
  error_code: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface ColumnStats {
  name: string;
  dtype: string;
  semantic_type: string;
  missing_count: number;
  missing_ratio: number;
  unique_count: number;
  unique_ratio: number;
  min?: number | null;
  max?: number | null;
  mean?: number | null;
  median?: number | null;
  std?: number | null;
  histogram?: [number, number][] | null;
  top_values?: [string, number][] | null;
  is_constant: boolean;
  is_potential_id: boolean;
  outlier_count?: number | null;
}

export interface TargetCandidate {
  column: string;
  task: string;
  confidence: number;
  rationale: string;
}

export interface EDAOut {
  dataset_id: string;
  shape: [number, number];
  column_stats: ColumnStats[];
  duplicate_rows: number;
  correlation_matrix: Record<string, Record<string, number>> | null;
  class_balance: { counts: Record<string, number>; ratios: Record<string, number>; n_classes: number; imbalanced: boolean } | null;
  quality_warnings: string[];
  target_candidates: TargetCandidate[];
  suggested_task: string | null;
  suggested_target: string | null;
}

export interface DashboardStatsOut {
  total_datasets: number;
  total_discovery_requests: number;
  total_sources: number;
  active_jobs: number;
  recent_datasets: DatasetOut[];
  recent_requests: DiscoveryRequestOut[];
  active_job_list: JobOut[];
}

export interface SourceSummary {
  id: string;
  slug: string;
  name: string;
  source_type: string;
  adapter: string | null;
  status: string;
  last_audited_at: string | null;
}
