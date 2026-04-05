import {
  AnalystOverrideHistoryItem,
  AnalysisHistoryResponse,
  AnalysisResponse,
  AuditLogEntry,
  DevOpsMonitoringSummaryResponse,
  DevOpsTelemetryEntry,
  DashboardSummaryResponse,
  GovernancePolicy,
  HealthResponse,
  ModelInfoResponse,
} from "@/lib/api-types";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

export function getServerApiBaseUrl() {
  return process.env.FASTAPI_URL || process.env.NEXT_PUBLIC_FASTAPI_URL || DEFAULT_API_BASE;
}

export function getBrowserApiBaseUrl() {
  return process.env.NEXT_PUBLIC_FASTAPI_URL || DEFAULT_API_BASE;
}

export function resolveApiUrl(path: string, baseUrl = getServerApiBaseUrl()) {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  return `${baseUrl}${path}`;
}

async function apiFetch<T>(path: string): Promise<T> {
  const response = await fetch(resolveApiUrl(path), {
    cache: "no-store",
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`API ${response.status}: ${detail}`);
  }

  return response.json() as Promise<T>;
}

export function fetchDashboardSummary() {
  return apiFetch<DashboardSummaryResponse>("/api/v1/dashboard/summary");
}

export function fetchAnalyses(pageSize = 20) {
  return apiFetch<AnalysisHistoryResponse>(`/api/v1/analyze?page=1&page_size=${pageSize}`);
}

export function fetchAnalysis(analysisId: string) {
  return apiFetch<AnalysisResponse>(`/api/v1/analyze/${analysisId}`);
}

export function fetchHealth() {
  return apiFetch<HealthResponse>("/api/v1/health");
}

export function fetchModelInfo() {
  return apiFetch<ModelInfoResponse>("/api/v1/model/info");
}

export function fetchGovernancePolicies() {
  return apiFetch<GovernancePolicy[]>("/api/v1/compliance/policies");
}

export function fetchAuditLog() {
  return apiFetch<AuditLogEntry[]>("/api/v1/compliance/audit-log");
}

export function fetchAnalystOverrides() {
  return apiFetch<AnalystOverrideHistoryItem[]>("/api/v1/analyst/overrides");
}

export function fetchDevOpsTelemetry() {
  return apiFetch<DevOpsTelemetryEntry[]>("/api/v1/devops/telemetry");
}

export function fetchDevOpsMonitoring() {
  return apiFetch<DevOpsMonitoringSummaryResponse>("/api/v1/devops/monitoring");
}
