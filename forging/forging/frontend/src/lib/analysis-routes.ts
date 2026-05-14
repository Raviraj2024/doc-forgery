export type AnalystAnalysisView =
  | "status"
  | "document"
  | "evidence"
  | "diagnostics"
  | "timeline"
  | "case"
  | "report";

const ANALYST_ANALYSIS_VIEW_SUFFIX: Record<AnalystAnalysisView, string> = {
  status: "",
  document: "/document",
  evidence: "/evidence",
  diagnostics: "/diagnostics",
  timeline: "/timeline",
  case: "/case",
  report: "/report",
};

export function buildAnalystAnalysisPath(
  analysisId: string,
  view: AnalystAnalysisView = "status",
) {
  return `/analyst/analysis/${encodeURIComponent(analysisId)}${ANALYST_ANALYSIS_VIEW_SUFFIX[view]}`;
}
