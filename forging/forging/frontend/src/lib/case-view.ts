import { AnalysisResponse, TamperedRegion } from "@/lib/api-types";
import { formatMs, formatPercent } from "@/lib/format";

export type TimelineEvent = {
  icon: string;
  title: string;
  timestamp: string;
  detail: string;
  tone: "primary" | "accent-red" | "accent-amber" | "muted";
};

type IntegrityRow = {
  label: string;
  value: string;
  detail: string;
  tone: "danger" | "warning" | "clear";
};

const LAYER_LABELS: Record<string, string> = {
  ELA: "ELA Residual",
  SRM: "SRM Residual",
  Noiseprint: "Noiseprint",
  DINO_ViT: "DINO Vision",
  OCR_Anomaly: "OCR Consistency",
  pHash_Duplicate: "pHash Similarity",
};

export function getPrimaryPage(analysis: AnalysisResponse) {
  return analysis.pages[0] ?? null;
}

export function getTopRegion(analysis: AnalysisResponse): TamperedRegion | null {
  return (
    analysis.pages
      .flatMap((page) => page.tampered_regions)
      .sort((left, right) => right.max_mask_score - left.max_mask_score)[0] ?? null
  );
}

export function getTamperedRegionCount(analysis: AnalysisResponse) {
  return analysis.pages.reduce(
    (count, page) => count + page.tampered_regions.length,
    0,
  );
}

export function buildIntegrityRows(analysis: AnalysisResponse): IntegrityRow[] {
  return analysis.forensic_layers.map((layer) => ({
    label: LAYER_LABELS[layer.layer_name] ?? layer.layer_name.replaceAll("_", " "),
    value: formatPercent(layer.confidence_score),
    detail: `${formatMs(layer.processing_ms)} compute time`,
    tone: toneForLayer(layer.layer_name, layer.confidence_score),
  }));
}

export function buildTimelineEvents(analysis: AnalysisResponse): TimelineEvent[] {
  const regionCount = getTamperedRegionCount(analysis);
  const events: TimelineEvent[] = [
    {
      icon: "upload_file",
      title: "Document Ingested",
      timestamp: analysis.created_at,
      detail: `${analysis.filename} entered the analysis pipeline with ${analysis.page_count} page${analysis.page_count === 1 ? "" : "s"}.`,
      tone: "primary",
    },
    {
      icon: "frame_inspect",
      title: "Layer Stack Completed",
      timestamp: analysis.created_at,
      detail: `${analysis.forensic_layers.length} forensic layers executed in ${analysis.processing_time_ms} ms.`,
      tone: "primary",
    },
  ];

  analysis.forensic_layers.forEach((layer) => {
    events.push({
      icon: iconForLayer(layer.layer_name),
      title: `${(LAYER_LABELS[layer.layer_name] ?? layer.layer_name).replaceAll("_", " ")} scored ${formatPercent(layer.confidence_score)}`,
      timestamp: analysis.created_at,
      detail: `Layer completed in ${formatMs(layer.processing_ms)} and contributed to the final risk score.`,
      tone: toneForTimeline(layer.confidence_score),
    });
  });

  if (analysis.duplicate_check.duplicate_status !== "NO_MATCH") {
    events.push({
      icon: "content_copy",
      title: "Duplicate Match Raised",
      timestamp: analysis.created_at,
      detail: `${analysis.duplicate_check.duplicate_status.replaceAll("_", " ")} against ${analysis.duplicate_check.nearest_match_analysis_id ?? "a stored case"}.`,
      tone: "accent-amber",
    });
  }

  if (regionCount > 0) {
    events.push({
      icon: "crop_free",
      title: "Tampered Regions Localized",
      timestamp: analysis.created_at,
      detail: `${regionCount} segmented region${regionCount === 1 ? "" : "s"} were extracted from the document.`,
      tone: analysis.engine_scores.segmentation_score >= 0.6 ? "accent-red" : "accent-amber",
    });
  }

  analysis.ocr_anomalies.forEach((anomaly) => {
    events.push({
      icon: "warning",
      title: anomaly.type.replaceAll("_", " "),
      timestamp: analysis.created_at,
      detail: anomaly.description,
      tone: "accent-amber",
    });
  });

  analysis.rule_triggers.forEach((trigger) => {
    events.push({
      icon: "policy_alert",
      title: trigger.policy_id.replaceAll("_", " "),
      timestamp: trigger.triggered_at,
      detail: `${trigger.severity} severity governance rule fired for this analysis.`,
      tone:
        trigger.severity === "CRITICAL" || trigger.severity === "HIGH"
          ? "accent-red"
          : "accent-amber",
    });
  });

  analysis.warnings.forEach((warning) => {
    events.push({
      icon: "error",
      title: "Pipeline Warning",
      timestamp: analysis.created_at,
      detail: warning,
      tone: "accent-red",
    });
  });

  events.push({
    icon: "policy",
    title: `Verdict: ${analysis.verdict.replaceAll("_", " ")}`,
    timestamp: analysis.created_at,
    detail: `Overall risk score closed at ${formatPercent(analysis.forensic_risk_score)}.`,
    tone:
      analysis.verdict === "CONFIRMED_FORGERY"
        ? "accent-red"
        : analysis.verdict === "SUSPICIOUS"
          ? "accent-amber"
          : "muted",
  });

  return events;
}

function toneForLayer(layerName: string, score: number): IntegrityRow["tone"] {
  if (layerName === "pHash_Duplicate") {
    if (score >= 0.95) return "danger";
    if (score >= 0.35) return "warning";
    return "clear";
  }

  if (score >= 0.7) return "danger";
  if (score >= 0.35) return "warning";
  return "clear";
}

function toneForTimeline(score: number): TimelineEvent["tone"] {
  if (score >= 0.7) return "accent-red";
  if (score >= 0.35) return "accent-amber";
  return "primary";
}

function iconForLayer(layerName: string) {
  if (layerName === "ELA") return "texture";
  if (layerName === "SRM") return "blur_on";
  if (layerName === "Noiseprint") return "grain";
  if (layerName === "DINO_ViT") return "psychology";
  if (layerName === "OCR_Anomaly") return "text_snippet";
  if (layerName === "pHash_Duplicate") return "content_copy";
  return "layers";
}
