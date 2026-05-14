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

export type CaseFinding = {
  title: string;
  detail: string;
  icon: string;
  tone: "danger" | "warning" | "clear";
  meta?: string;
};

export type CaseSummary = {
  headline: string;
  verdictReason: string;
  primaryFindings: CaseFinding[];
  supportingSignals: CaseFinding[];
  limitations: CaseFinding[];
};

const LAYER_LABELS: Record<string, string> = {
  ELA: "Compression Shifts",
  SRM: "Texture Changes",
  Noiseprint: "Pattern Changes",
  DINO_ViT: "Visual Outliers",
  OCR_Anomaly: "Text Consistency",
  pHash_Duplicate: "Similarity Check",
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
    detail: `${formatMs(layer.processing_ms)} review time`,
    tone: toneForLayer(layer.layer_name, layer.confidence_score),
  }));
}

export function buildCaseSummary(analysis: AnalysisResponse): CaseSummary {
  const primaryFindings: CaseFinding[] = [];
  const supportingSignals: CaseFinding[] = [];
  const limitations: CaseFinding[] = [];
  const regionCount = getTamperedRegionCount(analysis);

  analysis.ocr_anomalies.forEach((anomaly) => {
    const finding = {
      title: titleForOcrAnomaly(anomaly.type, anomaly.description),
      detail: anomaly.description,
      icon: iconForOcrAnomaly(anomaly.type, anomaly.description),
      tone: anomaly.type === "OCR_WARNING" ? "warning" : "danger",
      meta: anomaly.page_index ? `Page ${anomaly.page_index}` : "Document level",
    } satisfies CaseFinding;

    if (anomaly.type === "OCR_WARNING") {
      limitations.push(finding);
    } else {
      primaryFindings.push(finding);
    }
  });

  if (analysis.duplicate_check.duplicate_status !== "NO_MATCH") {
    primaryFindings.push({
      title: "Duplicate or near-duplicate document",
      detail: `${analysis.duplicate_check.duplicate_status.replaceAll("_", " ")} against ${analysis.duplicate_check.nearest_match_analysis_id ?? "a stored case"}.`,
      icon: "content_copy",
      tone: analysis.duplicate_check.duplicate_status === "EXACT_DUPLICATE" ? "danger" : "warning",
      meta:
        analysis.duplicate_check.hamming_distance === null
          ? "Similarity check"
          : `Hamming distance ${analysis.duplicate_check.hamming_distance}`,
    });
  }

  if (regionCount > 0) {
    primaryFindings.push({
      title: "Marked visual tamper regions",
      detail: `${regionCount} region${regionCount === 1 ? "" : "s"} of interest were localized on the document image.`,
      icon: "crop_free",
      tone: analysis.engine_scores.segmentation_score >= 0.6 ? "danger" : "warning",
      meta: `Segmentation ${formatPercent(analysis.engine_scores.segmentation_score)}`,
    });
  }

  analysis.rule_triggers.forEach((trigger) => {
    const finding: CaseFinding = {
      title: trigger.policy_id.replaceAll("_", " "),
      detail: `${trigger.severity} severity governance rule triggered for this case.`,
      icon: "policy_alert",
      tone:
        trigger.severity === "CRITICAL" || trigger.severity === "HIGH"
          ? "danger"
          : "warning",
      meta: "Review policy",
    };

    if (finding.tone === "danger") {
      primaryFindings.push(finding);
    } else {
      supportingSignals.push(finding);
    }
  });

  analysis.forensic_layers
    .map((layer) => ({
      layer,
      tone: toneForLayer(layer.layer_name, layer.confidence_score),
    }))
    .filter((item) => item.tone !== "clear")
    .sort((left, right) => right.layer.confidence_score - left.layer.confidence_score)
    .slice(0, 4)
    .forEach(({ layer, tone }) => {
      supportingSignals.push({
        title: LAYER_LABELS[layer.layer_name] ?? layer.layer_name.replaceAll("_", " "),
        detail: `${formatPercent(layer.confidence_score)} signal strength from this engine.`,
        icon: iconForLayer(layer.layer_name),
        tone,
        meta: formatMs(layer.processing_ms),
      });
    });

  analysis.warnings.forEach((warning) => {
    limitations.push({
      title: "System note",
      detail: warning,
      icon: "error",
      tone: "warning",
      meta: "Pipeline",
    });
  });

  if (
    analysis.verdict === "CLEAN" &&
    primaryFindings.length === 0 &&
    supportingSignals.length === 0
  ) {
    primaryFindings.push({
      title: "No decisive forgery signal",
      detail: "The completed checks did not return OCR anomalies, duplicate matches, marked tamper regions, or high-risk visual signals.",
      icon: "verified",
      tone: "clear",
      meta: "Review summary",
    });
  }

  return {
    headline: headlineForVerdict(analysis),
    verdictReason: verdictReasonForAnalysis(analysis, primaryFindings, supportingSignals),
    primaryFindings,
    supportingSignals,
    limitations,
  };
}

export function buildTimelineEvents(analysis: AnalysisResponse): TimelineEvent[] {
  const regionCount = getTamperedRegionCount(analysis);
  const events: TimelineEvent[] = [
    {
      icon: "upload_file",
      title: "Document Received",
      timestamp: analysis.created_at,
      detail: `${analysis.filename} was added for review with ${analysis.page_count} page${analysis.page_count === 1 ? "" : "s"}.`,
      tone: "primary",
    },
    {
      icon: "frame_inspect",
      title: "Document Checks Completed",
      timestamp: analysis.created_at,
      detail: `${analysis.forensic_layers.length} review checks completed in ${analysis.processing_time_ms} ms.`,
      tone: "primary",
    },
  ];

  analysis.forensic_layers.forEach((layer) => {
    events.push({
      icon: iconForLayer(layer.layer_name),
      title: `${(LAYER_LABELS[layer.layer_name] ?? layer.layer_name).replaceAll("_", " ")} returned ${formatPercent(layer.confidence_score)}`,
      timestamp: analysis.created_at,
      detail: `This check finished in ${formatMs(layer.processing_ms)} and contributed to the final review.`,
      tone: toneForTimeline(layer.confidence_score),
    });
  });

  if (analysis.duplicate_check.duplicate_status !== "NO_MATCH") {
    events.push({
      icon: "content_copy",
      title: "Similar Document Found",
      timestamp: analysis.created_at,
      detail: `${analysis.duplicate_check.duplicate_status.replaceAll("_", " ")} against ${analysis.duplicate_check.nearest_match_analysis_id ?? "a stored case"}.`,
      tone: "accent-amber",
    });
  }

  if (regionCount > 0) {
    events.push({
      icon: "crop_free",
      title: "Marked Areas Identified",
      timestamp: analysis.created_at,
      detail: `${regionCount} area${regionCount === 1 ? "" : "s"} of interest were marked on the document.`,
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
      detail: `${trigger.severity} severity review rule was triggered for this case.`,
      tone:
        trigger.severity === "CRITICAL" || trigger.severity === "HIGH"
          ? "accent-red"
          : "accent-amber",
    });
  });

  analysis.warnings.forEach((warning) => {
    events.push({
      icon: "error",
      title: "System Note",
      timestamp: analysis.created_at,
      detail: warning,
      tone: "accent-red",
    });
  });

  events.push({
    icon: "policy",
    title: `Verdict: ${analysis.verdict.replaceAll("_", " ")}`,
    timestamp: analysis.created_at,
    detail: `Current risk level is ${formatPercent(analysis.forensic_risk_score)}.`,
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

function titleForOcrAnomaly(type: string, description: string) {
  const lowered = description.toLowerCase();
  if (lowered.includes("future date")) return "Future date detected";
  if (lowered.includes("amount mismatch") || type === "AMOUNT_MISMATCH") {
    return "Amount arithmetic mismatch";
  }
  if (type === "DUPLICATE_REFERENCE") return "Duplicate reference detected";
  if (type === "SUSPICIOUS_KEYWORD") return "Suspicious document text";
  if (type === "INVALID_DATE") return "Invalid or implausible date";
  if (type === "OCR_WARNING") return "OCR extraction limitation";
  return type.replaceAll("_", " ");
}

function iconForOcrAnomaly(type: string, description: string) {
  const lowered = description.toLowerCase();
  if (lowered.includes("future date") || type === "INVALID_DATE") return "event_busy";
  if (type === "AMOUNT_MISMATCH") return "calculate";
  if (type === "DUPLICATE_REFERENCE") return "tag";
  if (type === "SUSPICIOUS_KEYWORD") return "text_snippet";
  if (type === "OCR_WARNING") return "warning";
  return "plagiarism";
}

function headlineForVerdict(analysis: AnalysisResponse) {
  if (analysis.verdict === "CONFIRMED_FORGERY") {
    return "Critical case: the system found high-risk evidence that needs immediate analyst review.";
  }
  if (analysis.verdict === "SUSPICIOUS") {
    return "Suspicious case: one or more checks found inconsistencies that need manual verification.";
  }
  return "Clean case: the completed checks did not find enough evidence to escalate.";
}

function verdictReasonForAnalysis(
  analysis: AnalysisResponse,
  primaryFindings: CaseFinding[],
  supportingSignals: CaseFinding[],
) {
  const risk = `Risk score ${formatPercent(analysis.forensic_risk_score)}`;
  const findingCount = primaryFindings.filter((finding) => finding.tone !== "clear").length;
  const supportingCount = supportingSignals.length;

  if (analysis.verdict === "CLEAN") {
    return `${risk}. The document stayed below the escalation threshold after OCR, duplicate, visual, and metadata checks.`;
  }

  if (findingCount > 0) {
    return `${risk}. The verdict was driven by ${findingCount} primary issue${findingCount === 1 ? "" : "s"} and ${supportingCount} supporting signal${supportingCount === 1 ? "" : "s"}.`;
  }

  return `${risk}. The verdict was driven by elevated engine scores and review policy thresholds.`;
}
