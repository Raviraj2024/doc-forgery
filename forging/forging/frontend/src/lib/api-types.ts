export type Verdict = "CLEAN" | "SUSPICIOUS" | "CONFIRMED_FORGERY";
export type DuplicateStatus = "NO_MATCH" | "NEAR_DUPLICATE" | "EXACT_DUPLICATE";

export interface EngineScores {
  ela_score: number;
  srm_score: number;
  noiseprint_score: number;
  dino_vit_score: number;
  ocr_anomaly_score: number;
  phash_score: number;
  segmentation_score: number;
}

export interface OCRAnomaly {
  type: string;
  description: string;
  page_index: number | null;
}

export interface DuplicateCheck {
  md5_hash: string;
  phash: string;
  duplicate_status: DuplicateStatus;
  nearest_match_analysis_id: string | null;
  hamming_distance: number | null;
}

export interface PageArtifacts {
  original_url: string;
  mask_url: string;
  overlay_url: string;
  ela_heatmap_url: string;
  srm_heatmap_url: string;
  noiseprint_heatmap_url: string;
  dino_heatmap_url: string;
  combined_heatmap_url: string;
  contours_url: string;
}

export interface TamperedRegion {
  region_id: string;
  page_index: number;
  x: number;
  y: number;
  width: number;
  height: number;
  area_px: number;
  mean_mask_score: number;
  max_mask_score: number;
}

export interface PageResult {
  page_index: number;
  width: number;
  height: number;
  artifacts: PageArtifacts;
  tampered_regions: TamperedRegion[];
}

export interface ExtractedMetadata {
  metadata_type: string;
  software_signature: string | null;
  camera_make?: string | null;
  camera_model?: string | null;
  modification_date_raw?: string | null;
  gps_data?: Record<string, unknown> | null;
  raw_dump: Record<string, unknown>;
}

export interface DeviceFingerprint {
  device_hash: string | null;
  user_agent: string | null;
  browser: string | null;
  os: string | null;
  is_known_fraud_device: boolean;
}

export interface ForensicLayer {
  layer_name: string;
  confidence_score: number;
  processing_ms: number;
}

export interface AnalystReview {
  review_id: number;
  analyst_user_id: string;
  previous_verdict: string;
  new_verdict: string;
  override_reason: string;
  reviewed_at: string;
}

export interface AnalystOverrideHistoryItem extends AnalystReview {
  analysis_id: string;
  filename: string;
}

export interface GovernancePolicy {
  policy_id: string;
  description: string;
  threshold_value: number;
  is_active: boolean;
  updated_at: string;
}

export interface RuleTrigger {
  policy_id: string;
  severity: string;
  triggered_at: string;
}

export interface AuditLogEntry {
  id: number;
  analysis_id: string;
  filename: string;
  verdict: Verdict;
  forensic_risk_score: number;
  policy_id: string;
  severity: string;
  triggered_at: string;
}

export interface DevOpsTelemetryEntry {
  layer_name: string;
  execution_count: number;
  avg_processing_ms: number;
  avg_confidence_score: number;
}

export interface MonitoringWarningEvent {
  analysis_id: string;
  filename: string;
  warning: string;
  created_at: string;
}

export interface DevOpsMonitoringSummaryResponse {
  total_analyses: number;
  analyses_with_warnings: number;
  analyses_with_segmentation_fallback: number;
  average_processing_time_ms: number;
  p50_processing_time_ms: number;
  p95_processing_time_ms: number;
  warning_rate: number;
  latest_analysis_at: string | null;
  calibration_loaded: boolean;
  calibration_generated_at: string | null;
  calibration_sample_count: number | null;
  calibration_mean_iou: number | null;
  calibration_mean_f1: number | null;
  recent_warning_events: MonitoringWarningEvent[];
}

export interface AnalysisResponse {
  analysis_id: string;
  filename: string;
  document_type: string | null;
  submitter_id: string | null;
  tenant_id: string | null;
  session_ip_address: string | null;
  session_geolocation: string | null;
  page_count: number;
  device: string;
  verdict: Verdict;
  forensic_risk_score: number;
  is_human_reviewed: boolean;
  engine_scores: EngineScores;
  forensic_layers: ForensicLayer[];
  ocr_anomalies: OCRAnomaly[];
  duplicate_check: DuplicateCheck;
  extracted_metadata: ExtractedMetadata[];
  device_fingerprint: DeviceFingerprint | null;
  pages: PageResult[];
  rule_triggers: RuleTrigger[];
  analyst_reviews: AnalystReview[];
  warnings: string[];
  processing_time_ms: number;
  created_at: string;
}

export interface AnalysisHistoryItem {
  analysis_id: string;
  filename: string;
  document_type: string | null;
  submitter_id: string | null;
  tenant_id: string | null;
  session_geolocation: string | null;
  page_count: number;
  verdict: Verdict;
  forensic_risk_score: number;
  duplicate_status: DuplicateStatus;
  is_human_reviewed: boolean;
  ocr_anomaly_count: number;
  warning_count: number;
  tampered_region_count: number;
  processing_time_ms: number;
  created_at: string;
}

export interface AnalysisHistoryResponse {
  page: number;
  page_size: number;
  total: number;
  items: AnalysisHistoryItem[];
}

export interface DashboardSummaryResponse {
  total_analyses: number;
  clean_count: number;
  suspicious_count: number;
  confirmed_forgery_count: number;
  exact_duplicate_count: number;
  near_duplicate_count: number;
  total_ocr_anomalies: number;
  average_risk_score: number;
  average_processing_time_ms: number;
  engine_averages: EngineScores;
  recent_analyses: AnalysisHistoryItem[];
  flagged_analyses: AnalysisHistoryItem[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  checkpoint_exists: boolean;
  database_ready: boolean;
}

export interface ModelInfoResponse {
  checkpoint_path: string;
  selected_encoder: string | null;
  input_channels: number | null;
  checkpoint_input_channels: number | null;
  device: string;
  model_parameter_count: number | null;
  model_loaded: boolean;
  load_error: string | null;
  tried_architectures: string[];
  checkpoint_sha256: string | null;
  checkpoint_size_bytes: number | null;
  calibration_profile_path: string | null;
  calibration_loaded: boolean;
  calibration_generated_at: string | null;
  calibration_sample_count: number | null;
}
