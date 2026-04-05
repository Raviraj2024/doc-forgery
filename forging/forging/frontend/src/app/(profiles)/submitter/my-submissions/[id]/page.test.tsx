import { render, screen } from "@testing-library/react";
import { vi } from "vitest";
import Page from "./page";

vi.mock("@/lib/api", () => ({
  fetchAnalysis: vi.fn(async () => ({
    analysis_id: "analysis-123",
    filename: "invoice.png",
    document_type: "invoice",
    submitter_id: "submitter-1",
    tenant_id: "tenant-1",
    session_ip_address: null,
    session_geolocation: "IN",
    page_count: 1,
    device: "cpu",
    verdict: "CONFIRMED_FORGERY",
    forensic_risk_score: 0.91,
    is_human_reviewed: false,
    engine_scores: {
      ela_score: 0.8,
      srm_score: 0.74,
      noiseprint_score: 0.72,
      dino_vit_score: 0.7,
      ocr_anomaly_score: 0.2,
      phash_score: 0,
      segmentation_score: 0.95,
    },
    forensic_layers: [],
    ocr_anomalies: [],
    duplicate_check: {
      md5_hash: "abc",
      phash: "def",
      duplicate_status: "NO_MATCH",
      nearest_match_analysis_id: null,
      hamming_distance: null,
    },
    extracted_metadata: [],
    device_fingerprint: null,
    pages: [
      {
        page_index: 1,
        width: 800,
        height: 600,
        artifacts: {
          original_url: "/api/v1/artifacts/analysis-123/original.png",
          mask_url: "/api/v1/artifacts/analysis-123/mask.png",
          overlay_url: "/api/v1/artifacts/analysis-123/overlay.png",
          ela_heatmap_url: "/api/v1/artifacts/analysis-123/ela.png",
          srm_heatmap_url: "/api/v1/artifacts/analysis-123/srm.png",
          noiseprint_heatmap_url: "/api/v1/artifacts/analysis-123/noise.png",
          dino_heatmap_url: "/api/v1/artifacts/analysis-123/dino.png",
          combined_heatmap_url: "/api/v1/artifacts/analysis-123/combined.png",
          contours_url: "/api/v1/artifacts/analysis-123/contours.png",
        },
        tampered_regions: [
          {
            region_id: "r1",
            page_index: 1,
            x: 10,
            y: 20,
            width: 100,
            height: 60,
            area_px: 6000,
            mean_mask_score: 0.81,
            max_mask_score: 0.97,
          },
        ],
      },
    ],
    rule_triggers: [],
    analyst_reviews: [],
    warnings: [],
    processing_time_ms: 812,
    created_at: "2026-04-05T12:00:00Z",
  })),
  resolveApiUrl: vi.fn((path: string) => `http://127.0.0.1:8000${path}`),
}));

vi.mock("next/navigation", () => ({
  notFound: vi.fn(() => {
    throw new Error("notFound should not be called in this test");
  }),
}));

describe("SubmitterSubmissionDetailPage", () => {
  it("renders the analysis detail from backend-shaped data", async () => {
    const element = await Page({
      params: Promise.resolve({ id: "analysis-123" }),
    });
    render(element);

    expect(screen.getByText("invoice.png")).toBeInTheDocument();
    expect(screen.getByText(/91%/i)).toBeInTheDocument();
    const tamperedRegionsCard = screen.getByText("Tampered Regions").parentElement;
    expect(tamperedRegionsCard).not.toBeNull();
    expect(tamperedRegionsCard).toHaveTextContent("1");
  });
});
