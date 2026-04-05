import { render, screen } from "@testing-library/react";
import { vi } from "vitest";
import Page from "./page";

vi.mock("@/lib/api", () => ({
  fetchAnalyses: vi.fn(async () => ({
    page: 1,
    page_size: 50,
    total: 1,
    items: [
      {
        analysis_id: "analysis-123",
        filename: "invoice.png",
        document_type: "invoice",
        submitter_id: "submitter-1",
        tenant_id: "tenant-1",
        session_geolocation: "IN",
        page_count: 1,
        verdict: "SUSPICIOUS",
        forensic_risk_score: 0.67,
        duplicate_status: "NO_MATCH",
        is_human_reviewed: false,
        ocr_anomaly_count: 1,
        warning_count: 0,
        tampered_region_count: 2,
        processing_time_ms: 712,
        created_at: "2026-04-05T12:00:00Z",
      },
    ],
  })),
}));

describe("SubmitterMySubmissionsPage", () => {
  it("renders backend-backed submissions", async () => {
    const element = await Page();
    render(element);

    expect(screen.getByText("Submission Casebook")).toBeInTheDocument();
    expect(screen.getByText("invoice.png")).toBeInTheDocument();
    expect(screen.getByText("analysis-123")).toBeInTheDocument();
    expect(screen.getByText(/67% risk/i)).toBeInTheDocument();
  });
});
