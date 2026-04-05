"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo, useRef, useState } from "react";
import { AnalysisHistoryItem, AnalysisResponse } from "@/lib/api-types";
import {
  formatDateTime,
  formatMs,
  formatPercent,
  formatVerdict,
  verdictTone,
} from "@/lib/format";
import { getBrowserApiBaseUrl } from "@/lib/api";

type UploadConsoleProps = {
  initialRecentUploads: AnalysisHistoryItem[];
};

export function UploadConsole({ initialRecentUploads }: UploadConsoleProps) {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [documentType, setDocumentType] = useState("invoice");
  const [submitterId, setSubmitterId] = useState("manual-review");
  const [tenantId, setTenantId] = useState("default-tenant");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const acceptedFormats = useMemo(
    () => ".pdf,.png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp",
    [],
  );

  async function handleSubmit() {
    if (!selectedFile) {
      setError("Choose a document before submitting.");
      return;
    }

    setIsSubmitting(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("document_type", documentType);
    formData.append("submitter_id", submitterId);
    formData.append("tenant_id", tenantId);

    try {
      const response = await fetch(`${getBrowserApiBaseUrl()}/api/v1/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || "Upload failed.");
      }

      const analysis = (await response.json()) as AnalysisResponse;
      router.push(`/submitter/my-submissions/${analysis.analysis_id}`);
      router.refresh();
    } catch (uploadError) {
      setError(
        uploadError instanceof Error ? uploadError.message : "Upload failed.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="flex min-h-screen flex-col bg-background-light text-text-main">
      <header className="border-b border-border-color bg-white px-6 py-6 shadow-subtle lg:px-10">
        <div className="flex flex-col gap-2 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.22em] text-primary">
              Submitter Console
            </p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight">
              Document Triage Intake
            </h1>
            <p className="mt-2 max-w-2xl text-sm font-medium text-muted">
              Submit a document into the multi-layer ELA, SRM, Noiseprint, DINO,
              OCR, pHash, and segmentation pipeline.
            </p>
          </div>
          <div className="rounded-2xl border border-border-color bg-surface px-5 py-4">
            <p className="text-xs font-bold uppercase tracking-[0.2em] text-muted">
              Flow
            </p>
            <p className="mt-1 text-sm font-bold text-text-main">
              Direct Neon-backed case creation
            </p>
          </div>
        </div>
      </header>

      <main className="grid flex-1 gap-8 p-6 lg:grid-cols-[minmax(0,1.35fr)_420px] lg:p-8">
        <section className="rounded-[28px] border border-border-color bg-white p-6 shadow-subtle lg:p-8">
          <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
            <button
              className="group relative flex min-h-[420px] flex-col items-center justify-center overflow-hidden rounded-[28px] border-2 border-dashed border-border-color bg-surface px-8 text-center transition-colors hover:border-primary/40"
              onClick={() => fileInputRef.current?.click()}
              type="button"
            >
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(33,9,170,0.08),transparent_35%),radial-gradient(circle_at_bottom_left,rgba(0,130,59,0.08),transparent_35%)]" />
              <div className="relative z-10 flex max-w-lg flex-col items-center">
                <div className="mb-6 flex size-24 items-center justify-center rounded-full bg-white text-primary shadow-subtle transition-transform group-hover:scale-105">
                  <span className="material-symbols-outlined text-5xl">
                    upload_file
                  </span>
                </div>
                <h2 className="text-2xl font-bold tracking-tight">
                  Drag in a suspicious document or select from disk
                </h2>
                <p className="mt-3 text-sm font-medium text-muted">
                  Upload files for immediate multi-model verification and case
                  creation.
                </p>
                <div className="mt-8 flex items-center gap-3 rounded-full bg-primary px-6 py-3 text-sm font-bold text-white">
                  <span>Select File</span>
                  <span className="material-symbols-outlined text-lg">
                    arrow_forward
                  </span>
                </div>
                <p className="mt-4 text-xs font-medium uppercase tracking-[0.18em] text-muted">
                  PDF, JPEG, PNG, TIFF, BMP, WEBP
                </p>
              </div>
            </button>

            <div className="rounded-[28px] border border-border-color bg-background-light p-5">
              <h3 className="text-lg font-bold">Submission Metadata</h3>
              <div className="mt-5 space-y-4">
                <div>
                  <label className="mb-2 block text-xs font-bold uppercase tracking-[0.18em] text-muted">
                    Selected File
                  </label>
                  <div className="rounded-2xl border border-border-color bg-white px-4 py-3 text-sm font-medium">
                    {selectedFile ? selectedFile.name : "No file chosen"}
                  </div>
                </div>

                <div>
                  <label className="mb-2 block text-xs font-bold uppercase tracking-[0.18em] text-muted">
                    Document Type
                  </label>
                  <input
                    className="w-full rounded-2xl border border-border-color bg-white px-4 py-3 text-sm font-medium outline-none transition-colors focus:border-primary"
                    onChange={(event) => setDocumentType(event.target.value)}
                    value={documentType}
                  />
                </div>

                <div>
                  <label className="mb-2 block text-xs font-bold uppercase tracking-[0.18em] text-muted">
                    Submitter Id
                  </label>
                  <input
                    className="w-full rounded-2xl border border-border-color bg-white px-4 py-3 text-sm font-medium outline-none transition-colors focus:border-primary"
                    onChange={(event) => setSubmitterId(event.target.value)}
                    value={submitterId}
                  />
                </div>

                <div>
                  <label className="mb-2 block text-xs font-bold uppercase tracking-[0.18em] text-muted">
                    Tenant Id
                  </label>
                  <input
                    className="w-full rounded-2xl border border-border-color bg-white px-4 py-3 text-sm font-medium outline-none transition-colors focus:border-primary"
                    onChange={(event) => setTenantId(event.target.value)}
                    value={tenantId}
                  />
                </div>

                {error ? (
                  <div className="rounded-2xl border border-accent-red/20 bg-accent-red/10 px-4 py-3 text-sm font-medium text-accent-red">
                    {error}
                  </div>
                ) : null}

                <button
                  className="flex w-full items-center justify-center gap-2 rounded-full bg-primary px-6 py-3 text-sm font-bold text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
                  disabled={isSubmitting}
                  onClick={handleSubmit}
                  type="button"
                >
                  <span className="material-symbols-outlined text-lg">
                    {isSubmitting ? "hourglass_top" : "play_arrow"}
                  </span>
                  <span>
                    {isSubmitting
                      ? "Running Forensic Layers"
                      : "Submit to Pipeline"}
                  </span>
                </button>
              </div>
            </div>
          </div>

          <input
            accept={acceptedFormats}
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0] ?? null;
              setSelectedFile(file);
            }}
            ref={fileInputRef}
            type="file"
          />
        </section>

        <aside className="rounded-[28px] border border-border-color bg-white p-6 shadow-subtle">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.18em] text-muted">
                Recent Analyses
              </p>
              <h2 className="mt-2 text-2xl font-bold tracking-tight">
                Submission History
              </h2>
            </div>
            <Link
              className="text-sm font-bold text-primary hover:underline"
              href="/submitter/my-submissions"
            >
              My Submissions
            </Link>
          </div>

          <div className="mt-6 space-y-3">
            {initialRecentUploads.length === 0 ? (
              <div className="rounded-3xl border border-dashed border-border-color bg-surface px-5 py-8 text-center text-sm font-medium text-muted">
                No analyses stored yet. The first completed upload will appear
                here.
              </div>
            ) : (
              initialRecentUploads.map((item) => {
                const tone = verdictTone(item.verdict);

                return (
                  <div
                    className="rounded-[24px] border border-border-color bg-background-light px-5 py-4"
                    key={item.analysis_id}
                  >
                    <div className="flex items-start gap-4">
                      <div className={`mt-1 size-3 rounded-full ${tone.dot}`} />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-start justify-between gap-4">
                          <div className="min-w-0">
                            <p className="truncate text-base font-bold">
                              {item.filename}
                            </p>
                            <p className="mt-1 text-xs font-medium uppercase tracking-[0.18em] text-muted">
                              {item.analysis_id}
                            </p>
                          </div>
                          <span
                            className={`shrink-0 rounded-full px-3 py-1 text-xs font-bold ${tone.chip}`}
                          >
                            {formatVerdict(item.verdict)}
                          </span>
                        </div>
                        <div className="mt-4 flex flex-wrap items-center gap-3 text-xs font-medium text-muted">
                          <span>{formatPercent(item.forensic_risk_score)} risk</span>
                          <span>{item.tampered_region_count} regions</span>
                          <span>{item.ocr_anomaly_count} OCR flags</span>
                          <span>{formatMs(item.processing_time_ms)}</span>
                          <span>{formatDateTime(item.created_at)}</span>
                        </div>
                        <Link
                          className="mt-4 inline-flex items-center gap-2 text-sm font-bold text-primary hover:underline"
                          href={`/submitter/my-submissions/${item.analysis_id}`}
                        >
                          <span>Open analysis</span>
                          <span className="material-symbols-outlined text-base">
                            arrow_forward
                          </span>
                        </Link>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}
