import Link from "next/link";
import { notFound } from "next/navigation";
import { fetchAnalysis, resolveApiUrl } from "@/lib/api";
import {
  formatDateTime,
  formatPercent,
  formatVerdict,
  verdictTone,
} from "@/lib/format";
import { getPrimaryPage, getTamperedRegionCount } from "@/lib/case-view";

export default async function SubmitterSubmissionDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const analysis = await fetchAnalysis(id).catch(() => null);

  if (!analysis) {
    notFound();
  }

  const primaryPage = getPrimaryPage(analysis);
  const tamperedRegionCount = getTamperedRegionCount(analysis);
  const tone = verdictTone(analysis.verdict);

  return (
    <div className="flex min-h-screen flex-col bg-background-light text-text-main">
      <header className="border-b border-border-color bg-white px-6 py-6 shadow-subtle lg:px-10">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.22em] text-primary">
              Submission Detail
            </p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight">
              {analysis.filename}
            </h1>
            <p className="mt-2 max-w-2xl text-sm font-medium text-muted">
              Analysis ID {analysis.analysis_id}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span
              className={`rounded-full px-4 py-2 text-xs font-bold ${tone.chip}`}
            >
              {formatVerdict(analysis.verdict)}
            </span>
            <Link
              className="rounded-full border border-border-color px-5 py-3 text-sm font-bold transition-colors hover:bg-background-light"
              href="/submitter/my-submissions"
            >
              Back to My Submissions
            </Link>
          </div>
        </div>
      </header>

      <main className="grid flex-1 gap-8 p-6 lg:grid-cols-[minmax(0,1.2fr)_380px] lg:p-8">
        <section className="rounded-[28px] border border-border-color bg-white p-6 shadow-subtle">
          <div className="flex items-center justify-between border-b border-border-color pb-5">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.18em] text-muted">
                Primary Evidence
              </p>
              <h2 className="mt-2 text-2xl font-bold tracking-tight">
                Rendered Document View
              </h2>
            </div>
          </div>

          <div className="mt-6">
            {primaryPage ? (
              <img
                alt={`${analysis.filename} forensic overlay`}
                className="w-full rounded-[24px] border border-border-color bg-background-light object-contain"
                src={resolveApiUrl(primaryPage.artifacts.overlay_url)}
              />
            ) : (
              <div className="rounded-[24px] border border-dashed border-border-color bg-background-light px-6 py-16 text-center text-sm font-medium text-muted">
                No page artifact is available for this submission.
              </div>
            )}
          </div>
        </section>

        <aside className="space-y-4">
          <div className="rounded-[28px] border border-border-color bg-white p-6 shadow-subtle">
            <p className="text-xs font-bold uppercase tracking-[0.18em] text-muted">
              Summary
            </p>
            <div className="mt-5 space-y-4">
              <div>
                <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-muted">
                  Verdict
                </p>
                <p className="mt-1 text-lg font-bold">{formatVerdict(analysis.verdict)}</p>
              </div>
              <div>
                <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-muted">
                  Risk Score
                </p>
                <p className="mt-1 text-lg font-bold">
                  {formatPercent(analysis.forensic_risk_score)}
                </p>
              </div>
              <div>
                <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-muted">
                  Completed At
                </p>
                <p className="mt-1 text-sm font-bold">{formatDateTime(analysis.created_at)}</p>
              </div>
              <div>
                <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-muted">
                  Duplicate Status
                </p>
                <p className="mt-1 text-sm font-bold">
                  {analysis.duplicate_check.duplicate_status.replaceAll("_", " ")}
                </p>
              </div>
            </div>
          </div>

          <div className="rounded-[28px] border border-border-color bg-white p-6 shadow-subtle">
            <p className="text-xs font-bold uppercase tracking-[0.18em] text-muted">
              Signals
            </p>
            <div className="mt-5 grid gap-3">
              <div className="rounded-2xl border border-border-color bg-background-light px-4 py-3">
                <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-muted">
                  Pages
                </p>
                <p className="mt-1 text-lg font-bold">{analysis.page_count}</p>
              </div>
              <div className="rounded-2xl border border-border-color bg-background-light px-4 py-3">
                <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-muted">
                  Tampered Regions
                </p>
                <p className="mt-1 text-lg font-bold">{tamperedRegionCount}</p>
              </div>
              <div className="rounded-2xl border border-border-color bg-background-light px-4 py-3">
                <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-muted">
                  OCR Anomalies
                </p>
                <p className="mt-1 text-lg font-bold">{analysis.ocr_anomalies.length}</p>
              </div>
              <div className="rounded-2xl border border-border-color bg-background-light px-4 py-3">
                <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-muted">
                  Warning Count
                </p>
                <p className="mt-1 text-lg font-bold">{analysis.warnings.length}</p>
              </div>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
