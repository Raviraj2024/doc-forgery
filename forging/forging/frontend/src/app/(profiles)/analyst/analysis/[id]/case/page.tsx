import Link from "next/link";
import { notFound } from "next/navigation";
import { DocumentViewer } from "@/components/analysis/DocumentViewer";
import { AnalysisTabs } from "@/components/restored/AnalysisTabs";
import { fetchAnalysis, resolveApiUrl } from "@/lib/api";
import {
  CaseFinding,
  buildCaseSummary,
  getPrimaryPage,
  getTopRegion,
} from "@/lib/case-view";
import {
  formatDocumentType,
  formatPercent,
  formatProvider,
  formatVerdict,
} from "@/lib/format";

export default async function CasePage({
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
  const topRegion = getTopRegion(analysis);
  const caseSummary = buildCaseSummary(analysis);
  const isWarning = analysis.verdict === "SUSPICIOUS";
  const isRed = analysis.verdict === "CONFIRMED_FORGERY";
  const pillColor = isRed
    ? "bg-accent-red"
    : isWarning
      ? "bg-accent-amber text-[#121212]"
      : "bg-accent-green";

  return (
    <div className="flex h-screen w-full flex-col bg-white">
      <header className="flex h-20 flex-shrink-0 items-center justify-between whitespace-nowrap border-b border-solid border-border-color bg-primary px-6 py-4 text-white">
        <div className="flex items-center gap-6">
          <Link
            href="/analyst/queue"
            className="flex items-center justify-center rounded-full p-2 transition-colors hover:bg-white/10"
          >
            <span className="material-symbols-outlined text-2xl text-white">
              arrow_back
            </span>
          </Link>
          <div>
            <h1 className="text-xl font-bold leading-tight">
              {analysis.analysis_id}
            </h1>
            <p className="text-sm font-medium text-white/80">
              {analysis.filename}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <span
            className={`rounded-full px-3 py-1 text-sm font-bold text-white ${pillColor}`}
          >
            {formatVerdict(analysis.verdict)}
          </span>
          <Link
            href={`/analyst/analysis/${analysis.analysis_id}/report`}
            className="inline-flex items-center justify-center rounded-full bg-white px-6 py-2 text-sm font-bold text-primary shadow-sm transition-colors hover:bg-gray-100"
          >
            Generate Report
          </Link>
        </div>
      </header>

      <main className="flex flex-1 overflow-hidden">
        <section className="relative flex h-full w-[60%] items-center justify-center overflow-hidden bg-white">
          {primaryPage ? (
            <DocumentViewer
              alt="Document Scan"
              imageUrl={resolveApiUrl(primaryPage.artifacts.overlay_url)}
              pageHeight={primaryPage.height}
              pageWidth={primaryPage.width}
              topRegion={topRegion}
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center p-8">
              <div className="rounded-[24px] border border-dashed border-border-color bg-surface px-8 py-20 text-center text-sm font-medium text-muted">
                No rendered page artifacts were returned by the backend.
              </div>
            </div>
          )}
        </section>

        <aside className="z-10 flex h-full w-[40%] flex-col border-l border-border-color bg-surface shadow-panel">
          <AnalysisTabs active="case" caseId={analysis.analysis_id} />

          <div className="flex-1 overflow-y-auto bg-surface p-6">
            <section className="rounded-xl border border-border-color bg-white p-5 shadow-sm">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-xs font-bold uppercase tracking-[0.18em] text-primary">
                    Case Summary
                  </p>
                  <h2 className="mt-2 text-2xl font-bold leading-tight text-text-main">
                    {formatVerdict(analysis.verdict)}
                  </h2>
                </div>
                <span
                  className={`rounded-full px-3 py-1 text-xs font-bold ${isRed ? "bg-accent-red/10 text-accent-red" : isWarning ? "bg-accent-amber/15 text-[#b45309]" : "bg-accent-green/10 text-accent-green"}`}
                >
                  {formatPercent(analysis.forensic_risk_score)}
                </span>
              </div>
              <p className="mt-4 text-sm font-semibold leading-6 text-text-main">
                {caseSummary.headline}
              </p>
              <p className="mt-2 text-sm font-medium leading-6 text-muted">
                {caseSummary.verdictReason}
              </p>
            </section>

            <section className="mt-5 grid grid-cols-2 gap-3">
              <CaseMetric
                label="Document type"
                value={formatDocumentType(analysis.document_type)}
              />
              <CaseMetric
                label="OCR findings"
                value={`${analysis.ocr_anomalies.length}`}
              />
              <CaseMetric
                label="Text provider"
                value={formatProvider(analysis.document_routing?.provider)}
              />
              <CaseMetric
                label="Pages"
                value={`${analysis.page_count}`}
              />
            </section>

            <FindingSection
              emptyText="No primary issue was returned for this document."
              findings={caseSummary.primaryFindings}
              title="Why This Verdict"
            />

            <FindingSection
              emptyText="No additional supporting signals crossed review thresholds."
              findings={caseSummary.supportingSignals}
              title="Supporting Signals"
            />

            {caseSummary.limitations.length > 0 && (
              <FindingSection
                emptyText=""
                findings={caseSummary.limitations}
                title="Pipeline Notes"
              />
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}

function CaseMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-border-color bg-white px-4 py-3 shadow-sm">
      <p className="text-xs font-bold uppercase tracking-[0.12em] text-muted">
        {label}
      </p>
      <p className="mt-1 truncate text-sm font-bold capitalize text-text-main">
        {value}
      </p>
    </div>
  );
}

function FindingSection({
  emptyText,
  findings,
  title,
}: {
  emptyText: string;
  findings: CaseFinding[];
  title: string;
}) {
  return (
    <section className="mt-6">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-lg font-bold text-text-main">{title}</h3>
        <span className="text-xs font-bold text-muted">{findings.length}</span>
      </div>
      {findings.length > 0 ? (
        <div className="flex flex-col gap-3">
          {findings.map((finding, index) => (
            <FindingCard finding={finding} key={`${finding.title}-${index}`} />
          ))}
        </div>
      ) : (
        <p className="rounded-xl border border-border-color bg-white px-4 py-5 text-sm font-medium text-muted shadow-sm">
          {emptyText}
        </p>
      )}
    </section>
  );
}

function FindingCard({ finding }: { finding: CaseFinding }) {
  const tone =
    finding.tone === "danger"
      ? {
          border: "border-accent-red/30",
          stripe: "bg-accent-red",
          icon: "text-accent-red",
          chip: "bg-accent-red/10 text-accent-red",
        }
      : finding.tone === "warning"
        ? {
            border: "border-accent-amber/40",
            stripe: "bg-accent-amber",
            icon: "text-[#b45309]",
            chip: "bg-accent-amber/15 text-[#b45309]",
          }
        : {
            border: "border-accent-green/30",
            stripe: "bg-accent-green",
            icon: "text-accent-green",
            chip: "bg-accent-green/10 text-accent-green",
          };

  return (
    <article
      className={`flex overflow-hidden rounded-xl border bg-white shadow-sm ${tone.border}`}
    >
      <div className={`w-2 flex-shrink-0 ${tone.stripe}`} />
      <div className="flex flex-1 gap-3 p-4">
        <span className={`material-symbols-outlined mt-0.5 text-2xl ${tone.icon}`}>
          {finding.icon}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-3">
            <h4 className="text-sm font-bold leading-5 text-text-main">
              {finding.title}
            </h4>
            {finding.meta ? (
              <span
                className={`shrink-0 rounded-full px-2 py-1 text-[11px] font-bold ${tone.chip}`}
              >
                {finding.meta}
              </span>
            ) : null}
          </div>
          <p className="mt-1 text-sm font-medium leading-5 text-muted">
            {finding.detail}
          </p>
        </div>
      </div>
    </article>
  );
}
