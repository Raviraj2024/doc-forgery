import Link from "next/link";
import { notFound } from "next/navigation";
import { AnalysisTabs } from "@/components/restored/AnalysisTabs";
import { fetchAnalysis, resolveApiUrl } from "@/lib/api";
import {
  buildIntegrityRows,
  getPrimaryPage,
  getTopRegion,
} from "@/lib/case-view";
import {
  formatVerdict,
} from "@/lib/format";

export default async function AnalysisStatusPage({
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
  const integrityRows = buildIntegrityRows(analysis);

  const isWarning = analysis.verdict === "SUSPICIOUS";
  const isRed = analysis.verdict === "CONFIRMED_FORGERY";
  const pillColor = isRed
    ? "bg-accent-red"
    : isWarning
      ? "bg-accent-amber text-[#121212]"
      : "bg-accent-green";

  return (
    <div className="h-screen w-full flex flex-col bg-white">
      {/* Top Navigation (Shared Component Logic: Excluded for focused analysis view, using simple back header) */}
      <header className="flex items-center justify-between whitespace-nowrap border-b border-solid border-border-color bg-primary text-white px-6 py-4 flex-shrink-0 h-20">
        <div className="flex items-center gap-6">
          <Link
            href="/analyst/queue"
            className="flex items-center justify-center p-2 rounded-full hover:bg-white/10 transition-colors"
          >
            <span className="material-symbols-outlined text-white text-2xl">
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
            className={`px-3 py-1 ${pillColor} text-white text-sm font-bold rounded-full`}
          >
            {formatVerdict(analysis.verdict)}
          </span>
          <Link
            href={`/analyst/analysis/${analysis.analysis_id}/report`}
            className="bg-white text-primary px-6 py-2 rounded-full font-bold text-sm hover:bg-gray-100 transition-colors shadow-sm inline-flex items-center justify-center"
          >
            Generate Report
          </Link>
        </div>
      </header>

      {/* Main Workspace: Split Pane */}
      <main className="flex flex-1 overflow-hidden">
        {/* Left: Document Canvas (60%) */}
        <section className="w-[60%] h-full relative bg-white flex items-center justify-center overflow-hidden">
          {/* Document Image Placeholder */}
          <div className="w-full h-full p-8 flex items-center justify-center">
            {primaryPage ? (
              <div className="relative max-w-full max-h-full shadow-lg border border-border-color rounded-lg overflow-hidden">
                <img
                  alt="Document Scan"
                  className="object-contain max-h-[819px] w-auto"
                  src={resolveApiUrl(primaryPage.artifacts.overlay_url)}
                />

                {/* Target box for anomaly */}
                {topRegion ? (
                  <div
                    className="absolute border-2 border-accent-red bg-accent-red/10 rounded pointer-events-none"
                    style={{
                      left: `${(topRegion.x / primaryPage.width) * 100}%`,
                      top: `${(topRegion.y / primaryPage.height) * 100}%`,
                      width: `${(topRegion.width / primaryPage.width) * 100}%`,
                      height: `${(topRegion.height / primaryPage.height) * 100}%`,
                    }}
                  ></div>
                ) : null}
              </div>
            ) : (
              <div className="rounded-[24px] border border-dashed border-border-color bg-surface px-8 py-20 text-center text-sm font-medium text-muted">
                No rendered page artifacts were returned by the backend.
              </div>
            )}
          </div>

          {/* Floating Control Pill */}
          <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-surface p-2 rounded-full shadow-subtle border border-border-color">
            <button
              className="p-2 rounded-full hover:bg-white text-text-main transition-colors flex items-center justify-center"
              title="Zoom Out"
            >
              <span className="material-symbols-outlined text-xl">remove</span>
            </button>
            <span className="text-sm font-bold w-12 text-center select-none">
              100%
            </span>
            <button
              className="p-2 rounded-full hover:bg-white text-text-main transition-colors flex items-center justify-center"
              title="Zoom In"
            >
              <span className="material-symbols-outlined text-xl">add</span>
            </button>
            <div className="w-px h-6 bg-border-color mx-1"></div>
            <button
              className="p-2 rounded-full hover:bg-white text-text-main transition-colors flex items-center justify-center"
              title="Pan Tool"
            >
              <span className="material-symbols-outlined text-xl">
                pan_tool
              </span>
            </button>
            <button
              className="p-2 rounded-full hover:bg-white text-text-main transition-colors flex items-center justify-center"
              title="Reset View"
            >
              <span className="material-symbols-outlined text-xl">
                fit_screen
              </span>
            </button>
          </div>
        </section>

        {/* Right: Contextual Analysis Panel (40%) */}
        <aside className="w-[40%] h-full bg-surface border-l border-border-color shadow-panel flex flex-col z-10">
          <AnalysisTabs active="status" caseId={analysis.analysis_id} />

          {/* Panel Content Area */}
          <div className="flex-1 overflow-y-auto p-6 bg-surface">
            <div className="mb-6 flex items-center justify-between">
              <h2 className="text-xl font-bold text-text-main">
                Integrity Checks
              </h2>
              <span className="text-sm font-medium text-muted">
                {integrityRows.length} Checks Completed
              </span>
            </div>

            {/* Findings Status List (TfL Line Status Style) */}
            <div className="flex flex-col gap-3">
              {integrityRows.map((row) => {
                const isDanger = row.tone === "danger";
                const isWarning = row.tone === "warning";
                const leftBorderColor = isDanger
                  ? "bg-accent-red"
                  : isWarning
                    ? "bg-accent-amber"
                    : "bg-accent-green";
                const subTextColor = isDanger
                  ? "text-accent-red"
                  : isWarning
                    ? "text-accent-amber"
                    : "text-muted";

                const statusMessage = isDanger
                  ? "Immediate analyst attention"
                  : isWarning
                    ? "Needs manual verification"
                    : row.value || "Clear";

                return (
                  <button
                    key={row.label}
                    className="w-full bg-white border border-border-color rounded-xl h-16 flex items-center overflow-hidden hover:bg-gray-50 transition-colors shadow-sm group"
                  >
                    <div
                      className={`w-2 h-full ${leftBorderColor} flex-shrink-0`}
                    ></div>
                    <div className="flex-1 px-4 flex items-center justify-between">
                      <div className="flex flex-col items-start">
                        <span className="font-bold text-text-main text-base">
                          {row.label}
                        </span>
                        <span className={`text-xs font-medium ${subTextColor}`}>
                          {statusMessage}
                        </span>
                      </div>
                      <span className="material-symbols-outlined text-muted group-hover:text-primary transition-colors">
                        chevron_right
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>

            {analysis.ocr_anomalies.length > 0 && (
              <>
                <div className="mt-8 mb-6 flex items-center justify-between">
                  <h2 className="text-xl font-bold text-text-main">
                    OCR Anomalies
                  </h2>
                  <span className="text-sm font-medium text-muted">
                    {analysis.ocr_anomalies.length} Detected
                  </span>
                </div>
                <div className="flex flex-col gap-3">
                  {analysis.ocr_anomalies.map((anomaly, index) => (
                    <button
                      key={`${anomaly.type}-${index}`}
                      className="w-full bg-white border border-border-color rounded-xl h-16 flex items-center overflow-hidden hover:bg-gray-50 transition-colors shadow-sm group"
                    >
                      <div className="w-2 h-full bg-accent-amber flex-shrink-0"></div>
                      <div className="flex-1 px-4 flex items-center justify-between">
                        <div className="flex flex-col items-start">
                          <span className="font-bold text-text-main text-base truncate max-w-[280px]">
                            {anomaly.type.replaceAll("_", " ")}
                          </span>
                          <span className="text-xs font-medium text-accent-amber truncate max-w-[280px]">
                            {anomaly.description}
                          </span>
                        </div>
                        <span className="material-symbols-outlined text-muted group-hover:text-primary transition-colors">
                          chevron_right
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}
