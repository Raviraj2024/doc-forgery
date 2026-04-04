import Link from "next/link";
import { fetchAnalyses } from "@/lib/api";
import {
  formatDateTime,
  formatDuplicateStatus,
  formatMs,
  formatPercent,
  formatVerdict,
  verdictTone,
} from "@/lib/format";

export default async function SubmitterMySubmissionsPage() {
  const history = await fetchAnalyses(12).catch(() => ({
    page: 1,
    page_size: 12,
    total: 0,
    items: [],
  }));

  return (
    <div className="flex min-h-screen flex-col bg-background-light text-text-main">
      <header className="border-b border-border-color bg-white px-6 py-6 shadow-subtle lg:px-10">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.22em] text-primary">
              My Submissions
            </p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight">
              Uploaded Document History
            </h1>
            <p className="mt-2 max-w-2xl text-sm font-medium text-muted">
              Review previously submitted files, verdicts, and pipeline runtime.
            </p>
          </div>
          <Link
            className="rounded-full bg-primary px-5 py-3 text-sm font-bold text-white transition-opacity hover:opacity-90"
            href="/submitter/upload"
          >
            Open Upload Center
          </Link>
        </div>
      </header>

      <main className="flex-1 p-6 lg:p-8">
        <section className="rounded-[28px] border border-border-color bg-white p-6 shadow-subtle lg:p-8">
          <div className="flex items-center justify-between border-b border-border-color pb-5">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.18em] text-muted">
                Submission Log
              </p>
              <h2 className="mt-2 text-2xl font-bold tracking-tight">
                Recent Uploads
              </h2>
            </div>
            <span className="text-sm font-medium text-muted">
              {history.total} total submissions
            </span>
          </div>

          <div className="mt-6 space-y-3">
            {history.items.length === 0 ? (
              <div className="rounded-[24px] border border-dashed border-border-color bg-background-light px-6 py-14 text-center">
                <p className="text-xs font-bold uppercase tracking-[0.18em] text-muted">
                  Empty State
                </p>
                <h3 className="mt-3 text-2xl font-bold tracking-tight">
                  No submissions available yet
                </h3>
                <p className="mt-3 text-sm font-medium text-muted">
                  Uploads from the submitter workspace will appear here after
                  pipeline processing starts.
                </p>
              </div>
            ) : (
              history.items.map((item) => {
                const tone = verdictTone(item.verdict);

                return (
                  <div
                    className="rounded-[24px] border border-border-color bg-background-light px-5 py-4"
                    key={item.analysis_id}
                  >
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                      <div className="min-w-0">
                        <div className="flex items-center gap-3">
                          <div className={`size-3 rounded-full ${tone.dot}`} />
                          <p className="truncate text-base font-bold">
                            {item.filename}
                          </p>
                        </div>
                        <p className="mt-2 text-xs font-medium uppercase tracking-[0.18em] text-muted">
                          {item.analysis_id}
                        </p>
                      </div>

                      <div className="flex flex-wrap gap-3">
                        <span
                          className={`rounded-full px-3 py-1 text-xs font-bold ${tone.chip}`}
                        >
                          {formatVerdict(item.verdict)}
                        </span>
                        <span className="rounded-full bg-white px-3 py-1 text-xs font-bold text-text-main">
                          {formatDuplicateStatus(item.duplicate_status)}
                        </span>
                      </div>
                    </div>

                    <div className="mt-4 grid gap-3 text-sm font-medium text-muted md:grid-cols-4">
                      <div>
                        <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-muted">
                          Risk
                        </p>
                        <p className="mt-1 text-sm font-bold text-text-main">
                          {formatPercent(item.forensic_risk_score)}
                        </p>
                      </div>
                      <div>
                        <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-muted">
                          Runtime
                        </p>
                        <p className="mt-1 text-sm font-bold text-text-main">
                          {formatMs(item.processing_time_ms)}
                        </p>
                      </div>
                      <div>
                        <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-muted">
                          Pages
                        </p>
                        <p className="mt-1 text-sm font-bold text-text-main">
                          {item.page_count}
                        </p>
                      </div>
                      <div>
                        <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-muted">
                          Created
                        </p>
                        <p className="mt-1 text-sm font-bold text-text-main">
                          {formatDateTime(item.created_at)}
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
