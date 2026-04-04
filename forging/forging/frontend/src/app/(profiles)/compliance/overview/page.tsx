import Link from "next/link";
import { fetchDashboardSummary } from "@/lib/api";
import { formatMs, formatPercent } from "@/lib/format";

export default async function ComplianceOverviewPage() {
  const summary = await fetchDashboardSummary().catch(() => null);
  const total = summary?.total_analyses ?? 0;
  const flagged =
    (summary?.suspicious_count ?? 0) + (summary?.confirmed_forgery_count ?? 0);
  const detectionRate = total > 0 ? flagged / total : 0;

  return (
    <div className="flex min-h-screen flex-col bg-background-light text-text-main">
      <header className="border-b border-border-color bg-white px-6 py-6 shadow-subtle lg:px-10">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.22em] text-primary">
              Compliance Overview
            </p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight">
              System risk analytics
            </h1>
            <p className="mt-2 max-w-2xl text-sm font-medium text-muted">
              Portfolio-level metrics generated from persisted FastAPI analyses
              instead of local mock dashboards.
            </p>
          </div>
          <Link
            className="rounded-full border border-border-color px-5 py-3 text-sm font-bold transition-colors hover:bg-white"
            href="/compliance/overview"
          >
            Risk Overview
          </Link>
        </div>
      </header>

      <main className="flex-1 space-y-8 p-6 lg:p-8">
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {[
            ["Total Submissions", `${total}`],
            ["Detection Rate", formatPercent(detectionRate)],
            ["Avg Runtime", formatMs(summary?.average_processing_time_ms ?? 0)],
            ["OCR Anomalies", `${summary?.total_ocr_anomalies ?? 0}`],
          ].map(([label, value]) => (
            <div
              className="rounded-[24px] border border-border-color bg-white p-5 shadow-subtle"
              key={label}
            >
              <p className="text-xs font-bold uppercase tracking-[0.18em] text-muted">
                {label}
              </p>
              <p className="mt-4 text-4xl font-bold tracking-tight">{value}</p>
            </div>
          ))}
        </div>

        <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_360px]">
          <section className="rounded-[28px] border border-border-color bg-white p-6 shadow-subtle">
            <div className="flex items-center justify-between border-b border-border-color pb-5">
              <div>
                <p className="text-xs font-bold uppercase tracking-[0.18em] text-muted">
                  Recent flags
                </p>
                <h2 className="mt-2 text-2xl font-bold tracking-tight">
                  Highest-risk documents
                </h2>
              </div>
            </div>

            <div className="mt-6 space-y-3">
              {(summary?.flagged_analyses ?? []).map((item) => (
                <div
                  className="flex items-center justify-between gap-4 rounded-[24px] border border-border-color bg-background-light px-5 py-4"
                  key={item.analysis_id}
                >
                  <div className="min-w-0">
                    <p className="truncate text-base font-bold">
                      {item.filename}
                    </p>
                    <p className="mt-1 text-xs font-medium uppercase tracking-[0.18em] text-muted">
                      {item.analysis_id}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-bold">
                      {formatPercent(item.forensic_risk_score)}
                    </p>
                    <p className="mt-1 text-xs font-medium text-muted">
                      {item.verdict.replaceAll("_", " ")}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </section>

          <aside className="space-y-4">
            {[
              ["Exact Duplicates", `${summary?.exact_duplicate_count ?? 0}`],
              ["Near Duplicates", `${summary?.near_duplicate_count ?? 0}`],
              ["Confirmed Forgery", `${summary?.confirmed_forgery_count ?? 0}`],
              ["Suspicious", `${summary?.suspicious_count ?? 0}`],
            ].map(([label, value]) => (
              <div
                className="rounded-[24px] border border-border-color bg-white p-5 shadow-subtle"
                key={label}
              >
                <p className="text-xs font-bold uppercase tracking-[0.18em] text-muted">
                  {label}
                </p>
                <p className="mt-4 text-3xl font-bold tracking-tight">
                  {value}
                </p>
              </div>
            ))}
          </aside>
        </div>
      </main>
    </div>
  );
}
