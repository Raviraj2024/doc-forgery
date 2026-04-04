import Link from "next/link";
import { fetchDashboardSummary } from "@/lib/api";
import { formatMs, formatPercent } from "@/lib/format";

export default async function ComplianceReportsPage() {
  const summary = await fetchDashboardSummary().catch(() => null);

  const reportCards = [
    ["Analyses Reviewed", `${summary?.total_analyses ?? 0}`],
    ["Detection Rate", formatPercent((summary?.average_risk_score ?? 0))],
    ["Average Runtime", formatMs(summary?.average_processing_time_ms ?? 0)],
    ["OCR Alerts", `${summary?.total_ocr_anomalies ?? 0}`],
  ];

  return (
    <div className="flex min-h-screen flex-col bg-background-light text-text-main">
      <header className="border-b border-border-color bg-white px-6 py-6 shadow-subtle lg:px-10">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.22em] text-primary">
              Reports
            </p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight">
              Compliance Reporting
            </h1>
            <p className="mt-2 max-w-2xl text-sm font-medium text-muted">
              Operational summaries for review volume, runtime, and anomaly monitoring.
            </p>
          </div>
          <Link
            className="rounded-full bg-primary px-5 py-3 text-sm font-bold text-white transition-opacity hover:opacity-90"
            href="/compliance/policy-config"
          >
            Open Policy Config
          </Link>
        </div>
      </header>

      <main className="flex-1 p-6 lg:p-8">
        <section className="rounded-[28px] border border-border-color bg-white p-6 shadow-subtle lg:p-8">
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {reportCards.map(([label, value]) => (
              <div
                className="rounded-[24px] border border-border-color bg-background-light p-5"
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
          </div>
        </section>
      </main>
    </div>
  );
}
