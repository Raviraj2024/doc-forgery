import Link from "next/link";
import { fetchAnalystOverrides } from "@/lib/api";
import { formatDateTime } from "@/lib/format";

export default async function AnalystOverrideHistoryPage() {
  const overrides = await fetchAnalystOverrides().catch(() => []);
  return (
    <div className="flex min-h-screen flex-col bg-background-light text-text-main">
      <header className="border-b border-border-color bg-white px-6 py-6 shadow-subtle lg:px-10">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.22em] text-primary">
              Override History
            </p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight">
              Manual Review Overrides
            </h1>
            <p className="mt-2 max-w-2xl text-sm font-medium text-muted">
              Audit trail for analyst override decisions and escalations.
            </p>
          </div>
          <Link
            className="rounded-full bg-primary px-5 py-3 text-sm font-bold text-white transition-opacity hover:opacity-90"
            href="/analyst/queue"
          >
            Open Review Queue
          </Link>
        </div>
      </header>

      <main className="flex-1 p-6 lg:p-8">
        <section className="rounded-[28px] border border-border-color bg-white p-6 shadow-subtle lg:p-8">
          {overrides.length === 0 ? (
            <div className="rounded-[24px] border border-dashed border-border-color bg-background-light px-6 py-16 text-center">
              <p className="text-xs font-bold uppercase tracking-[0.18em] text-muted">
                Audit Log
              </p>
              <h2 className="mt-3 text-2xl font-bold tracking-tight">
                No override events recorded yet
              </h2>
              <p className="mt-3 text-sm font-medium text-muted">
                Analyst override activity will appear here once manual decisions
                are captured by the workflow.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {overrides.map((override) => (
                <div key={override.review_id} className="rounded-[24px] border border-border-color bg-background-light p-5">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-bold uppercase tracking-[0.18em] text-muted">Review #{override.review_id}</span>
                    <span className="text-xs font-bold text-muted">{formatDateTime(override.reviewed_at)}</span>
                  </div>
                    <div className="flex flex-col gap-2">
                      <p className="text-lg font-bold">{override.filename}</p>
                      <div className="flex items-center gap-3 mt-2">
                        <span className="text-sm font-bold line-through text-accent-red">{override.previous_verdict.replace(/_/g, " ")}</span>
                        <span className="material-symbols-outlined text-muted text-sm">arrow_forward</span>
                        <span className="text-sm font-bold text-accent-green">{override.new_verdict.replace(/_/g, " ")}</span>
                      </div>
                      <p className="mt-3 text-sm font-medium bg-white p-3 rounded-xl border border-border-color text-muted">
                        {override.override_reason}
                      </p>
                      <p className="mt-2 text-xs font-medium text-muted">Analyst ID: {override.analyst_user_id}</p>
                    </div>
                  </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
