import Link from "next/link";

const policyCards = [
  {
    label: "Forgery Threshold",
    value: "83%",
    detail: "Escalate documents above this forensic risk score.",
  },
  {
    label: "Duplicate Review",
    value: "Enabled",
    detail: "Near and exact duplicate findings require retention review.",
  },
  {
    label: "OCR Escalation",
    value: "Tier 2",
    detail: "Trigger enhanced review when OCR anomaly count crosses policy band.",
  },
  {
    label: "Retention Window",
    value: "180 days",
    detail: "Preserve evidence and related audit entries for compliance requests.",
  },
];

export default function CompliancePolicyConfigPage() {
  return (
    <div className="flex min-h-screen flex-col bg-background-light text-text-main">
      <header className="border-b border-border-color bg-white px-6 py-6 shadow-subtle lg:px-10">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.22em] text-primary">
              Policy Config
            </p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight">
              Governance Controls
            </h1>
            <p className="mt-2 max-w-2xl text-sm font-medium text-muted">
              Snapshot of policy settings that drive escalations, retention, and review flow.
            </p>
          </div>
          <Link
            className="rounded-full bg-primary px-5 py-3 text-sm font-bold text-white transition-opacity hover:opacity-90"
            href="/compliance/audit-log"
          >
            View Audit Log
          </Link>
        </div>
      </header>

      <main className="flex-1 p-6 lg:p-8">
        <section className="rounded-[28px] border border-border-color bg-white p-6 shadow-subtle lg:p-8">
          <div className="grid gap-4 md:grid-cols-2">
            {policyCards.map((policy) => (
              <div
                className="rounded-[24px] border border-border-color bg-background-light p-5"
                key={policy.label}
              >
                <p className="text-xs font-bold uppercase tracking-[0.18em] text-muted">
                  {policy.label}
                </p>
                <p className="mt-4 text-3xl font-bold tracking-tight">
                  {policy.value}
                </p>
                <p className="mt-3 text-sm font-medium text-muted">
                  {policy.detail}
                </p>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
