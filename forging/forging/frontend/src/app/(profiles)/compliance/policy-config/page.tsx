import Link from "next/link";

import { fetchGovernancePolicies } from "@/lib/api";

export default async function CompliancePolicyConfigPage() {
  const policies = await fetchGovernancePolicies().catch(() => []);
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
            {policies.length === 0 ? (
              <div className="md:col-span-2 rounded-[24px] border border-dashed border-border-color bg-background-light p-10 text-center">
                <p className="text-sm font-medium text-muted">No governance policies defined in the database.</p>
              </div>
            ) : (
              policies.map((policy) => (
                <div
                  className="rounded-[24px] border border-border-color bg-background-light p-5"
                  key={policy.policy_id}
                >
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-bold uppercase tracking-[0.18em] text-muted">
                      {policy.policy_id.replace(/_/g, " ")}
                    </p>
                    <span className={`px-2 py-1 text-[10px] font-bold uppercase rounded-full ${policy.is_active ? 'bg-accent-green text-white' : 'bg-gray-300 text-gray-700'}`}>
                      {policy.is_active ? "Active" : "Disabled"}
                    </span>
                  </div>
                  <p className="mt-4 text-3xl font-bold tracking-tight">
                    {policy.threshold_value > 0 ? policy.threshold_value : "Enabled"}
                  </p>
                  <p className="mt-3 text-sm font-medium text-muted">
                    {policy.description}
                  </p>
                </div>
              ))
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
