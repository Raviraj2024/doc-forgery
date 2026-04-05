import Link from "next/link";
import { fetchDashboardSummary } from "@/lib/api";
import {
  formatRelativeTime,
  formatPercent,
  formatVerdict,
} from "@/lib/format";

export default async function AnalystQueuePage() {
  const summary = await fetchDashboardSummary().catch(() => null);
  const flagged = summary?.flagged_analyses ?? [];
  const criticalCount = flagged.filter(
    (f) => f.verdict === "CONFIRMED_FORGERY",
  ).length;

  return (
    <div
      className="min-h-screen flex flex-col antialiased"
      style={{
        fontFamily: "'Plus Jakarta Sans', sans-serif",
        backgroundColor: "#FFFFFF",
        color: "#121212",
      }}
    >
      {/* Header — clone of findings_status/code.html line 59-73, exact same primary bg */}
      <header
        className="flex items-center justify-between whitespace-nowrap px-10 py-4 text-white"
        style={{ backgroundColor: "#0019A8" }}
      >
        <div className="flex items-center gap-4">
          <div className="size-6 text-white">
            <span
              className="material-symbols-outlined"
              style={{ fontSize: "24px" }}
            >
              shield
            </span>
          </div>
          <h2 className="text-white text-xl font-bold leading-tight tracking-tight">
            Operational Trust
          </h2>
        </div>
        <div className="flex flex-1 justify-end gap-6 items-center">
          <div className="relative w-64">
            <span
              className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2"
              style={{ color: "#737373", fontSize: "20px" }}
            >
              search
            </span>
            <input
              className="w-full bg-white/10 border border-white/20 text-white placeholder-white/50 rounded-full py-2 pl-10 pr-4 focus:outline-none focus:ring-2 focus:ring-white/50 text-sm font-medium"
              placeholder="Search Case ID..."
              type="text"
            />
          </div>
          <div
            className="bg-center bg-no-repeat aspect-square bg-cover rounded-full size-10 border-2 border-white/20"
            style={{
              backgroundImage:
                "url('https://ui-avatars.com/api/?name=A&background=0019A8&color=fff&bold=true')",
            }}
          ></div>
        </div>
      </header>

      {/* Main Content — split screen like findings_status/code.html line 78-197 but full-width for queue */}
      <main
        className="flex-1 flex flex-col"
        style={{ backgroundColor: "#F4F6F8" }}
      >
        <div
          className="flex-1 flex flex-col h-full"
          style={{ boxShadow: "0 4px 20px rgba(0,0,0,0.04)" }}
        >
          {/* Context Header — clone of line 108-111 */}
          <div className="px-8 pt-8 pb-4">
            <p
              className="text-sm font-medium mb-1 uppercase tracking-wider"
              style={{ color: "#737373" }}
            >
              Analyst Review Queue
            </p>
            <h1
              className="text-3xl font-bold tracking-tight"
              style={{ color: "#121212" }}
            >
              Documents Pending Review
            </h1>
          </div>

          {/* Tabs — clone of findings_status/code.html line 113-126 */}
          <div
            className="flex px-8 gap-8"
            style={{ borderBottom: "1px solid #E5E5E5" }}
          >
            <a
              className="flex flex-col items-center justify-center pb-3 pt-4"
              style={{ borderBottom: "4px solid #0019A8", color: "#121212" }}
              href="#"
            >
              <p className="text-sm font-bold leading-normal">
                All ({flagged.length})
              </p>
            </a>
            <a
              className="flex flex-col items-center justify-center pb-3 pt-4 hover:text-[#121212] transition-colors"
              style={{
                borderBottom: "4px solid transparent",
                color: "#737373",
              }}
              href="#"
            >
              <p className="text-sm font-bold leading-normal">
                Critical ({criticalCount})
              </p>
            </a>
            <a
              className="flex flex-col items-center justify-center pb-3 pt-4 hover:text-[#121212] transition-colors"
              style={{
                borderBottom: "4px solid transparent",
                color: "#737373",
              }}
              href="#"
            >
              <p className="text-sm font-bold leading-normal">
                Suspicious ({flagged.length - criticalCount})
              </p>
            </a>
          </div>

          {/* Scrollable Content: Queue Items — clone of findings_status/code.html line 128-188, using the exact status row pattern */}
          <div className="flex-1 overflow-y-auto p-8 space-y-4">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold" style={{ color: "#121212" }}>
                Integrity Findings
              </h2>
              {criticalCount > 0 && (
                <span className="px-3 py-1 rounded-full text-sm font-bold bg-accent-red text-white">
                  {criticalCount} Critical
                </span>
              )}
            </div>

            {flagged.length > 0 ? (
              flagged.map((item) => {
                const isRed = item.verdict === "CONFIRMED_FORGERY";
                const stripColor = isRed
                  ? "var(--color-accent-red)"
                  : "var(--color-accent-amber)";
                const statusText = isRed
                  ? formatVerdict(item.verdict)
                  : formatVerdict(item.verdict);
                const statusColor = isRed
                  ? "var(--color-accent-red)"
                  : "var(--color-accent-amber)";

                return (
                  <Link
                    key={item.analysis_id}
                    href={`/analyst/analysis/${item.analysis_id}`}
                    className="w-full flex items-stretch h-16 bg-white rounded-lg overflow-hidden shadow-sm hover:shadow-md transition-shadow group text-left cursor-pointer"
                    style={{ border: "1px solid #E5E5E5" }}
                  >
                    <div
                      className="w-2 shrink-0"
                      style={{ backgroundColor: stripColor }}
                    ></div>
                    <div className="flex-1 flex items-center justify-between px-6">
                      <div className="flex flex-col">
                        <span
                          className="text-lg font-bold group-hover:text-[#0019A8] transition-colors"
                          style={{ color: "#121212" }}
                        >
                          {item.filename}
                        </span>
                        <span
                          className="text-sm font-medium"
                          style={{ color: statusColor }}
                        >
                          {statusText}
                        </span>
                      </div>
                      <div className="flex items-center gap-4">
                        <span
                          className="text-sm font-bold"
                          style={{ color: "#121212" }}
                        >
                          {formatPercent(item.forensic_risk_score)}
                        </span>
                        <span
                          className="text-xs font-medium"
                          style={{ color: "#737373" }}
                        >
                          {formatRelativeTime(item.created_at)}
                        </span>
                        <span
                          className="material-symbols-outlined group-hover:text-[#0019A8] transition-colors"
                          style={{ color: "#737373" }}
                        >
                          chevron_right
                        </span>
                      </div>
                    </div>
                  </Link>
                );
              })
            ) : (
              <div
                className="p-12 text-center text-sm font-medium"
                style={{ color: "#737373" }}
              >
                No documents pending review.
              </div>
            )}
          </div>

          {/* Bottom Action Area — clone of findings_status/code.html line 189-195 */}
          <div
            className="p-8"
            style={{
              borderTop: "1px solid #E5E5E5",
              backgroundColor: "#F4F6F8",
            }}
          >
            <Link
              href="/analyst/override-history"
              className="w-full font-bold h-14 rounded-xl hover:opacity-90 transition-colors flex items-center justify-center gap-2 text-white"
              style={{ backgroundColor: "#0019A8" }}
            >
              <span className="material-symbols-outlined">history</span>
              Open Override History
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
