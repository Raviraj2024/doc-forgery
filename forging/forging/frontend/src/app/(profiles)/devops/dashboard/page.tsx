import {
  fetchDevOpsMonitoring,
  fetchHealth,
  fetchModelInfo,
  fetchDevOpsTelemetry,
} from "@/lib/api";
import { formatDateTime, formatPercent } from "@/lib/format";

export default async function DevopsDashboardPage() {
  const [health, modelInfo, telemetry, monitoring] = await Promise.all([
    fetchHealth().catch(() => null),
    fetchModelInfo().catch(() => null),
    fetchDevOpsTelemetry().catch(() => []),
    fetchDevOpsMonitoring().catch(() => null),
  ]);

  const cards = [
    ["Health Endpoint", health?.status === "ok" ? "OK" : "Unavailable"],
    ["Model Loaded", modelInfo?.model_loaded ? "Yes" : "No"],
    ["Database Ready", health?.database_ready ? "Yes" : "No"],
    ["Checkpoint Exists", health?.checkpoint_exists ? "Yes" : "No"],
  ];

  return (
    <div className="flex min-h-screen flex-col bg-background-light text-text-main">
      <header className="border-b border-border-color bg-white px-6 py-6 shadow-subtle lg:px-10">
        <p className="text-xs font-bold uppercase tracking-[0.22em] text-primary">
          DevOps
        </p>
        <h1 className="mt-2 text-3xl font-bold tracking-tight">
          Backend system health
        </h1>
      </header>

      <main className="grid flex-1 gap-8 p-6 lg:grid-cols-[minmax(0,1fr)_380px] lg:p-8">
        <section className="rounded-[28px] border border-border-color bg-white p-6 shadow-subtle">
          <div className="grid gap-4 md:grid-cols-2">
            {cards.map(([label, value]) => (
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

          <div className="mt-8 grid gap-4 md:grid-cols-2">
            <div className="rounded-[24px] border border-border-color bg-background-light p-5">
              <p className="text-xs font-bold uppercase tracking-[0.18em] text-muted">
                Checkpoint
              </p>
              <p className="mt-4 break-all text-base font-bold">
                {modelInfo?.checkpoint_path ?? "Unavailable"}
              </p>
            </div>
            <div className="rounded-[24px] border border-border-color bg-background-light p-5">
              <p className="text-xs font-bold uppercase tracking-[0.18em] text-muted">
                Encoder
              </p>
              <p className="mt-4 text-base font-bold">
                {modelInfo?.selected_encoder ?? "Unknown"}
              </p>
              <p className="mt-2 text-sm font-medium text-muted">
                Input channels {modelInfo?.input_channels ?? "?"}
              </p>
            </div>
          </div>

          <div className="mt-8 rounded-[24px] border border-border-color bg-background-light p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-bold uppercase tracking-[0.18em] text-muted">
                  Serving Calibration
                </p>
                <h2 className="mt-2 text-xl font-bold tracking-tight">
                  Threshold and benchmark profile
                </h2>
              </div>
              <span
                className={`rounded-full px-3 py-1 text-xs font-bold ${
                  monitoring?.calibration_loaded
                    ? "bg-accent-green/10 text-accent-green"
                    : "bg-accent-amber/10 text-accent-amber"
                }`}
              >
                {monitoring?.calibration_loaded ? "Loaded" : "Not Loaded"}
              </span>
            </div>

            <div className="mt-5 grid gap-4 md:grid-cols-3">
              <div className="rounded-2xl border border-border-color bg-white px-4 py-3">
                <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-muted">
                  Samples
                </p>
                <p className="mt-1 text-lg font-bold">
                  {monitoring?.calibration_sample_count ?? 0}
                </p>
              </div>
              <div className="rounded-2xl border border-border-color bg-white px-4 py-3">
                <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-muted">
                  Mean IoU
                </p>
                <p className="mt-1 text-lg font-bold">
                  {monitoring?.calibration_mean_iou != null
                    ? formatPercent(monitoring.calibration_mean_iou)
                    : "Unavailable"}
                </p>
              </div>
              <div className="rounded-2xl border border-border-color bg-white px-4 py-3">
                <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-muted">
                  Mean F1
                </p>
                <p className="mt-1 text-lg font-bold">
                  {monitoring?.calibration_mean_f1 != null
                    ? formatPercent(monitoring.calibration_mean_f1)
                    : "Unavailable"}
                </p>
              </div>
            </div>
            <p className="mt-4 text-sm font-medium text-muted">
              Generated{" "}
              {monitoring?.calibration_generated_at
                ? formatDateTime(monitoring.calibration_generated_at)
                : "not yet benchmarked"}
            </p>
          </div>
        </section>

        <aside className="space-y-4">
          <div className="rounded-[28px] border border-border-color bg-white p-6 shadow-subtle">
            <h2 className="text-2xl font-bold tracking-tight">
              Runtime Monitoring
            </h2>
            <div className="mt-6 grid gap-3">
              {[
                ["Total Analyses", `${monitoring?.total_analyses ?? 0}`],
                ["Warning Rate", formatPercent(monitoring?.warning_rate ?? 0)],
                ["P50 Runtime", `${Math.round(monitoring?.p50_processing_time_ms ?? 0)}ms`],
                ["P95 Runtime", `${Math.round(monitoring?.p95_processing_time_ms ?? 0)}ms`],
              ].map(([label, value]) => (
                <div
                  className="flex items-center justify-between rounded-2xl border border-border-color bg-background-light px-4 py-3"
                  key={label}
                >
                  <span className="text-sm font-bold uppercase tracking-widest text-muted">
                    {label}
                  </span>
                  <span className="text-sm font-bold text-text-main">{value}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-[28px] border border-border-color bg-white p-6 shadow-subtle">
          <h2 className="text-2xl font-bold tracking-tight">
            Engine Latency & Telemetry
          </h2>
          <div className="mt-6 space-y-3">
            {telemetry.length === 0 ? (
              <p className="text-sm text-muted text-center py-10">No execution telemetry recorded.</p>
            ) : (
              telemetry.map((layer) => (
                <div
                  className="flex items-center justify-between rounded-2xl border border-border-color bg-background-light px-4 py-3"
                  key={layer.layer_name}
                >
                  <div className="flex flex-col">
                    <span className="text-sm font-bold uppercase tracking-widest">{layer.layer_name.replace(/_/g, " ")}</span>
                    <span className="text-xs font-medium text-muted mt-1">{layer.execution_count} inferences</span>
                  </div>
                  <div className="flex flex-col items-end">
                    <span className="text-sm font-bold tracking-widest text-[#0019a8]">
                      {Math.round(layer.avg_processing_ms)}ms
                    </span>
                    <span className="text-xs font-medium text-muted mt-1">{formatPercent(layer.avg_confidence_score)} conf</span>
                  </div>
                </div>
              ))
            )}
          </div>
          </div>

          <div className="rounded-[28px] border border-border-color bg-white p-6 shadow-subtle">
            <h2 className="text-2xl font-bold tracking-tight">
              Recent Warnings
            </h2>
            <div className="mt-6 space-y-3">
              {(monitoring?.recent_warning_events ?? []).length === 0 ? (
                <p className="py-8 text-center text-sm text-muted">
                  No recent pipeline warnings recorded.
                </p>
              ) : (
                monitoring?.recent_warning_events.map((event) => (
                  <div
                    className="rounded-2xl border border-border-color bg-background-light px-4 py-3"
                    key={`${event.analysis_id}-${event.warning}`}
                  >
                    <p className="text-sm font-bold">{event.filename}</p>
                    <p className="mt-1 text-xs font-medium uppercase tracking-[0.18em] text-muted">
                      {event.analysis_id}
                    </p>
                    <p className="mt-3 text-sm font-medium text-text-main">
                      {event.warning}
                    </p>
                  </div>
                ))
              )}
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
