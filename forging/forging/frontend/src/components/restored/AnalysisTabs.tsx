import Link from "next/link";
import { cn } from "@/lib/utils";

const TAB_DEFS = [
  { key: "status", label: "Status", suffix: "" },
  { key: "evidence", label: "Evidence", suffix: "/evidence" },
  { key: "diagnostics", label: "Diagnostics", suffix: "/diagnostics" },
  { key: "timeline", label: "Timeline", suffix: "/timeline" },
  { key: "case", label: "Case", suffix: "/case" },
] as const;

export function AnalysisTabs({
  caseId,
  active,
}: {
  caseId: string;
  active: "status" | "evidence" | "diagnostics" | "timeline" | "case";
}) {
  return (
    <div className="pb-0 bg-white border-b border-border-color flex-shrink-0">
      <div className="flex px-4 gap-8">
        {TAB_DEFS.map((tab) => (
          <Link
            key={tab.key}
            href={`/analyst/analysis/${caseId}${tab.suffix}`}
            className={cn(
              "flex flex-col items-center justify-center border-b-[4px] pb-[13px] pt-4 transition-colors",
              active === tab.key
                ? "border-b-primary text-text-main"
                : "border-b-transparent text-muted hover:text-text-main",
            )}
          >
            <p className="text-sm font-bold leading-normal tracking-[0.015em]">
              {tab.label}
            </p>
          </Link>
        ))}
      </div>
    </div>
  );
}
