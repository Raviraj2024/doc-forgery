"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { getProfileFromPathname, type UserProfile } from "@/lib/auth";
import { cn } from "@/lib/utils";

/* ─────────────────────────────────────────────────────────────────
   Icon mapping from design_assets/ai_engine_diagnostics/code.html
   Use solid, simple Material Symbols matching the company style.
───────────────────────────────────────────────────────────────── */

type SidebarItem = {
  name: string;
  icon: string; // Material Symbol name
  href: string;
};

const profileItems: Record<UserProfile, SidebarItem[]> = {
  analyst: [
    { name: "Review Queue", icon: "manage_search", href: "/analyst/queue" },
    { name: "Forensic Lab", icon: "biotech", href: "/analyst/forensics" },
    {
      name: "Override History",
      icon: "history",
      href: "/analyst/override-history",
    },
  ],
  submitter: [
    {
      name: "My Submissions",
      icon: "inventory_2",
      href: "/submitter/my-submissions",
    },
    { name: "Upload Center", icon: "upload_file", href: "/submitter/upload" },
    {
      name: "Certificates",
      icon: "workspace_premium",
      href: "/submitter/certificates",
    },
  ],
  compliance: [
    {
      name: "Global Overview",
      icon: "dashboard",
      href: "/compliance/overview",
    },
    {
      name: "Audit Log",
      icon: "fact_check",
      href: "/compliance/audit-log",
    },
    {
      name: "Policy Config",
      icon: "policy",
      href: "/compliance/policy-config",
    },
    {
      name: "Reports",
      icon: "lab_profile",
      href: "/compliance/reports",
    },
  ],
  devops: [{ name: "System Health", icon: "hub", href: "/devops/dashboard" }],
};

/* Bottom utility items — always shown */
const bottomItems: SidebarItem[] = [
  { name: "Settings", icon: "settings", href: "#" },
  { name: "Help", icon: "help_outline", href: "#" },
];

export function Sidebar() {
  const pathname = usePathname();

  // Determine active profile from path segment
  const profile = getProfileFromPathname(pathname) ?? "analyst";
  const items = profileItems[profile];
  const utilityItems =
    profile === "analyst" || profile === "submitter" || profile === "compliance"
      ? []
      : bottomItems;

  return (
    <aside
      className="fixed inset-y-0 left-0 z-50 flex w-64 flex-col"
      style={{
        backgroundColor: "#131022",
        borderRight: "1px solid rgba(255,255,255,0.08)",
      }}
    >
      {/* ── Brand Header ── */}
      <div className="flex items-center gap-4 px-6 py-6 border-b border-white/5">
        <span
          className="material-symbols-outlined text-white"
          style={{
            fontSize: "28px",
            fontVariationSettings: "'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24",
          }}
        >
          security
        </span>
        <span
          className="text-lg font-bold tracking-tight text-white"
          style={{ whiteSpace: "nowrap" }}
        >
          Operational Trust
        </span>
      </div>

      {/* ── Profile pill ── */}
      <div
        className="mx-4 mt-6 mb-4 rounded-full px-4 py-2"
        style={{ backgroundColor: "rgba(255,255,255,0.06)" }}
      >
        <p
          className="text-[10px] font-bold uppercase tracking-[0.18em]"
          style={{ color: "rgba(255,255,255,0.45)" }}
        >
          Active Profile
        </p>
        <p className="text-sm font-bold capitalize text-white">{profile}</p>
      </div>

      {/* ── Nav Items ── */}
      <nav className="flex-1 overflow-y-auto px-2 pb-4">
        <ul className="space-y-1 mt-2">
          {items.map((item) => {
            const isActive =
              pathname === item.href ||
              (item.href !== "/" && pathname.startsWith(item.href));

            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className={cn(
                    "group relative flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-bold transition-all duration-150",
                    isActive
                      ? "text-white bg-[#0f2bb9]" // active tab color matching royal blue #2109aa variation for dark bg
                      : "hover:text-white",
                  )}
                  style={!isActive ? { color: "rgba(255,255,255,0.60)" } : {}}
                  onMouseEnter={(e) => {
                    if (!isActive) {
                      (e.currentTarget as HTMLElement).style.backgroundColor =
                        "rgba(255,255,255,0.05)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive) {
                      (e.currentTarget as HTMLElement).style.backgroundColor =
                        "";
                    }
                  }}
                >
                  {/* Left bar */}
                  {isActive && (
                    <span
                      className="absolute left-0 top-2 bottom-2 w-1 rounded-r-md"
                      style={{ backgroundColor: "#3a61f2" }}
                    />
                  )}

                  {/* Material Symbol icon */}
                  <span
                    className="material-symbols-outlined shrink-0 transition-colors"
                    style={{
                      fontSize: "20px",
                      color: isActive ? "#ffffff" : "rgba(255,255,255,0.50)",
                      fontVariationSettings:
                        "'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24",
                    }}
                  >
                    {item.icon}
                  </span>

                  <span className="tracking-wide">{item.name}</span>
                </Link>
              </li>
            );
          })}
        </ul>

        {utilityItems.length > 0 ? (
          <>
            {/* ── Section Divider ── */}
            <div
              className="my-6 mx-2"
              style={{
                height: "1px",
                backgroundColor: "rgba(255,255,255,0.08)",
              }}
            />

            {/* ── Bottom utility links ── */}
            <ul className="space-y-1">
              {utilityItems.map((item) => (
                <li key={`bottom-${item.name}`}>
                  <Link
                    href={item.href}
                    className="flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-bold transition-all duration-150"
                    style={{ color: "rgba(255,255,255,0.40)" }}
                    onMouseEnter={(e) => {
                      (e.currentTarget as HTMLElement).style.backgroundColor =
                        "rgba(255,255,255,0.05)";
                      (e.currentTarget as HTMLElement).style.color =
                        "rgba(255,255,255,0.80)";
                    }}
                    onMouseLeave={(e) => {
                      (e.currentTarget as HTMLElement).style.backgroundColor =
                        "";
                      (e.currentTarget as HTMLElement).style.color =
                        "rgba(255,255,255,0.40)";
                    }}
                  >
                    <span
                      className="material-symbols-outlined shrink-0 text-white/50"
                      style={{
                        fontSize: "20px",
                        fontVariationSettings:
                          "'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24",
                      }}
                    >
                      {item.icon}
                    </span>
                    <span>{item.name}</span>
                  </Link>
                </li>
              ))}
            </ul>
          </>
        ) : null}
      </nav>

      {/* ── Version Footer ── */}
      <div
        className="px-6 py-5"
        style={{ borderTop: "1px solid rgba(255,255,255,0.08)" }}
      >
        <div className="flex items-center gap-3">
          <div
            className="flex h-9 w-9 items-center justify-center rounded-lg text-sm font-bold text-white"
            style={{ backgroundColor: "rgba(255,255,255,0.12)" }}
          >
            {profile.charAt(0).toUpperCase()}
          </div>
          <div className="overflow-hidden">
            <p className="truncate text-sm font-bold capitalize text-white">
              {profile}
            </p>
            <p
              className="text-[10px] font-semibold uppercase tracking-wider"
              style={{ color: "rgba(255,255,255,0.35)" }}
            >
              Enterprise Node · Active
            </p>
          </div>
        </div>
      </div>
    </aside>
  );
}
