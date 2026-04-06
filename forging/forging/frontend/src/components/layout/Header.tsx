"use client";

import React, { useState, useRef, useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { getProfileFromPathname, PROFILE_EMAILS, AUTH_PROFILE_COOKIE } from "@/lib/auth";
import { User, LogOut, ChevronDown } from "lucide-react";

interface HeaderProps {
  title: string;
  subtitle?: string;
  badge?: string;
}

export function Header({ title, subtitle, badge }: HeaderProps) {
  const pathname = usePathname();
  const router = useRouter();
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const profile = pathname ? getProfileFromPathname(pathname) : null;
  const email = profile ? PROFILE_EMAILS[profile] : "";

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSignOut = () => {
    document.cookie = `${AUTH_PROFILE_COOKIE}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;`;
    router.push("/login");
  };

  return (
    <header className="flex items-center justify-between px-8 py-8 bg-transparent">
      <div className="flex flex-col gap-1.5">
        <h1
          className="text-3xl font-bold leading-tight tracking-tight"
          style={{ color: "#121212" }}
        >
          {title}
        </h1>
        {subtitle && (
          <p
            className="text-xs font-bold uppercase tracking-[0.2em]"
            style={{ color: "#737373" }}
          >
            {subtitle}
          </p>
        )}
      </div>

      <div className="flex items-center gap-6 z-50">
        {badge && (
          /* TfL line-status style: green pill = "Good Service" */
          <div
            className="flex items-center gap-2 rounded-full px-5 py-2.5 text-xs font-bold text-white shadow-md"
            style={{ backgroundColor: "#00823b" }}
          >
            <span
              className="h-2 w-2 animate-pulse rounded-full bg-white"
              style={{ boxShadow: "0 0 6px rgba(255,255,255,0.8)" }}
            />
            {badge}
          </div>
        )}

        {profile && (
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              className="flex items-center gap-2 rounded-full bg-white py-1.5 pl-1.5 pr-3 shadow-sm border border-gray-200 transition-colors hover:bg-gray-50 focus:outline-none"
            >
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-[#0019a8]/10 text-[#0019a8]">
                <User size={18} />
              </div>
              <div className="hidden flex-col items-start md:flex ml-1">
                <span className="text-sm font-semibold capitalize tracking-tight" style={{ color: "#191c1d" }}>
                  {profile}
                </span>
              </div>
              <ChevronDown size={16} className="text-gray-400 ml-1" />
            </button>

            {isDropdownOpen && (
              <div className="absolute right-0 mt-2 w-56 transform rounded-xl bg-white p-2 shadow-lg border border-gray-100 ring-1 ring-black ring-opacity-5 transition-all">
                <div className="mb-2 px-3 py-2">
                  <span className="block text-sm font-semibold capitalize" style={{ color: "#191c1d" }}>
                    {profile} Workspace
                  </span>
                  <span className="block text-xs mt-0.5" style={{ color: "#68707d" }}>
                    {email}
                  </span>
                </div>
                <div className="h-px bg-gray-100 my-1" />
                <button
                  onClick={handleSignOut}
                  className="flex w-full items-center gap-2 rounded-lg px-3 py-2.5 text-sm font-medium text-red-600 transition-colors hover:bg-red-50 focus:outline-none"
                >
                  <LogOut size={16} />
                  Sign Out
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </header>
  );
}
