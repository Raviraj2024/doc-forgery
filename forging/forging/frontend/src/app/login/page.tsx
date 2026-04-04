"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  AUTH_PROFILE_COOKIE,
  isValidProfilePassword,
  PROFILE_HOME_ROUTES,
  resolveProfileFromEmail,
} from "@/lib/auth";

const SunflowerHero = dynamic(() => import("@/components/ui/SunflowerHero"), {
  ssr: false,
});

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const handleSignIn = (event: React.FormEvent) => {
    event.preventDefault();
    const profile = resolveProfileFromEmail(email);

    if (!isValidProfilePassword(profile, password)) {
      setError("Incorrect password. Use abc123 to access this profile.");
      return;
    }

    setError(null);
    document.cookie = `${AUTH_PROFILE_COOKIE}=${profile}; Path=/; SameSite=Lax`;
    router.push(PROFILE_HOME_ROUTES[profile]);
  };

  return (
    <div className="fixed inset-0 overflow-hidden bg-[#f8f9fa] text-[#191c1d]">
      <SunflowerHero />

      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(circle at 50% 46%, rgba(255,255,255,0.54), transparent 24%), linear-gradient(180deg, rgba(255,255,255,0.66), rgba(248,249,250,0.16) 20%, rgba(248,249,250,0.34) 100%)",
        }}
      />

      <div className="relative z-10 mx-auto flex h-full w-full max-w-[1680px] flex-col px-5 py-5 md:px-8 md:py-6 lg:scale-[0.94] lg:origin-center 2xl:scale-[0.9]">
        <header className="flex items-center justify-start gap-6">
          <Link
            className="text-[26px] font-bold tracking-[-0.05em] text-[#0019a8] md:text-[30px]"
            href="#"
          >
            SequelForensics
          </Link>
        </header>

        <main className="flex flex-1 items-center justify-center">
          <div className="relative flex w-full max-w-[1500px] items-center justify-center">
            <div
              className="pointer-events-none absolute h-[470px] w-[470px] rounded-full blur-[120px]"
              style={{
                background:
                  "radial-gradient(circle, rgba(240,201,66,0.22) 0%, rgba(0,25,168,0.08) 58%, transparent 80%)",
              }}
            />

            <section className="relative w-full max-w-[374px]">
              <div
                className="absolute inset-[-18px] rounded-[36px] opacity-72 blur-[30px]"
                style={{
                  background:
                    "radial-gradient(circle at 50% 30%, rgba(0,25,168,0.18), transparent 40%), radial-gradient(circle at 54% 78%, rgba(240,201,66,0.20), transparent 32%)",
                }}
              />

              <div
                className="relative rounded-[30px] px-6 py-6 shadow-[0_22px_60px_rgba(0,13,110,0.06)] md:px-7 md:py-7"
                style={{
                  backgroundColor: "rgba(255,255,255,0.54)",
                  backdropFilter: "blur(24px)",
                  WebkitBackdropFilter: "blur(24px)",
                }}
              >
                <div
                  className="absolute inset-[1px] rounded-[29px]"
                  style={{
                    background:
                      "linear-gradient(180deg, rgba(255,255,255,0.52), rgba(243,244,245,0.26))",
                  }}
                />

                <div
                  className="absolute inset-0 rounded-[30px] opacity-60"
                  style={{
                    backgroundImage:
                      "radial-gradient(circle at 24% 22%, rgba(0,25,168,0.08), transparent 18%), radial-gradient(circle at 76% 30%, rgba(240,201,66,0.14), transparent 16%), radial-gradient(circle at 42% 82%, rgba(0,25,168,0.06), transparent 16%)",
                  }}
                />

                <div className="relative z-10">
                  <div className="flex justify-center">
                    <div
                      className="flex h-12 w-12 items-center justify-center rounded-full text-white shadow-[0_14px_28px_rgba(0,25,168,0.18)]"
                      style={{
                        background:
                          "linear-gradient(180deg, #1a42da 0%, #0019a8 100%)",
                      }}
                    >
                      <span
                        className="material-symbols-outlined"
                        style={{
                          fontSize: "26px",
                          fontVariationSettings:
                            "'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24",
                        }}
                      >
                        fingerprint
                      </span>
                    </div>
                  </div>

                  <div className="mt-5 text-center">
                    <h1 className="text-[34px] font-semibold tracking-[-0.055em] text-[#181b20] md:text-[38px]">
                      Sign in
                    </h1>
                    <p className="mt-2 text-[13px] leading-5 text-[#68707d]">
                      Use your workspace email to continue.
                    </p>
                  </div>

                  <form className="mt-7 space-y-3.5" onSubmit={handleSignIn}>
                    <label className="block">
                      <span className="block text-[12px] font-medium text-[#5f6673]">
                        Email
                      </span>
                      <div
                        className="mt-2 rounded-full px-5 py-1"
                        style={{
                          backgroundColor: "rgba(255,255,255,0.74)",
                          outline: "1px solid rgba(197,197,214,0.28)",
                        }}
                      >
                        <input
                          className="h-10 w-full bg-transparent text-[14px] text-[#1a1d21] outline-none placeholder:text-[#99a0ae]"
                          onChange={(event) => {
                            setEmail(event.target.value);
                            if (error) {
                              setError(null);
                            }
                          }}
                          placeholder="analyst@company.com"
                          required
                          type="email"
                          value={email}
                        />
                      </div>
                    </label>

                    <label className="block">
                      <div className="flex items-center justify-between">
                        <span className="block text-[12px] font-medium text-[#5f6673]">
                          Password
                        </span>
                        <Link
                          className="text-[12px] font-medium text-[#68707d] transition-colors hover:text-[#0019a8]"
                          href="#"
                        >
                          Forgot?
                        </Link>
                      </div>

                      <div
                        className="mt-2 flex items-center rounded-full px-5 py-1"
                        style={{
                          backgroundColor: "rgba(255,255,255,0.74)",
                          outline: "1px solid rgba(197,197,214,0.28)",
                        }}
                      >
                        <input
                          className="h-10 w-full bg-transparent text-[14px] text-[#1a1d21] outline-none placeholder:text-[#99a0ae]"
                          onChange={(event) => {
                            setPassword(event.target.value);
                            if (error) {
                              setError(null);
                            }
                          }}
                          placeholder="password"
                          required
                          type={showPassword ? "text" : "password"}
                          value={password}
                        />
                        <button
                          className="ml-3 text-[#7d8190] transition-colors hover:text-[#0019a8]"
                          onClick={() => setShowPassword(!showPassword)}
                          type="button"
                        >
                          <span
                            className="material-symbols-outlined"
                            style={{
                              fontSize: "22px",
                              fontVariationSettings:
                                "'FILL' 0, 'wght' 300, 'GRAD' 0, 'opsz' 24",
                            }}
                          >
                            {showPassword ? "visibility_off" : "visibility"}
                          </span>
                        </button>
                      </div>
                    </label>

                    {error ? (
                      <div className="rounded-[20px] border border-[rgba(208,37,37,0.14)] bg-[rgba(208,37,37,0.08)] px-4 py-3 text-[12px] font-medium text-[#b42318]">
                        {error}
                      </div>
                    ) : null}

                    <button
                      className="mt-1 flex h-12 w-full items-center justify-center gap-2 rounded-full text-[14px] font-semibold text-white transition-transform hover:scale-[1.01]"
                      style={{
                        background:
                          "linear-gradient(180deg, #0019a8 0%, #000d6e 100%)",
                        boxShadow: "0 18px 30px rgba(0,25,168,0.14)",
                      }}
                      type="submit"
                    >
                      <span>Continue</span>
                      <span
                        className="material-symbols-outlined"
                        style={{
                          fontSize: "20px",
                          fontVariationSettings:
                            "'FILL' 0, 'wght' 300, 'GRAD' 0, 'opsz' 24",
                        }}
                      >
                        arrow_forward
                      </span>
                    </button>
                  </form>

                  <div className="mt-5 text-center">
                    <p className="text-[13px] text-[#666c79]">
                      Need access?{" "}
                      <Link
                        className="font-semibold text-[#0019a8] transition-opacity hover:opacity-80"
                        href="#"
                      >
                        Request access
                      </Link>
                    </p>
                  </div>
                </div>
              </div>
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}
