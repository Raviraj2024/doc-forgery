export const AUTH_PROFILE_COOKIE = "ot_profile";

export const PROFILE_HOME_ROUTES = {
  analyst: "/analyst/queue",
  submitter: "/submitter/upload",
  compliance: "/compliance/overview",
  devops: "/devops/dashboard",
} as const;

export const PROFILE_PASSWORDS = {
  analyst: "abc123",
  submitter: "abc123",
  compliance: "abc123",
  devops: "abc123",
} as const;

export type UserProfile = keyof typeof PROFILE_HOME_ROUTES;

export function isUserProfile(value: string): value is UserProfile {
  return value in PROFILE_HOME_ROUTES;
}

export function getProfileFromPathname(pathname: string): UserProfile | null {
  const segment = pathname.split("/").filter(Boolean)[0];

  if (!segment || !isUserProfile(segment)) {
    return null;
  }

  return segment;
}

export function resolveProfileFromEmail(email: string): UserProfile {
  const normalizedEmail = email.toLowerCase();

  if (normalizedEmail.includes("submitter")) {
    return "submitter";
  }

  if (normalizedEmail.includes("compliance")) {
    return "compliance";
  }

  if (normalizedEmail.includes("devops")) {
    return "devops";
  }

  return "analyst";
}

export function isValidProfilePassword(profile: UserProfile, password: string) {
  return PROFILE_PASSWORDS[profile] === password;
}
