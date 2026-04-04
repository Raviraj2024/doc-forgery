import { NextRequest, NextResponse } from "next/server";
import {
  AUTH_PROFILE_COOKIE,
  getProfileFromPathname,
  isUserProfile,
  PROFILE_HOME_ROUTES,
} from "./src/lib/auth";

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const requestedProfile = getProfileFromPathname(pathname);

  if (!requestedProfile) {
    return NextResponse.next();
  }

  const activeProfile = request.cookies.get(AUTH_PROFILE_COOKIE)?.value;

  if (!activeProfile || !isUserProfile(activeProfile)) {
    const loginUrl = new URL("/login", request.url);
    return NextResponse.redirect(loginUrl);
  }

  if (activeProfile !== requestedProfile) {
    const authorizedUrl = new URL(PROFILE_HOME_ROUTES[activeProfile], request.url);
    return NextResponse.redirect(authorizedUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico|.*\\..*).*)"],
};
