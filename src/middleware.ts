import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";

const CLERK_ENABLED = !!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;

// Everything except the auth pages, static assets and API callbacks is protected.
const isProtectedRoute = createRouteMatcher([
  "/((?!sign-in|sign-up|_next|favicon.ico|.*\\.png$|.*\\.svg$|.*\\.ico$).*)",
]);

export default CLERK_ENABLED
  ? clerkMiddleware(async (auth, req) => {
      if (isProtectedRoute(req)) {
        await auth.protect();
      }
    })
  : function devMiddleware() {
      return NextResponse.next();
    };

export const config = {
  matcher: [
    // Skip Next.js internals and static files.
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    "/(api|trpc)(.*)",
  ],
};
