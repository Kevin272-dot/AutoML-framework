"use client";

import { ClerkProvider } from "@clerk/nextjs";
import { createContext, useContext, useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider, useTheme } from "@/components/theme/ThemeProvider";

/** Clerk is enabled only when the publishable key is configured. Without keys the
 * app stays usable anonymously (dev), and auth UI is hidden instead of throwing. */
const CLERK_ENABLED = !!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;

const ClerkEnabledContext = createContext<boolean>(CLERK_ENABLED);

export function useClerkEnabled() {
  return useContext(ClerkEnabledContext);
}

const clerkAppearance = {
  variables: {
    colorPrimary: "#2f6fed",
    colorBackground: "#ffffff",
    colorText: "#171b23",
    colorTextSecondary: "#5b6474",
    colorInputBackground: "#f1f3f7",
    colorInputBorder: "#e1e5ec",
    borderRadius: "0.5rem",
  },
} as const;

function ThemeAwareBody({ children }: { children: React.ReactNode }) {
  const { resolvedTheme } = useTheme();
  // Clerk has no CSS-var theming; swap its light/dark variables with the app theme.
  const dark = resolvedTheme === "dark";
  const appearance = dark
    ? {
        variables: {
          colorPrimary: "#4c8dff",
          colorBackground: "#11141a",
          colorText: "#e2e6ee",
          colorTextSecondary: "#8b94a7",
          colorInputBackground: "#161a21",
          colorInputBorder: "#242a35",
          borderRadius: "0.5rem",
        },
      }
    : clerkAppearance;
  return (
    <ClerkEnabledContext.Provider value={CLERK_ENABLED}>
      {CLERK_ENABLED ? (
        <ClerkProvider
          appearance={appearance}
          signInUrl="/sign-in"
          signUpUrl="/sign-up"
          signInFallbackRedirectUrl="/dashboard"
          signUpFallbackRedirectUrl="/dashboard"
        >
          {children}
        </ClerkProvider>
      ) : (
        children
      )}
    </ClerkEnabledContext.Provider>
  );
}

export function AppProviders({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 15_000,
            retry: 1,
            refetchOnWindowFocus: false,
          },
        },
      })
  );
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <ThemeAwareBody>{children}</ThemeAwareBody>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
