import type { Metadata } from "next";
import "./globals.css";
import { AppProviders } from "./providers";
import { ThemeScript } from "@/components/theme/ThemeProvider";

export const metadata: Metadata = {
  title: "Intelligent AutoML — Dataset Discovery + AutoML",
  description:
    "Describe your ML problem. Discover and audit data sources, approve them, select a dataset, and train models with an RL-driven AutoML pipeline.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <ThemeScript />
      </head>
      <body>
        {/* ClerkProvider is mounted inside AppProviders (theme-aware, inside <body>). */}
        <AppProviders>{children}</AppProviders>
      </body>
    </html>
  );
}
