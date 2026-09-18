import type { Metadata } from "next";
import "./globals.css";
import { AppProviders } from "./providers";
import { ThemeScript } from "@/components/theme/ThemeProvider";
import { Sidebar } from "@/components/layout/Sidebar";
import { TopBar } from "@/components/layout/TopBar";

export const metadata: Metadata = {
  title: "Intelligent AutoML — Dataset Discovery + AutoML",
  description:
    "Describe your ML problem. Discover and audit data sources, approve them, select a dataset, and run the automated ML pipeline.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <ThemeScript />
      </head>
      <body>
        {/* ClerkProvider is mounted inside AppProviders (theme-aware, inside <body>). */}
        <AppProviders>
          <div className="flex min-h-screen">
            <Sidebar />
            <div className="flex min-w-0 flex-1 flex-col">
              <TopBar />
              <main className="flex-1 overflow-y-auto p-6">{children}</main>
            </div>
          </div>
        </AppProviders>
      </body>
    </html>
  );
}
