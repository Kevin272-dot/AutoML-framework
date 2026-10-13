import Link from "next/link";
import { ThemeToggle } from "@/components/theme/ThemeToggle";

export default function MarketingLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-40 border-b border-border bg-bg/95 backdrop-blur">
        <div className="mx-auto flex h-14 w-full max-w-6xl items-center justify-between px-6">
          <Link href="/" className="flex items-center gap-2">
            <span className="flex size-7 items-center justify-center rounded bg-accent/15 text-xs font-bold text-accent">IA</span>
            <span className="text-sm font-semibold tracking-tight">Intelligent AutoML</span>
          </Link>
          <div className="flex items-center gap-6">
            <nav className="hidden items-center gap-6 text-sm text-muted sm:flex" aria-label="Marketing">
              <a href="#pipeline" className="hover:text-text">How it works</a>
              <a href="#features" className="hover:text-text">Platform</a>
            </nav>
            <ThemeToggle />
          </div>
        </div>
      </header>
      {children}
      <footer className="border-t border-border py-8">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 text-xs text-faint">
          <span>Intelligent AutoML — Dataset Discovery + Automated Machine Learning</span>
          <span>Every gate needs a human.</span>
        </div>
      </footer>
    </div>
  );
}
