import { SignUp } from "@clerk/nextjs";
import Link from "next/link";

export default function SignUpPage() {
  if (!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY) {
    return (
      <div className="mx-auto max-w-md pt-20 text-center">
        <h1 className="text-lg font-semibold">Authentication is not configured</h1>
        <p className="mt-2 text-sm text-muted">
          Add <code className="rounded bg-elevated px-1 py-0.5 text-xs">NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY</code> and{" "}
          <code className="rounded bg-elevated px-1 py-0.5 text-xs">CLERK_SECRET_KEY</code> to{" "}
          <code className="rounded bg-elevated px-1 py-0.5 text-xs">.env.local</code> to enable sign-up.
        </p>
        <Link href="/" className="mt-4 inline-block text-sm text-accent hover:underline">
          Back to home
        </Link>
      </div>
    );
  }
  return (
    <div className="flex min-h-[80vh] flex-col items-center justify-center gap-6">
      <Link href="/" className="flex items-center gap-2">
        <span className="flex size-7 items-center justify-center rounded bg-accent/15 text-xs font-bold text-accent">IA</span>
        <span className="text-base font-semibold tracking-tight">Intelligent AutoML</span>
      </Link>
      <SignUp signInUrl="/sign-in" fallbackRedirectUrl="/dashboard" />
    </div>
  );
}
