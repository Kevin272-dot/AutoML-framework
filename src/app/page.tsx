import { auth } from "@clerk/nextjs/server";
import { redirect } from "next/navigation";
import { DiscoverySearchForm } from "@/components/discovery/DiscoverySearchForm";

export default async function HomePage() {
  // Signed-out visitors land on sign-up; anonymous browsing is disabled when Clerk is configured.
  if (process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY) {
    const { userId } = await auth();
    if (!userId) {
      redirect("/sign-up");
    }
  }

  return (
    <div className="flex min-h-[70vh] items-center justify-center">
      <div className="w-full px-4">
        <DiscoverySearchForm />
      </div>
    </div>
  );
}
