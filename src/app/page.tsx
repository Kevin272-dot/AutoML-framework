import { DiscoverySearchForm } from "@/components/discovery/DiscoverySearchForm";

export default function HomePage() {
  return (
    <div className="flex min-h-[70vh] items-center justify-center">
      <div className="w-full px-4">
        <DiscoverySearchForm />
      </div>
    </div>
  );
}
