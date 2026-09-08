"use client";
// src/app/page.tsx -- Home / dataset requirement search
import { useState } from "react";
import { useRouter } from "next/navigation";

export default function HomePage() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [dateStart, setDateStart] = useState("");
  const [dateEnd, setDateEnd] = useState("");

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    const params = new URLSearchParams({ query });
    if (dateStart) params.set("date_start", dateStart);
    if (dateEnd) params.set("date_end", dateEnd);
    router.push(`/search?${params.toString()}`);
  };

  const examples = [
    "traffic congestion chennai 2023-2025",
    "crop yield agriculture india",
    "telecom customer churn",
  ];

  return (
    <main className="container" style={{ paddingTop: "80px" }}>
      <span className="badge badge-score" style={{ marginBottom: "20px", display: "inline-block" }}>
        DATASET DISCOVERY + AUTOML
      </span>
      <h1>What dataset are you looking for?</h1>
      <p className="subtitle">
        Describe your ML problem in plain language. We&apos;ll find a suitable dataset,
        show you what&apos;s inside it, then automatically run EDA, preprocessing,
        model training, tuning, and evaluation.
      </p>

      <form onSubmit={handleSearch} className="search-box">
        <input
          type="text"
          placeholder='e.g. "traffic congestion in Chennai from 2023 to 2025"'
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <input
          type="text"
          placeholder="Start year (optional)"
          style={{ flex: "0 0 160px" }}
          value={dateStart}
          onChange={(e) => setDateStart(e.target.value)}
        />
        <input
          type="text"
          placeholder="End year (optional)"
          style={{ flex: "0 0 160px" }}
          value={dateEnd}
          onChange={(e) => setDateEnd(e.target.value)}
        />
        <button type="submit" className="btn btn-primary">Search</button>
      </form>

      <p style={{ color: "var(--muted)", fontSize: ".85rem", marginBottom: "12px" }}>Try:</p>
      <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
        {examples.map((ex) => (
          <button
            key={ex}
            className="btn btn-outline"
            onClick={() => router.push(`/search?query=${encodeURIComponent(ex)}`)}
          >
            {ex}
          </button>
        ))}
      </div>
    </main>
  );
}
