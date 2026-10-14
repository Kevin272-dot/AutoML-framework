import { notFound } from "next/navigation";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { promises as fs } from "fs";
import path from "path";

const DOCS: Record<string, { title: string; file: string }> = {
  "user-guide": { title: "User Guide", file: "USER_GUIDE.md" },
  "data-extraction": { title: "Data Extraction Engine", file: "DATA_EXTRACTION.md" },
  architecture: { title: "Architecture", file: "ARCHITECTURE.md" },
  integration: { title: "Integration Guide", file: "INTEGRATION.md" },
};

export function generateStaticParams() {
  return Object.keys(DOCS).map((slug) => ({ slug }));
}

export default async function DocPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const doc = DOCS[slug];
  if (!doc) notFound();

  const md = await fs.readFile(path.join(process.cwd(), "docs", doc.file), "utf-8");

  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <Link href="/guide" className="text-xs text-muted hover:text-text">
        ← Back to guide
      </Link>
      <article className="rounded-lg border border-border bg-surface p-8">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            h1: (p) => <h1 className="mb-6 border-b border-border pb-3 text-2xl font-semibold" {...p} />,
            h2: (p) => <h2 className="mt-10 mb-4 border-b border-border pb-2 text-xl font-semibold" {...p} />,
            h3: (p) => <h3 className="mt-6 mb-3 text-base font-semibold" {...p} />,
            p: (p) => <p className="my-3 text-sm leading-relaxed text-muted" {...p} />,
            a: (p) => <a className="text-accent hover:underline" target="_blank" rel="noreferrer" {...p} />,
            ul: (p) => <ul className="my-3 list-disc space-y-1 pl-6 text-sm text-muted" {...p} />,
            ol: (p) => <ol className="my-3 list-decimal space-y-1 pl-6 text-sm text-muted" {...p} />,
            li: (p) => <li className="leading-relaxed" {...p} />,
            strong: (p) => <strong className="font-semibold text-text" {...p} />,
            blockquote: (p) => (
              <blockquote className="my-4 border-l-2 border-accent/50 bg-accent/5 px-4 py-2 text-sm text-muted" {...p} />
            ),
            code: ({ className, children, ...p }) => {
              const isBlock = /language-/.test(className ?? "") || String(children).includes("\n");
              if (isBlock) {
                return (
                  <code
                    className="block overflow-x-auto rounded-md border border-border bg-elevated p-3 font-mono text-[11px] leading-relaxed text-text"
                    {...p}
                  >
                    {children}
                  </code>
                );
              }
              return (
                <code className="rounded bg-elevated px-1.5 py-0.5 font-mono text-xs text-text" {...p}>
                  {children}
                </code>
              );
            },
            pre: (p) => <pre className="my-3" {...p} />,
            table: (p) => (
              <div className="my-4 overflow-x-auto rounded-md border border-border">
                <table className="w-full text-sm" {...p} />
              </div>
            ),
            th: (p) => <th className="border-b border-border bg-elevated/60 px-3 py-2 text-left text-xs font-medium uppercase tracking-wider text-muted" {...p} />,
            td: (p) => <td className="border-b border-border/60 px-3 py-2 align-top text-muted" {...p} />,
            hr: (p) => <hr className="my-8 border-border" {...p} />,
          }}
        >
          {md}
        </ReactMarkdown>
      </article>
    </div>
  );
}
