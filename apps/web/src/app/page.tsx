import { Suspense } from "react";

import { AskPanel } from "@/components/AskPanel";
import { DocumentsPanel } from "@/components/DocumentsPanel";
import { getHealth } from "@/lib/api";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-8 px-4 py-8">
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">LexIQ</h1>
          <p className="text-sm text-foreground/60">Citation-grounded answers over your documents.</p>
        </div>
        {/* Uncached fetch: rendered per request, streamed into the static shell. */}
        <Suspense fallback={null}>
          <BackendStatus />
        </Suspense>
      </header>

      <div className="grid gap-8 md:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
        <DocumentsPanel />
        <AskPanel />
      </div>
    </main>
  );
}

async function BackendStatus() {
  const health = await getHealth();
  const ok = health?.status === "ok";
  const detail = health
    ? Object.entries(health.checks)
        .map(([k, v]) => `${k}: ${v}`)
        .join(", ")
    : "API unreachable";

  return (
    <p className="flex items-center gap-2 text-xs text-foreground/60" title={detail}>
      <span className={`size-2 rounded-full ${ok ? "bg-emerald-500" : "bg-red-500"}`} aria-hidden />
      {ok ? "Backend healthy" : health ? "Backend degraded" : "API unreachable"}
    </p>
  );
}
