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

      {/* grid-cols-1 is minmax(0, 1fr): without it the single mobile column grows to the
          longest unbroken document title instead of truncating it. */}
      <div className="grid grid-cols-1 gap-8 md:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
        <DocumentsPanel />
        {/* On phones the question box comes first, above a long document list. */}
        <div className="order-first md:order-none">
          <AskPanel />
        </div>
      </div>

      <footer className="mt-auto border-t border-foreground/10 pt-4 text-xs text-foreground/50">
        Contracts from{" "}
        <a className="underline" href="https://www.atticusprojectai.org/cuad">
          CUAD v1
        </a>
        , © The Atticus Project,{" "}
        <a className="underline" href="https://creativecommons.org/licenses/by/4.0/">
          CC BY 4.0
        </a>
        . Regulations from the 2025 Code of Federal Regulations via{" "}
        <a className="underline" href="https://www.govinfo.gov/app/collection/cfr">
          govinfo.gov
        </a>{" "}
        (public domain). Answers are generated and can be wrong; check the cited passages.
      </footer>
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
