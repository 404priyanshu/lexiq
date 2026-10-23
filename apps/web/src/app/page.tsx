import { Suspense } from "react";

import { getHealth } from "@/lib/api";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-8 px-4 py-16">
      <header className="flex flex-col gap-2">
        <h1 className="text-3xl font-semibold tracking-tight">LexIQ</h1>
        <p className="text-foreground/70">
          Citation-grounded answers over your documents. Upload and ask are coming in the
          next phases.
        </p>
      </header>

      <section className="rounded-lg border border-foreground/15 p-4">
        <h2 className="mb-3 text-sm font-medium uppercase tracking-wide text-foreground/60">
          Backend status
        </h2>
        {/* Uncached fetch: rendered per request, streamed into the static shell. */}
        <Suspense fallback={<p className="text-sm text-foreground/60">Checking…</p>}>
          <BackendStatus />
        </Suspense>
      </section>
    </main>
  );
}

async function BackendStatus() {
  const health = await getHealth();
  if (!health) {
    return <p className="text-sm text-red-600">API unreachable</p>;
  }

  return (
    <ul className="flex flex-col gap-1 font-mono text-sm">
      {Object.entries(health.checks).map(([name, state]) => (
        <li key={name} className="flex justify-between">
          <span>{name}</span>
          <span className={state === "ok" ? "text-emerald-600" : "text-red-600"}>{state}</span>
        </li>
      ))}
    </ul>
  );
}
