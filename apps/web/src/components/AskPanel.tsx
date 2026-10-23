"use client";

import { type FormEvent, Fragment, useRef, useState } from "react";

import { type AnswerDone, type Source, streamAnswer } from "@/lib/api";

type Phase = "idle" | "searching" | "writing" | "done" | "error";

const CITATION = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

function pages(s: Source) {
  return s.page_start === s.page_end ? `p. ${s.page_start}` : `pp. ${s.page_start}–${s.page_end}`;
}

export function AskPanel() {
  const [question, setQuestion] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [done, setDone] = useState<AnswerDone | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState<number | null>(null);
  const abort = useRef<AbortController | null>(null);

  async function ask(e?: FormEvent) {
    e?.preventDefault();
    const q = question.trim();
    if (!q) return;

    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    setPhase("searching");
    setAnswer("");
    setSources([]);
    setDone(null);
    setNotice(null);
    setError(null);
    setActive(null);

    try {
      for await (const ev of streamAnswer(q, controller.signal)) {
        if (ev.event === "sources") {
          setSources(ev.data.sources);
          setPhase("writing");
        } else if (ev.event === "delta") {
          setAnswer((a) => a + ev.data.text);
        } else if (ev.event === "notice") {
          setNotice("Answer continued on a fallback model after a safety check.");
        } else if (ev.event === "done") {
          setDone(ev.data);
          // The server's final text is authoritative (e.g. a refusal clears the partial).
          setAnswer(ev.data.answer);
          setPhase("done");
        } else if (ev.event === "error") {
          setError(ev.data.message);
          setPhase("error");
        }
      }
    } catch (err) {
      if (controller.signal.aborted) return;
      setError((err as Error).message);
      setPhase("error");
    }
  }

  function focusSource(n: number) {
    setActive(n);
    document.getElementById(`source-${n}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  const busy = phase === "searching" || phase === "writing";
  const invalid = new Set(done?.citations?.invalid ?? []);

  return (
    <section className="flex flex-col gap-5">
      <form onSubmit={ask} className="flex flex-col gap-2">
        <label htmlFor="question" className="text-sm font-medium uppercase tracking-wide text-foreground/60">
          Ask
        </label>
        <div className="flex gap-2">
          <textarea
            id="question"
            rows={2}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void ask();
              }
            }}
            placeholder="e.g. How much notice is needed to terminate?"
            className="min-w-0 flex-1 resize-none rounded-lg border border-foreground/20 bg-transparent px-3 py-2 text-sm outline-none focus:border-foreground/50"
          />
          <button
            type="submit"
            disabled={busy || !question.trim()}
            className="rounded-lg bg-foreground px-4 text-sm font-medium text-background disabled:opacity-40"
          >
            {busy ? "…" : "Ask"}
          </button>
        </div>
      </form>

      {phase !== "idle" && (
        <div className="flex flex-col gap-3" aria-live="polite">
          {phase === "searching" && <p className="text-sm text-foreground/60">Searching documents…</p>}
          {phase === "writing" && !answer && <p className="text-sm text-foreground/60">Writing answer…</p>}

          {answer && (
            <p className="whitespace-pre-wrap text-[15px] leading-7">
              {answer.split(CITATION).map((part, i) =>
                i % 2 === 0 ? (
                  <Fragment key={i}>{part}</Fragment>
                ) : (
                  part.split(",").map((raw) => {
                    const n = Number(raw.trim());
                    const ok = !invalid.has(n) && n >= 1 && n <= sources.length;
                    return ok ? (
                      <button
                        key={`${i}-${n}`}
                        type="button"
                        onClick={() => focusSource(n)}
                        className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded bg-foreground/10 px-1 align-text-top text-xs font-medium hover:bg-foreground/20"
                        aria-label={`Source ${n}`}
                      >
                        {n}
                      </button>
                    ) : (
                      <span key={`${i}-${n}`} className="mx-0.5 text-xs text-red-600" title="No such source">
                        [{n}?]
                      </span>
                    );
                  })
                ),
              )}
            </p>
          )}

          {done?.abstained && (
            <Callout tone="neutral">
              The documents don&apos;t contain enough to answer this.
              {sources.length > 0 && " The closest passages are below."}
            </Callout>
          )}
          {done?.refused && (
            <Callout tone="warn">The model declined to answer this question.</Callout>
          )}
          {done?.truncated && <Callout tone="warn">The answer was cut off.</Callout>}
          {done?.citations && done.citations.cited.length === 0 && (
            <Callout tone="warn">This answer has no citations; treat it with caution.</Callout>
          )}
          {notice && <Callout tone="neutral">{notice}</Callout>}
          {error && <Callout tone="error">{error}</Callout>}

          {done && (done.timings_ms.total || done.usage) && (
            <p className="font-mono text-xs text-foreground/50">
              {[
                done.timings_ms.ttft && `first token ${Math.round(done.timings_ms.ttft)} ms`,
                done.timings_ms.total && `total ${Math.round(done.timings_ms.total)} ms`,
                done.usage && `${done.usage.input_tokens}→${done.usage.output_tokens} tokens`,
                done.citations && `${Math.round(done.citations.coverage * 100)}% sentences cited`,
              ]
                .filter(Boolean)
                .join(" · ")}
            </p>
          )}
        </div>
      )}

      {sources.length > 0 && (
        <div className="flex flex-col gap-2">
          <h3 className="text-sm font-medium uppercase tracking-wide text-foreground/60">Sources</h3>
          <ol className="flex flex-col gap-2">
            {sources.map((s) => (
              <li
                key={s.n}
                id={`source-${s.n}`}
                className={`rounded-lg border p-3 text-sm transition-colors ${
                  active === s.n ? "border-foreground/50 bg-foreground/5" : "border-foreground/15"
                }`}
              >
                <button
                  type="button"
                  onClick={() => setActive(active === s.n ? null : s.n)}
                  className="flex w-full items-baseline justify-between gap-3 text-left"
                  aria-expanded={active === s.n}
                >
                  <span className="min-w-0 truncate">
                    <span className="mr-2 font-mono text-xs text-foreground/50">[{s.n}]</span>
                    <span className="font-medium">{s.document_title}</span>
                    <span className="text-foreground/50"> · {pages(s)}</span>
                  </span>
                  <span className="shrink-0 font-mono text-xs text-foreground/50">
                    {s.relevance.toFixed(2)}
                  </span>
                </button>
                <p
                  className={`mt-2 whitespace-pre-wrap text-foreground/75 ${active === s.n ? "" : "line-clamp-2"}`}
                >
                  {s.content}
                </p>
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}

function Callout({ tone, children }: { tone: "neutral" | "warn" | "error"; children: React.ReactNode }) {
  const style = {
    neutral: "border-foreground/15 bg-foreground/5",
    warn: "border-amber-500/30 bg-amber-500/10",
    error: "border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-400",
  }[tone];
  return <p className={`rounded-lg border px-3 py-2 text-sm ${style}`}>{children}</p>;
}
