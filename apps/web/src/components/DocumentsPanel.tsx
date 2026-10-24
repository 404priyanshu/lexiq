"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { type DocumentItem, listDocuments, uploadDocument } from "@/lib/api";

const ACCEPT = ".pdf,.html,.htm,.md,.markdown,.txt";

const STATUS_STYLE: Record<DocumentItem["status"], string> = {
  pending: "bg-foreground/10 text-foreground/70",
  processing: "bg-amber-500/15 text-amber-700 dark:text-amber-400",
  ready: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400",
  failed: "bg-red-500/15 text-red-700 dark:text-red-400",
};

export function DocumentsPanel() {
  const [docs, setDocs] = useState<DocumentItem[] | null>(null);
  const [uploading, setUploading] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      setDocs(await listDocuments());
    } catch {
      setError("Couldn't reach the API.");
    }
  }, []);

  useEffect(() => {
    // Initial load; setState happens after the awaited fetch, not synchronously.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
  }, [refresh]);

  // Poll while anything is still being ingested.
  const busy = docs?.some((d) => d.status === "pending" || d.status === "processing");
  useEffect(() => {
    if (!busy) return;
    const id = setInterval(refresh, 1500);
    return () => clearInterval(id);
  }, [busy, refresh]);

  async function upload(files: FileList | File[]) {
    setError(null);
    const list = Array.from(files);
    setUploading((n) => n + list.length);
    await Promise.all(
      list.map(async (file) => {
        try {
          await uploadDocument(file);
        } catch (e) {
          setError(`${file.name}: ${(e as Error).message}`);
        } finally {
          setUploading((n) => n - 1);
        }
      }),
    );
    await refresh();
  }

  return (
    <section className="flex flex-col gap-3">
      <h2 className="text-sm font-medium uppercase tracking-wide text-foreground/60">Documents</h2>

      <button
        type="button"
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          void upload(e.dataTransfer.files);
        }}
        className={`rounded-lg border border-dashed px-4 py-6 text-sm transition-colors ${
          dragging ? "border-foreground/60 bg-foreground/5" : "border-foreground/20 hover:bg-foreground/5"
        }`}
      >
        {uploading > 0 ? `Uploading ${uploading}…` : "Drop files or click to upload"}
        <span className="mt-1 block text-xs text-foreground/50">PDF, HTML, Markdown, text</span>
      </button>
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        multiple
        hidden
        onChange={(e) => {
          if (e.target.files?.length) void upload(e.target.files);
          e.target.value = "";
        }}
      />

      {error && <p className="text-sm text-red-600">{error}</p>}

      <ul className="flex flex-col divide-y divide-foreground/10">
        {docs?.length === 0 && (
          <li className="py-3 text-sm text-foreground/50">No documents yet.</li>
        )}
        {docs?.map((d) => (
          <li key={d.id} className="flex items-start justify-between gap-3 py-2.5">
            <div className="min-w-0">
              <p className="truncate text-sm font-medium" title={d.title}>
                {d.title}
              </p>
              <p className="text-xs text-foreground/50">
                {d.status === "failed"
                  ? d.error
                  : d.status === "ready"
                    ? `${d.page_count} pages · ${d.chunk_count} chunks`
                    : "Ingesting…"}
              </p>
            </div>
            <span className={`shrink-0 rounded px-1.5 py-0.5 text-xs ${STATUS_STYLE[d.status]}`}>
              {d.status}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
