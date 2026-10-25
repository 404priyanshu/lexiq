// Server components may reach the API on an internal address; the browser uses the public one.
const SERVER_API_URL =
  process.env.API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
export const PUBLIC_API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Health = {
  status: "ok" | "degraded";
  checks: Record<string, string>;
};

export async function getHealth(): Promise<Health | null> {
  try {
    // /health returns 503 with a body when degraded, so don't gate on res.ok.
    const res = await fetch(`${SERVER_API_URL}/health`, { cache: "no-store" });
    return (await res.json()) as Health;
  } catch {
    return null;
  }
}

export type AppConfig = { read_only: boolean; examples: string[] };

const DEFAULT_CONFIG: AppConfig = { read_only: false, examples: [] };

/** Deployment settings from the API (the public demo is read-only and shows examples). */
export async function getConfig(): Promise<AppConfig> {
  try {
    const res = await fetch(`${PUBLIC_API_URL}/config`, { cache: "no-store" });
    return res.ok ? ((await res.json()) as AppConfig) : DEFAULT_CONFIG;
  } catch {
    return DEFAULT_CONFIG;
  }
}

export type DocumentStatus = "pending" | "processing" | "ready" | "failed";

export type DocumentItem = {
  id: string;
  title: string;
  mime_type: string;
  page_count: number | null;
  status: DocumentStatus;
  error: string | null;
  chunk_count: number;
  created_at: string;
};

export async function listDocuments(): Promise<DocumentItem[]> {
  const res = await fetch(`${PUBLIC_API_URL}/documents?limit=200`, { cache: "no-store" });
  if (!res.ok) throw new Error(`list failed: ${res.status}`);
  return (await res.json()).items;
}

export async function uploadDocument(file: File): Promise<DocumentItem> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(`${PUBLIC_API_URL}/documents`, { method: "POST", body });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? `upload failed: ${res.status}`);
  }
  return res.json();
}

export type Source = {
  n: number;
  chunk_id: number;
  document_id: string;
  document_title: string;
  page_start: number;
  page_end: number;
  relevance: number;
  content: string;
};

export type AnswerDone = {
  answer: string;
  abstained: boolean;
  abstain_reason: string | null;
  refused: boolean;
  truncated: boolean;
  citations: { cited: number[]; invalid: number[]; coverage: number } | null;
  model: string | null;
  usage: { input_tokens: number; output_tokens: number } | null;
  timings_ms: { ttft?: number; generation?: number; total?: number };
};

export type AnswerEvent =
  | { event: "sources"; data: { sources: Source[] } }
  | { event: "delta"; data: { text: string } }
  | { event: "notice"; data: { fallback: { from: string | null; to: string | null } } }
  | { event: "done"; data: AnswerDone }
  | { event: "error"; data: { message: string; retryable: boolean } };

/** POST /answer and yield its server-sent events as they arrive. */
export async function* streamAnswer(
  query: string,
  signal: AbortSignal,
): AsyncGenerator<AnswerEvent> {
  const res = await fetch(`${PUBLIC_API_URL}/answer`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query }),
    signal,
  });
  if (!res.ok || !res.body) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? `request failed: ${res.status}`);
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = "";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (event && data) yield { event, data: JSON.parse(data) } as AnswerEvent;
    }
  }
}
