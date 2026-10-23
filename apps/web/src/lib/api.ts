const API_URL =
  process.env.API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Health = {
  status: "ok" | "degraded";
  checks: Record<string, string>;
};

export async function getHealth(): Promise<Health | null> {
  try {
    // /health returns 503 with a body when degraded, so don't gate on res.ok.
    const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
    return (await res.json()) as Health;
  } catch {
    return null;
  }
}
