"use client";

import { useRunLog } from "@/hooks/use-run-log";

export function LogViewer({ runId }: { runId: string }) {
  const { data, isLoading } = useRunLog(runId);

  if (isLoading) return <p>Cargando log...</p>;
  if (!data) return <p>Sin logs todavia.</p>;

  return <pre style={{ maxHeight: "24rem", overflow: "auto" }}>{data}</pre>;
}
