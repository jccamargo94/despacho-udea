"use client";

import { ArtifactDownloads } from "@/components/artifact-downloads";
import { DispatchChart } from "@/components/dispatch-chart";
import { LogViewer } from "@/components/log-viewer";
import { formatBogotaTime } from "@/lib/format-date";
import { useRunDetail } from "@/hooks/use-run-detail";
import { getRunDispatch } from "@/lib/api-client";
import { useQuery } from "@tanstack/react-query";
import { useParams } from "next/navigation";

export default function RunDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { data, isLoading } = useRunDetail(id);

  const dispatchQuery = useQuery({
    queryKey: ["run-dispatch", id],
    queryFn: () => getRunDispatch(id),
    enabled: Boolean(data?.artifacts.dispatch),
  });

  if (isLoading || !data) return <p>Cargando...</p>;

  return (
    <div>
      <h1>Ejecucion {data.run_id}</h1>
      <p>Fecha: {data.dispatch_date}</p>
      <p>Nivel: {data.level}</p>
      <p>Status: {data.status}</p>
      <p>Creado: {formatBogotaTime(data.created_at)}</p>
      <p>Iniciado: {formatBogotaTime(data.started_at)}</p>
      <p>Terminado: {formatBogotaTime(data.finished_at)}</p>
      {data.status === "failed" && data.error && (
        <p role="alert">Error: {data.error}</p>
      )}
      {data.metrics && (
        <dl>
          <dt>RMSE</dt>
          <dd>{data.metrics.rmse}</dd>
          <dt>R2</dt>
          <dd>{data.metrics.r2}</dd>
        </dl>
      )}

      <h2>Despacho por generador</h2>
      <DispatchChart rows={dispatchQuery.data ?? []} />

      <h2>Artefactos</h2>
      <ArtifactDownloads runId={data.run_id} artifacts={data.artifacts} />

      <h2>Log</h2>
      <LogViewer runId={data.run_id} />
    </div>
  );
}
