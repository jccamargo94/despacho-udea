"use client";

import { formatBogotaTime } from "@/lib/format-date";
import { useRunDetail } from "@/hooks/use-run-detail";
import { useParams } from "next/navigation";

export default function RunDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { data, isLoading } = useRunDetail(id);

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
    </div>
  );
}
