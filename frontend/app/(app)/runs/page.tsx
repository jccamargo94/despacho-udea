"use client";

import { CreateRunForm } from "@/components/create-run-form";
import { RunsTable } from "@/components/runs-table";
import { listRuns } from "@/lib/api-client";
import { useQuery, useQueryClient } from "@tanstack/react-query";

export default function RunsPage() {
  const queryClient = useQueryClient();
  const runsQuery = useQuery({ queryKey: ["runs"], queryFn: listRuns });

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="font-heading text-2xl font-bold">Ejecuciones</h1>
        <p className="text-sm text-muted-foreground">Historial de modelado de despacho</p>
      </div>
      <div className="rounded-xl border border-border bg-card p-6">
        <CreateRunForm onCreated={() => queryClient.invalidateQueries({ queryKey: ["runs"] })} />
      </div>
      <div className="rounded-xl border border-border bg-card">
        {runsQuery.isLoading && <p className="p-6 text-sm text-muted-foreground">Cargando...</p>}
        {runsQuery.data && <RunsTable runs={runsQuery.data} />}
      </div>
    </div>
  );
}
