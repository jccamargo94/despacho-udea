"use client";

import { CreateRunForm } from "@/components/create-run-form";
import { RunsTable } from "@/components/runs-table";
import { listRuns } from "@/lib/api-client";
import { useQuery, useQueryClient } from "@tanstack/react-query";

export default function RunsPage() {
  const queryClient = useQueryClient();
  const runsQuery = useQuery({ queryKey: ["runs"], queryFn: listRuns });

  return (
    <div>
      <h1>Ejecuciones</h1>
      <CreateRunForm onCreated={() => queryClient.invalidateQueries({ queryKey: ["runs"] })} />
      {runsQuery.isLoading && <p>Cargando...</p>}
      {runsQuery.data && <RunsTable runs={runsQuery.data} />}
    </div>
  );
}
