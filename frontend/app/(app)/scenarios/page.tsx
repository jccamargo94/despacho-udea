"use client";

import { CreateScenarioForm } from "@/components/create-scenario-form";
import { ScenariosTable } from "@/components/scenarios-table";
import { listScenarios } from "@/lib/api-client";
import { useQuery, useQueryClient } from "@tanstack/react-query";

export default function ScenariosPage() {
  const queryClient = useQueryClient();
  const scenariosQuery = useQuery({ queryKey: ["scenarios"], queryFn: listScenarios });

  return (
    <div>
      <h1>Escenarios BESS</h1>
      <CreateScenarioForm
        onCreated={() => queryClient.invalidateQueries({ queryKey: ["scenarios"] })}
      />
      {scenariosQuery.isLoading && <p>Cargando...</p>}
      {scenariosQuery.data && <ScenariosTable scenarios={scenariosQuery.data} />}
    </div>
  );
}
