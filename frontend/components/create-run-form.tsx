"use client";

import { createRun, listScenarios } from "@/lib/api-client";
import type { CreateRunRequest, DispatchLevel } from "@/lib/types";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";

export function CreateRunForm({ onCreated }: { onCreated: () => void }) {
  const [dispatchDate, setDispatchDate] = useState("");
  const [level, setLevel] = useState<DispatchLevel>("preideal");
  const [scenarioId, setScenarioId] = useState("");

  const scenariosQuery = useQuery({ queryKey: ["scenarios"], queryFn: listScenarios });
  const mutation = useMutation({
    mutationFn: (variables: CreateRunRequest) => createRun(variables),
    onSuccess: onCreated,
  });

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    mutation.mutate({
      dispatch_date: dispatchDate,
      level,
      scenario_id: scenarioId || null,
    });
  }

  return (
    <form onSubmit={handleSubmit}>
      <label htmlFor="dispatch_date">Fecha</label>
      <input
        id="dispatch_date"
        type="date"
        value={dispatchDate}
        onChange={(e) => setDispatchDate(e.target.value)}
        required
      />
      <label htmlFor="level">Nivel</label>
      <select
        id="level"
        value={level}
        onChange={(e) => setLevel(e.target.value as DispatchLevel)}
      >
        <option value="preideal">preideal</option>
        <option value="ideal">ideal</option>
      </select>
      <label htmlFor="scenario_id">Escenario BESS (opcional)</label>
      <select id="scenario_id" value={scenarioId} onChange={(e) => setScenarioId(e.target.value)}>
        <option value="">Ninguno</option>
        {(scenariosQuery.data ?? []).map((s) => (
          <option key={s.id} value={s.id}>
            {s.penetration_level} ({s.mode})
          </option>
        ))}
      </select>
      <button type="submit" disabled={mutation.isPending}>
        Crear ejecucion
      </button>
      {mutation.isError && <p role="alert">{(mutation.error as Error).message}</p>}
    </form>
  );
}
