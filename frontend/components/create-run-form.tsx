"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { createRun, listScenarios } from "@/lib/api-client";
import type { CreateRunRequest, DispatchLevel } from "@/lib/types";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";

const SELECT_CLASS =
  "h-8 rounded-lg border border-input bg-transparent px-2.5 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50";

export function CreateRunForm({ onCreated }: { onCreated: () => void }) {
  const [dispatchDate, setDispatchDate] = useState("");
  const [level, setLevel] = useState<DispatchLevel>("preideal");
  const [solver, setSolver] = useState("cbc");
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
      solver,
      scenario_id: scenarioId || null,
    });
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-wrap items-end gap-4">
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="dispatch_date">Fecha</Label>
        <Input
          id="dispatch_date"
          type="date"
          value={dispatchDate}
          onChange={(e) => setDispatchDate(e.target.value)}
          required
        />
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="level">Nivel</Label>
        <select
          id="level"
          value={level}
          onChange={(e) => setLevel(e.target.value as DispatchLevel)}
          className={SELECT_CLASS}
        >
          <option value="preideal">preideal</option>
          <option value="ideal">ideal</option>
        </select>
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="solver">Solver</Label>
        <select
          id="solver"
          value={solver}
          onChange={(e) => setSolver(e.target.value)}
          className={SELECT_CLASS}
        >
          <option value="cbc">CBC</option>
          <option value="highs" disabled>
            HiGHS (proximamente)
          </option>
        </select>
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="scenario_id">Escenario BESS (opcional)</Label>
        <select
          id="scenario_id"
          value={scenarioId}
          onChange={(e) => setScenarioId(e.target.value)}
          className={SELECT_CLASS}
        >
          <option value="">Ninguno</option>
          {(scenariosQuery.data ?? []).map((s) => (
            <option key={s.id} value={s.id}>
              {s.penetration_level} ({s.mode})
            </option>
          ))}
        </select>
      </div>
      <Button type="submit" disabled={mutation.isPending}>
        Crear ejecucion
      </Button>
      {mutation.isError && (
        <p role="alert" className="w-full text-sm text-destructive">
          {(mutation.error as Error).message}
        </p>
      )}
    </form>
  );
}
