"use client";

import type { RunSummary } from "@/lib/types";

export function RunSelector({
  runs,
  selectedIds,
  onToggle,
}: {
  runs: RunSummary[];
  selectedIds: string[];
  onToggle: (id: string) => void;
}) {
  const doneRuns = runs.filter((r) => r.status === "done");

  if (doneRuns.length === 0) return <p>No hay ejecuciones completas para comparar.</p>;

  return (
    <fieldset>
      <legend>Seleccionar ejecuciones</legend>
      {doneRuns.map((run) => (
        <label key={run.run_id}>
          <input
            type="checkbox"
            checked={selectedIds.includes(run.run_id)}
            onChange={() => onToggle(run.run_id)}
          />
          {run.dispatch_date} ({run.level})
        </label>
      ))}
    </fieldset>
  );
}
