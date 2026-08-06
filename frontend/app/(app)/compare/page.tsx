"use client";

import { RunComparisonTable } from "@/components/run-comparison-table";
import { RunSelector } from "@/components/run-selector";
import { useRunComparisons } from "@/hooks/use-run-comparisons";
import { listRuns } from "@/lib/api-client";
import type { RunDetail } from "@/lib/types";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

export default function ComparePage() {
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const runsQuery = useQuery({ queryKey: ["runs"], queryFn: listRuns });
  const comparisons = useRunComparisons(selectedIds);

  function toggle(id: string) {
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }

  const loadedRuns: RunDetail[] = comparisons
    .map((c) => c.data)
    .filter((d): d is RunDetail => d !== undefined);

  return (
    <div>
      <h1>Comparador</h1>
      {runsQuery.isLoading && <p>Cargando ejecuciones...</p>}
      {runsQuery.data && <RunSelector runs={runsQuery.data} selectedIds={selectedIds} onToggle={toggle} />}
      {loadedRuns.length > 0 && <RunComparisonTable runs={loadedRuns} />}
    </div>
  );
}
