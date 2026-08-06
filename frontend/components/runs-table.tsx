import { formatBogotaTime } from "@/lib/format-date";
import type { RunSummary } from "@/lib/types";
import Link from "next/link";

export function RunsTable({ runs }: { runs: RunSummary[] }) {
  if (runs.length === 0) return <p>Sin ejecuciones todavia.</p>;

  return (
    <table>
      <thead>
        <tr>
          <th>Fecha</th>
          <th>Nivel</th>
          <th>Status</th>
          <th>Creado</th>
        </tr>
      </thead>
      <tbody>
        {runs.map((run) => (
          <tr key={run.run_id}>
            <td>{run.dispatch_date}</td>
            <td>{run.level}</td>
            <td>{run.status}</td>
            <td>{formatBogotaTime(run.created_at)}</td>
            <td>
              <Link href={`/runs/${run.run_id}`}>Ver</Link>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
