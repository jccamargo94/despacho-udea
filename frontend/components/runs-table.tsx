import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { formatBogotaTime, formatDuration } from "@/lib/format-date";
import { statusBadgeVariant, statusLabel } from "@/lib/run-status";
import type { RunSummary } from "@/lib/types";
import Link from "next/link";

export function RunsTable({ runs }: { runs: RunSummary[] }) {
  if (runs.length === 0) {
    return <p className="p-6 text-sm text-muted-foreground">Sin ejecuciones todavia.</p>;
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Fecha</TableHead>
          <TableHead>Nivel</TableHead>
          <TableHead>Estado</TableHead>
          <TableHead>Creado</TableHead>
          <TableHead>Duracion</TableHead>
          <TableHead />
        </TableRow>
      </TableHeader>
      <TableBody>
        {runs.map((run) => (
          <TableRow key={run.run_id}>
            <TableCell className="font-mono">{run.dispatch_date}</TableCell>
            <TableCell>{run.level}</TableCell>
            <TableCell>
              <Badge variant={statusBadgeVariant(run.status)}>{statusLabel(run.status)}</Badge>
            </TableCell>
            <TableCell className="font-mono text-muted-foreground">
              {formatBogotaTime(run.created_at)}
            </TableCell>
            <TableCell className="font-mono text-muted-foreground">
              {formatDuration(run.started_at, run.finished_at)}
            </TableCell>
            <TableCell>
              <Link
                href={`/runs/${run.run_id}`}
                className="text-sm font-medium text-primary hover:underline"
              >
                Ver
              </Link>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
