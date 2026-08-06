import type { RunDetail, RunMetrics } from "@/lib/types";

const METRIC_ROWS: { key: keyof RunMetrics; label: string }[] = [
  { key: "rmse", label: "RMSE" },
  { key: "mae", label: "MAE" },
  { key: "bias", label: "Bias" },
  { key: "wape", label: "WAPE" },
  { key: "smape", label: "sMAPE" },
  { key: "r2", label: "R2" },
  { key: "bess_charge_mwh", label: "BESS carga (MWh)" },
  { key: "bess_discharge_mwh", label: "BESS descarga (MWh)" },
  { key: "bess_avg_soc_mwh", label: "BESS SOC promedio (MWh)" },
  { key: "bess_net_revenue", label: "BESS ingreso neto" },
];

export function RunComparisonTable({ runs }: { runs: RunDetail[] }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Metrica</th>
          {runs.map((run) => (
            <th key={run.run_id}>
              {run.dispatch_date} ({run.level})
              {run.metrics === null && " (sin metricas)"}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {METRIC_ROWS.map((row) => (
          <tr key={row.key}>
            <td>{row.label}</td>
            {runs.map((run) => (
              <td key={run.run_id}>{run.metrics ? (run.metrics[row.key] ?? "—") : "—"}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
