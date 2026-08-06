export type RunStatus = "pending" | "running" | "done" | "failed";
export type DispatchLevel = "preideal" | "ideal";

export interface RunSummary {
  run_id: string;
  status: RunStatus;
  dispatch_date: string;
  level: DispatchLevel;
  scenario_id: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
}

export interface RunMetrics {
  rmse: number | null;
  mae: number | null;
  bias: number | null;
  wape: number | null;
  smape: number | null;
  r2: number | null;
  bess_charge_mwh: number | null;
  bess_discharge_mwh: number | null;
  bess_avg_soc_mwh: number | null;
  bess_net_revenue: number | null;
}

export interface RunDetail extends RunSummary {
  metrics: RunMetrics | null;
}

export interface Scenario {
  id: string;
  mode: "arbitrage" | "grid_asset" | "generator";
  penetration_level: string;
  units: unknown[];
  created_at: string;
}

export interface CreateRunRequest {
  dispatch_date: string;
  level: DispatchLevel;
  solver?: string;
  compute_prices?: boolean;
  scenario_id?: string | null;
}
