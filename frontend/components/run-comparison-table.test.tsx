import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RunComparisonTable } from "./run-comparison-table";
import type { RunDetail } from "@/lib/types";

function makeRun(overrides: Partial<RunDetail>): RunDetail {
  return {
    run_id: "r1",
    status: "done",
    dispatch_date: "2024-04-18",
    level: "preideal",
    scenario_id: null,
    created_at: "2024-04-18T05:00:00Z",
    started_at: null,
    finished_at: null,
    error: null,
    metrics: null,
    artifacts: { dispatch: false, prices: false, bess: false },
    ...overrides,
  };
}

describe("RunComparisonTable", () => {
  it("renders one column per run with its metric values", () => {
    const runs = [
      makeRun({
        run_id: "r1",
        metrics: {
          rmse: 1.5,
          mae: 1.2,
          bias: 0.1,
          wape: 5.5,
          smape: 4.4,
          r2: 0.9,
          bess_charge_mwh: 10,
          bess_discharge_mwh: 9,
          bess_avg_soc_mwh: 5,
          bess_net_revenue: 1000,
        },
      }),
    ];
    render(<RunComparisonTable runs={runs} />);
    expect(screen.getByText("1.5")).toBeInTheDocument();
    expect(screen.getByText("1000")).toBeInTheDocument();
  });

  it("shows a no-metrics marker and dashes for a run with null metrics", () => {
    const runs = [makeRun({ run_id: "r2", metrics: null })];
    render(<RunComparisonTable runs={runs} />);
    expect(screen.getByText(/sin metricas/i)).toBeInTheDocument();
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });

  it("renders a dash for individual null metric fields within populated metrics", () => {
    const runs = [
      makeRun({
        run_id: "r3",
        metrics: {
          rmse: 1.5,
          mae: 1.2,
          bias: 0.1,
          wape: 5.5,
          smape: 4.4,
          r2: null,
          bess_charge_mwh: 10,
          bess_discharge_mwh: 9,
          bess_avg_soc_mwh: 5,
          bess_net_revenue: 1000,
        },
      }),
    ];
    render(<RunComparisonTable runs={runs} />);
    expect(screen.getByText("1.5")).toBeInTheDocument();
    expect(screen.getByText("1000")).toBeInTheDocument();
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });

  it("shows metrics from one run even when another run has null metrics", () => {
    const runs = [
      makeRun({
        run_id: "r1",
        metrics: {
          rmse: 1.5,
          mae: 1.2,
          bias: 0.1,
          wape: 5.5,
          smape: 4.4,
          r2: 0.9,
          bess_charge_mwh: 10,
          bess_discharge_mwh: 9,
          bess_avg_soc_mwh: 5,
          bess_net_revenue: 1000,
        },
      }),
      makeRun({
        run_id: "r2",
        metrics: null,
      }),
    ];
    render(<RunComparisonTable runs={runs} />);
    expect(screen.getByText("1.5")).toBeInTheDocument();
    expect(screen.getByText("1000")).toBeInTheDocument();
    expect(screen.getByText(/sin metricas/i)).toBeInTheDocument();
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });
});
