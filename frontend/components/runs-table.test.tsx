import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RunsTable } from "./runs-table";
import type { RunSummary } from "@/lib/types";

const runs: RunSummary[] = [
  {
    run_id: "r1",
    status: "done",
    dispatch_date: "2024-04-18",
    level: "preideal",
    scenario_id: null,
    created_at: "2024-04-18T05:00:00Z",
    started_at: null,
    finished_at: null,
    error: null,
  },
];

describe("RunsTable", () => {
  it("renders one row per run with date/level/status", () => {
    render(<RunsTable runs={runs} />);
    expect(screen.getByText("2024-04-18")).toBeInTheDocument();
    expect(screen.getByText("preideal")).toBeInTheDocument();
    expect(screen.getByText("done")).toBeInTheDocument();
  });

  it("renders an empty state with no runs", () => {
    render(<RunsTable runs={[]} />);
    expect(screen.getByText(/sin ejecuciones/i)).toBeInTheDocument();
  });
});
