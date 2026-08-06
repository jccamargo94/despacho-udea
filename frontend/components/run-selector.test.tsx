import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RunSelector } from "./run-selector";
import type { RunSummary } from "@/lib/types";

function makeRun(overrides: Partial<RunSummary>): RunSummary {
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
    ...overrides,
  };
}

describe("RunSelector", () => {
  it("only renders checkboxes for done runs", () => {
    const runs = [makeRun({ run_id: "r1", status: "done" }), makeRun({ run_id: "r2", status: "pending" })];
    render(<RunSelector runs={runs} selectedIds={[]} onToggle={vi.fn()} />);
    expect(screen.getAllByRole("checkbox")).toHaveLength(1);
  });

  it("calls onToggle with the run id when its checkbox is clicked", () => {
    const onToggle = vi.fn();
    const runs = [makeRun({ run_id: "r1" })];
    render(<RunSelector runs={runs} selectedIds={[]} onToggle={onToggle} />);
    fireEvent.click(screen.getByRole("checkbox"));
    expect(onToggle).toHaveBeenCalledWith("r1");
  });

  it("checks the box for a run id already in selectedIds", () => {
    const runs = [makeRun({ run_id: "r1" })];
    render(<RunSelector runs={runs} selectedIds={["r1"]} onToggle={vi.fn()} />);
    expect(screen.getByRole("checkbox")).toBeChecked();
  });

  it("renders a message when there are no done runs", () => {
    const runs = [makeRun({ run_id: "r1", status: "pending" })];
    render(<RunSelector runs={runs} selectedIds={[]} onToggle={vi.fn()} />);
    expect(screen.getByText(/no hay ejecuciones completas/i)).toBeInTheDocument();
  });
});
