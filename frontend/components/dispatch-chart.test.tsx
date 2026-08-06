import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DispatchChart } from "./dispatch-chart";
import type { DispatchRow } from "@/lib/types";

describe("DispatchChart", () => {
  it("renders a chart when there are rows", () => {
    const rows: DispatchRow[] = [
      { generador: "A", datetime: "2024-04-18 00:00:00", dispatch: 10 },
      { generador: "A", datetime: "2024-04-18 01:00:00", dispatch: 20 },
    ];

    const { container } = render(<DispatchChart rows={rows} />);

    expect(container.querySelector(".recharts-wrapper, svg")).toBeTruthy();
  });

  it("shows an empty-state message instead of a chart when there are no rows", () => {
    render(<DispatchChart rows={[]} />);

    expect(screen.getByText(/no hay datos de despacho/i)).toBeInTheDocument();
  });
});
