import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ScenariosTable } from "./scenarios-table";
import type { Scenario } from "@/lib/types";

const scenarios: Scenario[] = [
  {
    id: "scn-1",
    mode: "arbitrage",
    penetration_level: "baseline",
    units: [
      {
        name: "bess-1",
        mwh_nom: 10,
        hours_to_deplete: 4,
        initial_soc: 0.5,
        min_soc: 0.1,
        max_soc: 0.9,
        efficiency: 0.9,
        charge_bid: 50,
        discharge_bid: 200,
      },
    ],
    created_at: "2024-04-18T05:00:00Z",
  },
];

describe("ScenariosTable", () => {
  it("renders one row per scenario with mode/penetration/unit count", () => {
    render(<ScenariosTable scenarios={scenarios} />);
    expect(screen.getByText("baseline")).toBeInTheDocument();
    expect(screen.getByText("arbitrage")).toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();
  });

  it("renders an empty state with no scenarios", () => {
    render(<ScenariosTable scenarios={[]} />);
    expect(screen.getByText(/sin escenarios/i)).toBeInTheDocument();
  });
});
