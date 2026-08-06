import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import { CreateScenarioForm } from "./create-scenario-form";

vi.mock("@/lib/api-client", () => ({
  createScenario: vi.fn().mockResolvedValue({ id: "scn-1" }),
}));

import { createScenario } from "@/lib/api-client";

function renderWithQueryClient(ui: React.ReactElement) {
  const client = new QueryClient();
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

describe("CreateScenarioForm", () => {
  it("submits mode/penetration_level/units, including bid fields in arbitrage mode", async () => {
    const onCreated = vi.fn();
    renderWithQueryClient(<CreateScenarioForm onCreated={onCreated} />);

    fireEvent.change(screen.getByLabelText(/nivel de penetracion/i), {
      target: { value: "baseline" },
    });
    fireEvent.change(screen.getByLabelText(/^nombre$/i), { target: { value: "bess-1" } });
    fireEvent.change(screen.getByLabelText(/capacidad/i), { target: { value: "10" } });
    fireEvent.change(screen.getByLabelText(/oferta de carga/i), { target: { value: "50" } });
    fireEvent.change(screen.getByLabelText(/oferta de descarga/i), { target: { value: "200" } });
    fireEvent.click(screen.getByRole("button", { name: /crear escenario/i }));

    await waitFor(() =>
      expect(createScenario).toHaveBeenCalledWith(
        expect.objectContaining({
          mode: "arbitrage",
          penetration_level: "baseline",
          units: [expect.objectContaining({ name: "bess-1", mwh_nom: 10, charge_bid: 50, discharge_bid: 200 })],
        })
      )
    );
    await waitFor(() => expect(onCreated).toHaveBeenCalled());
  });

  it("hides bid fields when mode is grid_asset", () => {
    renderWithQueryClient(<CreateScenarioForm onCreated={vi.fn()} />);

    fireEvent.change(screen.getByLabelText(/^modo$/i), { target: { value: "grid_asset" } });

    expect(screen.queryByLabelText(/oferta de carga/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/oferta de descarga/i)).not.toBeInTheDocument();
  });

  it("adds a second unit row when 'Agregar unidad' is clicked", () => {
    renderWithQueryClient(<CreateScenarioForm onCreated={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: /agregar unidad/i }));

    expect(screen.getAllByLabelText(/^nombre$/i)).toHaveLength(2);
  });
});
