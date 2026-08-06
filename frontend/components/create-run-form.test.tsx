import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import { CreateRunForm } from "./create-run-form";

vi.mock("@/lib/api-client", () => ({
  createRun: vi.fn().mockResolvedValue({ run_id: "r1", status: "pending" }),
  listScenarios: vi.fn().mockResolvedValue([]),
}));

import { createRun } from "@/lib/api-client";

function renderWithQueryClient(ui: React.ReactElement) {
  const client = new QueryClient();
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

describe("CreateRunForm", () => {
  it("calls createRun with the form values on submit", async () => {
    const onCreated = vi.fn();
    renderWithQueryClient(<CreateRunForm onCreated={onCreated} />);

    fireEvent.change(screen.getByLabelText(/fecha/i), { target: { value: "2024-04-18" } });
    fireEvent.change(screen.getByLabelText(/nivel/i), { target: { value: "preideal" } });
    fireEvent.click(screen.getByRole("button", { name: /crear/i }));

    await waitFor(() =>
      expect(createRun).toHaveBeenCalledWith(
        expect.objectContaining({ dispatch_date: "2024-04-18", level: "preideal" })
      )
    );
    await waitFor(() => expect(onCreated).toHaveBeenCalled());
  });

  it("submits with the default solver (cbc)", async () => {
    renderWithQueryClient(<CreateRunForm onCreated={vi.fn()} />);

    fireEvent.change(screen.getByLabelText(/fecha/i), { target: { value: "2024-04-18" } });
    fireEvent.click(screen.getByRole("button", { name: /crear/i }));

    await waitFor(() =>
      expect(createRun).toHaveBeenCalledWith(expect.objectContaining({ solver: "cbc" }))
    );
  });

  it("renders the HiGHS solver option as disabled", () => {
    renderWithQueryClient(<CreateRunForm onCreated={vi.fn()} />);
    const highsOption = screen.getByRole("option", { name: /highs/i }) as HTMLOptionElement;
    expect(highsOption.disabled).toBe(true);
  });
});
