import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ResetPasswordPage from "./page";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

vi.mock("@/lib/supabase", () => ({
  supabase: { auth: { updateUser: vi.fn() } },
}));

import { supabase } from "@/lib/supabase";

describe("ResetPasswordPage", () => {
  beforeEach(() => vi.clearAllMocks());

  it("updates the password and redirects to /login on success", async () => {
    vi.mocked(supabase.auth.updateUser).mockResolvedValue({ error: null } as never);

    render(<ResetPasswordPage />);
    fireEvent.change(screen.getByLabelText(/nueva contrasena/i), { target: { value: "secret123" } });
    fireEvent.change(screen.getByLabelText(/confirmar contrasena/i), { target: { value: "secret123" } });
    fireEvent.click(screen.getByRole("button", { name: /guardar/i }));

    await waitFor(() => expect(supabase.auth.updateUser).toHaveBeenCalledWith({ password: "secret123" }));
    await waitFor(() => expect(push).toHaveBeenCalledWith("/login"));
  });

  it("shows an error when the passwords do not match", async () => {
    render(<ResetPasswordPage />);
    fireEvent.change(screen.getByLabelText(/nueva contrasena/i), { target: { value: "secret123" } });
    fireEvent.change(screen.getByLabelText(/confirmar contrasena/i), { target: { value: "different" } });
    fireEvent.click(screen.getByRole("button", { name: /guardar/i }));

    await waitFor(() => screen.getByText(/no coinciden/i));
    expect(supabase.auth.updateUser).not.toHaveBeenCalled();
  });

  it("shows the server error message on failure", async () => {
    vi.mocked(supabase.auth.updateUser).mockResolvedValue({
      error: { message: "Auth session missing" },
    } as never);

    render(<ResetPasswordPage />);
    fireEvent.change(screen.getByLabelText(/nueva contrasena/i), { target: { value: "secret123" } });
    fireEvent.change(screen.getByLabelText(/confirmar contrasena/i), { target: { value: "secret123" } });
    fireEvent.click(screen.getByRole("button", { name: /guardar/i }));

    await waitFor(() => screen.getByText("Auth session missing"));
  });
});
