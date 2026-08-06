import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SignupPage from "./page";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

vi.mock("@/lib/supabase", () => ({
  supabase: { auth: { signUp: vi.fn() } },
}));

import { supabase } from "@/lib/supabase";

describe("SignupPage", () => {
  beforeEach(() => vi.clearAllMocks());
  it("signs up and redirects to /login on success", async () => {
    vi.mocked(supabase.auth.signUp).mockResolvedValue({ error: null } as never);

    render(<SignupPage />);
    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: "a@b.com" } });
    fireEvent.change(screen.getByLabelText("Contrasena"), { target: { value: "secret123" } });
    fireEvent.change(screen.getByLabelText(/confirmar contrasena/i), { target: { value: "secret123" } });
    fireEvent.click(screen.getByRole("button", { name: /crear cuenta/i }));

    await waitFor(() => expect(push).toHaveBeenCalledWith("/login"));
  });

  it("shows an error when the passwords do not match", async () => {
    render(<SignupPage />);
    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: "a@b.com" } });
    fireEvent.change(screen.getByLabelText("Contrasena"), { target: { value: "secret123" } });
    fireEvent.change(screen.getByLabelText(/confirmar contrasena/i), { target: { value: "different" } });
    fireEvent.click(screen.getByRole("button", { name: /crear cuenta/i }));

    await waitFor(() => screen.getByText(/no coinciden/i));
    expect(supabase.auth.signUp).not.toHaveBeenCalled();
  });

  it("links to /login", () => {
    render(<SignupPage />);
    expect(screen.getByRole("link", { name: /iniciar sesion/i })).toHaveAttribute("href", "/login");
  });
});
