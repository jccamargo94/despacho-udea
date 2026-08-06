import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import LoginPage from "./page";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

vi.mock("@/lib/supabase", () => ({
  supabase: { auth: { signInWithPassword: vi.fn(), resetPasswordForEmail: vi.fn() } },
}));

import { supabase } from "@/lib/supabase";

describe("LoginPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("signs in and redirects to /runs on success", async () => {
    vi.mocked(supabase.auth.signInWithPassword).mockResolvedValue({ error: null } as never);

    render(<LoginPage />);
    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: "a@b.com" } });
    fireEvent.change(screen.getByLabelText(/contrase/i), { target: { value: "secret123" } });
    fireEvent.click(screen.getByRole("button", { name: /entrar/i }));

    await waitFor(() => expect(push).toHaveBeenCalledWith("/runs"));
  });

  it("shows the error message on failure", async () => {
    vi.mocked(supabase.auth.signInWithPassword).mockResolvedValue({
      error: { message: "Invalid login credentials" },
    } as never);

    render(<LoginPage />);
    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: "a@b.com" } });
    fireEvent.change(screen.getByLabelText(/contrase/i), { target: { value: "wrong" } });
    fireEvent.click(screen.getByRole("button", { name: /entrar/i }));

    await waitFor(() => screen.getByText("Invalid login credentials"));
  });

  it("sends a password reset email when the forgot-password link is clicked", async () => {
    vi.mocked(supabase.auth.resetPasswordForEmail).mockResolvedValue({
      data: {},
      error: null,
    } as never);

    render(<LoginPage />);
    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: "a@b.com" } });
    fireEvent.click(screen.getByRole("button", { name: /olvidaste tu contrasena/i }));

    await waitFor(() =>
      expect(supabase.auth.resetPasswordForEmail).toHaveBeenCalledWith(
        "a@b.com",
        expect.objectContaining({ redirectTo: expect.stringContaining("/reset-password") })
      )
    );
    await waitFor(() => screen.getByText(/revisa tu correo/i));
  });

  it("shows a validation message when requesting a reset with no email", async () => {
    render(<LoginPage />);
    fireEvent.click(screen.getByRole("button", { name: /olvidaste tu contrasena/i }));

    await waitFor(() => screen.getByText(/ingresa tu correo/i));
    expect(supabase.auth.resetPasswordForEmail).not.toHaveBeenCalled();
  });

  it("links to /signup", () => {
    render(<LoginPage />);
    expect(screen.getByRole("link", { name: /crear cuenta/i })).toHaveAttribute("href", "/signup");
  });
});
