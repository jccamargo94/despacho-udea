import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import LoginPage from "./page";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

vi.mock("@/lib/supabase", () => ({
  supabase: { auth: { signInWithPassword: vi.fn() } },
}));

import { supabase } from "@/lib/supabase";

describe("LoginPage", () => {
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
});
