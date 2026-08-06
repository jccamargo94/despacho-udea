import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AuthProvider, useAuth } from "./auth-context";

vi.mock("./supabase", () => ({
  supabase: {
    auth: {
      getSession: vi.fn(),
      onAuthStateChange: vi.fn(() => ({ data: { subscription: { unsubscribe: vi.fn() } } })),
    },
  },
}));

import { supabase } from "./supabase";

function Probe() {
  const { session, loading } = useAuth();
  if (loading) return <div>loading</div>;
  return <div>{session ? `logged in as ${session.user.id}` : "logged out"}</div>;
}

describe("AuthProvider/useAuth", () => {
  beforeEach(() => {
    vi.mocked(supabase.auth.getSession).mockReset();
  });

  it("exposes the session once getSession resolves", async () => {
    vi.mocked(supabase.auth.getSession).mockResolvedValue({
      data: { session: { user: { id: "user-1" } } },
    } as never);

    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>
    );

    await waitFor(() => screen.getByText("logged in as user-1"));
  });

  it("shows logged out when there is no session", async () => {
    vi.mocked(supabase.auth.getSession).mockResolvedValue({
      data: { session: null },
    } as never);

    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>
    );

    await waitFor(() => screen.getByText("logged out"));
  });
});
