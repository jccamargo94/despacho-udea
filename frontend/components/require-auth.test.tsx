import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RequireAuth } from "./require-auth";

const replace = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ replace }) }));

const useAuthMock = vi.fn();
vi.mock("@/lib/auth-context", () => ({ useAuth: () => useAuthMock() }));

describe("RequireAuth", () => {
  it("renders nothing and redirects to /login when there is no session", () => {
    useAuthMock.mockReturnValue({ session: null, loading: false });
    render(
      <RequireAuth>
        <div>secret</div>
      </RequireAuth>
    );
    expect(screen.queryByText("secret")).not.toBeInTheDocument();
    expect(replace).toHaveBeenCalledWith("/login");
  });

  it("renders children when there is a session", () => {
    useAuthMock.mockReturnValue({ session: { user: { id: "u1" } }, loading: false });
    render(
      <RequireAuth>
        <div>secret</div>
      </RequireAuth>
    );
    expect(screen.getByText("secret")).toBeInTheDocument();
  });
});
