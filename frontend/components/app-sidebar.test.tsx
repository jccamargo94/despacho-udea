import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AppSidebar } from "./app-sidebar";

vi.mock("next/navigation", () => ({ usePathname: () => "/runs" }));

const signOut = vi.fn();
vi.mock("@/lib/auth-context", () => ({ useAuth: () => ({ signOut }) }));

describe("AppSidebar", () => {
  it("renders the three nav links pointing at the right routes", () => {
    render(<AppSidebar />);
    expect(screen.getByRole("link", { name: /ejecuciones/i })).toHaveAttribute("href", "/runs");
    expect(screen.getByRole("link", { name: /escenarios/i })).toHaveAttribute("href", "/scenarios");
    expect(screen.getByRole("link", { name: /comparar/i })).toHaveAttribute("href", "/compare");
  });

  it("marks the current route active", () => {
    render(<AppSidebar />);
    expect(screen.getByRole("link", { name: /ejecuciones/i })).toHaveClass("bg-primary");
    expect(screen.getByRole("link", { name: /escenarios/i })).not.toHaveClass("bg-primary");
  });

  it("calls signOut when Salir is clicked", () => {
    render(<AppSidebar />);
    screen.getByRole("button", { name: /salir/i }).click();
    expect(signOut).toHaveBeenCalled();
  });
});
