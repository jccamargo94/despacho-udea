import { describe, expect, it } from "vitest";
import { isTerminalStatus, statusLabel, statusBadgeVariant } from "./run-status";

describe("isTerminalStatus", () => {
  it("is false for pending and running", () => {
    expect(isTerminalStatus("pending")).toBe(false);
    expect(isTerminalStatus("running")).toBe(false);
  });

  it("is true for done and failed", () => {
    expect(isTerminalStatus("done")).toBe(true);
    expect(isTerminalStatus("failed")).toBe(true);
  });
});

describe("statusLabel", () => {
  it("returns the Spanish label for each status", () => {
    expect(statusLabel("pending")).toBe("Pendiente");
    expect(statusLabel("running")).toBe("Ejecutando");
    expect(statusLabel("done")).toBe("Completado");
    expect(statusLabel("failed")).toBe("Fallido");
  });
});

describe("statusBadgeVariant", () => {
  it("maps done to default and failed to destructive", () => {
    expect(statusBadgeVariant("done")).toBe("default");
    expect(statusBadgeVariant("failed")).toBe("destructive");
  });
});
