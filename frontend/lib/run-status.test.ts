import { describe, expect, it } from "vitest";
import { isTerminalStatus } from "./run-status";

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
