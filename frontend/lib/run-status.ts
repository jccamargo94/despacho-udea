import type { RunStatus } from "./types";

export function isTerminalStatus(status: RunStatus): boolean {
  return status === "done" || status === "failed";
}
