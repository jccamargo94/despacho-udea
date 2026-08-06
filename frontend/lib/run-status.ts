import type { RunStatus } from "./types";

export function isTerminalStatus(status: RunStatus): boolean {
  return status === "done" || status === "failed";
}

const STATUS_LABELS: Record<RunStatus, string> = {
  pending: "Pendiente",
  running: "Ejecutando",
  done: "Completado",
  failed: "Fallido",
};

export function statusLabel(status: RunStatus): string {
  return STATUS_LABELS[status];
}

type BadgeVariant = "default" | "secondary" | "outline" | "destructive";

const STATUS_BADGE_VARIANT: Record<RunStatus, BadgeVariant> = {
  pending: "outline",
  running: "secondary",
  done: "default",
  failed: "destructive",
};

export function statusBadgeVariant(status: RunStatus): BadgeVariant {
  return STATUS_BADGE_VARIANT[status];
}
