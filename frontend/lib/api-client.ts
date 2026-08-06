import { supabase } from "./supabase";
import type { CreateRunRequest, CreateScenarioRequest, DispatchRow, RunDetail, RunSummary, Scenario } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function authHeader(): Promise<Record<string, string>> {
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  if (!token) throw new Error("not authenticated");
  return { Authorization: `Bearer ${token}` };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    ...(await authHeader()),
    "Content-Type": "application/json",
    ...((init?.headers as Record<string, string>) ?? {}),
  };
  const resp = await fetch(`${API_BASE_URL}${path}`, { ...init, headers });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`${resp.status} ${resp.statusText}: ${body}`);
  }
  return resp.json() as Promise<T>;
}

export function listRuns(): Promise<RunSummary[]> {
  return request<RunSummary[]>("/runs");
}

export function getRun(id: string): Promise<RunDetail> {
  return request<RunDetail>(`/runs/${id}`);
}

export function createRun(body: CreateRunRequest): Promise<{ run_id: string; status: string }> {
  return request("/runs", { method: "POST", body: JSON.stringify(body) });
}

export function listScenarios(): Promise<Scenario[]> {
  return request<Scenario[]>("/scenarios");
}

export function createScenario(body: CreateScenarioRequest): Promise<{ id: string }> {
  return request("/scenarios", { method: "POST", body: JSON.stringify(body) });
}

export function getRunDispatch(id: string): Promise<DispatchRow[]> {
  return request<DispatchRow[]>(`/runs/${id}/dispatch`);
}

export async function downloadRunArtifact(
  id: string,
  artifact: "dispatch" | "prices" | "bess"
): Promise<Blob> {
  const headers = await authHeader();
  const resp = await fetch(`${API_BASE_URL}/runs/${id}/download/${artifact}`, { headers });
  if (!resp.ok) {
    throw new Error(`${resp.status} ${resp.statusText}`);
  }
  return resp.blob();
}

export async function getRunLog(id: string): Promise<string | null> {
  const headers = await authHeader();
  const resp = await fetch(`${API_BASE_URL}/runs/${id}/log`, { headers });
  if (resp.status === 404) {
    return null;
  }
  if (!resp.ok) {
    throw new Error(`${resp.status} ${resp.statusText}`);
  }
  return resp.text();
}
