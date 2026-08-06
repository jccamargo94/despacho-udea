import { supabase } from "./supabase";
import type { CreateRunRequest, RunDetail, RunSummary, Scenario } from "./types";

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
