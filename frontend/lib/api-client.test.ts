import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./supabase", () => ({
  supabase: { auth: { getSession: vi.fn() } },
}));

import { supabase } from "./supabase";
import { createRun, createScenario, downloadRunArtifact, getRunDispatch, getRunLog, listRuns } from "./api-client";

const fetchMock = vi.fn();
vi.stubGlobal("fetch", fetchMock);

beforeEach(() => {
  fetchMock.mockReset();
  vi.mocked(supabase.auth.getSession).mockResolvedValue({
    data: { session: { access_token: "tok-123" } },
  } as never);
});

describe("api-client", () => {
  it("listRuns sends the Authorization header and returns parsed JSON", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => [{ run_id: "r1" }],
    });

    const runs = await listRuns();

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/runs"),
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer tok-123" }),
      })
    );
    expect(runs).toEqual([{ run_id: "r1" }]);
  });

  it("createRun POSTs the body as JSON", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ run_id: "r2", status: "pending" }),
    });

    await createRun({ dispatch_date: "2024-04-18", level: "preideal" });

    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      dispatch_date: "2024-04-18",
      level: "preideal",
    });
  });

  it("createScenario POSTs the body as JSON and returns the id", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ id: "scn-1" }),
    });

    const result = await createScenario({
      mode: "arbitrage",
      penetration_level: "baseline",
      units: [
        {
          name: "bess-1",
          mwh_nom: 10,
          hours_to_deplete: 4,
          initial_soc: 0.5,
          min_soc: 0.1,
          max_soc: 0.9,
          efficiency: 0.9,
          charge_bid: 50,
          discharge_bid: 200,
        },
      ],
    });

    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe("POST");
    const body = JSON.parse(init.body);
    expect(body.mode).toBe("arbitrage");
    expect(body.units).toHaveLength(1);
    expect(result).toEqual({ id: "scn-1" });
  });

  it("throws with status and body text when the response is not ok", async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 404,
      statusText: "Not Found",
      text: async () => "run not found",
    });

    await expect(listRuns()).rejects.toThrow("404");
  });

  it("getRunDispatch fetches the dispatch artifact as JSON rows", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => [{ generador: "TERMO1", datetime: "2024-04-18 00:00:00", dispatch: 300 }],
    });

    const rows = await getRunDispatch("run-1");

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/runs/run-1/dispatch"),
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer tok-123" }),
      })
    );
    expect(rows).toEqual([{ generador: "TERMO1", datetime: "2024-04-18 00:00:00", dispatch: 300 }]);
  });

  it("downloadRunArtifact fetches with auth header and returns a Blob", async () => {
    const fakeBlob = new Blob(["csv,data"], { type: "text/csv" });
    fetchMock.mockResolvedValue({
      ok: true,
      blob: async () => fakeBlob,
    });

    const blob = await downloadRunArtifact("run-1", "dispatch");

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/runs/run-1/download/dispatch"),
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer tok-123" }),
      })
    );
    expect(blob).toBe(fakeBlob);
  });

  it("downloadRunArtifact throws when the response is not ok", async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 404, statusText: "Not Found" });

    await expect(downloadRunArtifact("run-1", "bess")).rejects.toThrow("404");
  });

  it("getRunLog returns the text body on success", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => "line 1\nline 2",
    });

    const log = await getRunLog("run-1");

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/runs/run-1/log"),
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer tok-123" }),
      })
    );
    expect(log).toBe("line 1\nline 2");
  });

  it("getRunLog returns null on 404 instead of throwing", async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 404, statusText: "Not Found" });

    const log = await getRunLog("run-1");

    expect(log).toBeNull();
  });
});
