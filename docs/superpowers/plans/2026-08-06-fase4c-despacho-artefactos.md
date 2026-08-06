# Fase 4c: Despacho + Explorador de artefactos/logs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing run-detail page with three sections: a dispatch-by-generator chart, a downloadable-artifacts list, and a log viewer — closing the last two roadmap bullets of Fase 4.

**Architecture:** One backend field addition (`artifacts` on `GET /runs/{id}`) unblocks the frontend from having to probe. Three new frontend components (`DispatchChart`, `ArtifactDownloads`, `LogViewer`) render into the existing `frontend/app/(app)/runs/[id]/page.tsx` — no new routes.

**Tech Stack:** Same as fase4a/4b — Next.js App Router, TypeScript, TanStack Query, Vitest + React Testing Library, pytest (backend). New dependency: `recharts` (was assumed present since Fase 4's original stack decision, but never actually installed — no chart existed until now).

## Global Constraints

- Backend: FastAPI (`services/api/main.py`), SQLAlchemy models already have `dispatch_path`/`price_path`/`bess_path` on `Run` (`app/db/models.py`) — this plan reads them, never writes them (worker already sets them via `queries.finish_run_ok`).
- Frontend commands via `pnpm` inside `frontend/`. Backend commands via `pytest` from repo root (repo uses `uv` — if a task needs to install/run, use `uv run pytest`).
- Follow TDD: failing test first, confirm fail, implement, confirm pass.
- **"por recurso/tecnologia" = "por generador".** `extract_dispatch` (`app/pipeline/results.py:24-28`) has no technology field — the chart groups by `generador`, nothing else exists to group by.
- **Downloads MUST NOT be a plain `<a href=...>`.** `GET /runs/{id}/download/{artifact}` requires `Authorization: Bearer <jwt>` (`services/api/main.py:197-206`). Auth session lives in `localStorage` only (no cookies) — a bare anchor navigation sends no auth header and gets a 401. The only correct pattern: authenticated `fetch` → `res.blob()` → `URL.createObjectURL(blob)` → synthetic `<a>` click → `URL.revokeObjectURL(url)`.
- **The log endpoint returns plain text, not JSON.** `GET /runs/{id}/log` (`services/api/main.py:153-162`) is `PlainTextResponse` and 404s when `run.log_path is None`. The existing `request<T>()` helper in `frontend/lib/api-client.ts` always calls `resp.json()` — it cannot be reused as-is for this endpoint. A 404 here is a normal, expected state (a `pending`/`running` run has no log yet), not an error — it must render as "sin logs todavia", never as a thrown error or alert.
- **`mutationFn` in TanStack Query v5 is called with `(variables, context)`, not just `(variables)`.** Any `useMutation` in this plan must wrap the API call: `mutationFn: (variables: X) => someFunction(variables)` — never pass the API function directly as `mutationFn`.
- **No shadcn `Select`.** It wraps a headless popover component that's fragile in tests. Every dropdown in this repo uses a native `<select>` — not touched by this plan, noted only because no task here should introduce one either.
- Testing: Vitest + React Testing Library, same file-naming pattern as fase4a/4b (`Component.tsx` next to `Component.test.tsx`, `useSomething.ts` next to `useSomething.test.tsx`). No Playwright/e2e.
- A dataviz-specific skill was not available in this environment when this plan was written. Task 3 embeds explicit charting guidance (top-N + "Otros" bucket, stacked area, legend, tooltip) directly in its steps instead of delegating to a skill.

---

### Task 1: Backend — `artifacts` field on `GET /runs/{id}`

**Files:**
- Modify: `services/api/main.py:79-149` (`_run_summary`, `get_run_detail`)
- Test: `tests/test_api_runs.py`

**Interfaces:**
- Produces: `GET /runs/{id}` response gains `"artifacts": {"dispatch": bool, "prices": bool, "bess": bool}`, computed from whether `run.dispatch_path`/`run.price_path`/`run.bess_path` are non-`None`. This is a NEW top-level key alongside the existing `metrics` key (not nested inside it).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_api_runs.py` (append at the end of the file):

```python
def test_get_run_artifacts_all_false_before_worker_runs(api_client):
    resp = api_client.post("/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"})
    run_id = resp.json()["run_id"]

    resp = api_client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["artifacts"] == {"dispatch": False, "prices": False, "bess": False}


def test_get_run_artifacts_reflects_available_paths(api_client, tmp_path):
    from datetime import date

    from app.db import queries
    from app.schemas import DispatchCase, DispatchLevel, RunResult

    resp = api_client.post("/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"})
    run_id = resp.json()["run_id"]

    out_dir = tmp_path / "results" / run_id
    out_dir.mkdir(parents=True)

    session = api_client.SessionLocal()
    run = queries.get_run(session, run_id)
    result = RunResult(
        case=DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal),
        ok=True,
        dispatch_path=str(out_dir / "dispatch.csv"),
        price_path=str(out_dir / "price.csv"),
    )
    queries.finish_run_ok(session, run, result, out_dir=str(out_dir))
    session.close()

    resp = api_client.get(f"/runs/{run_id}")
    assert resp.json()["artifacts"] == {"dispatch": True, "prices": True, "bess": False}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api_runs.py -v -k artifacts`
Expected: FAIL with `KeyError: 'artifacts'` (the field doesn't exist yet).

- [ ] **Step 3: Implement**

In `services/api/main.py`, modify `get_run_detail` (around line 127-149) to add the `artifacts` key to `out` after it's built. Read the function first to see the exact current structure of `out` before editing — do not guess line numbers, the file may have shifted slightly since this plan was written. Add this logic right after `out = _run_summary(run, case)` and before the function's `return out`:

```python
    out["artifacts"] = {
        "dispatch": run.dispatch_path is not None,
        "prices": run.price_path is not None,
        "bess": run.bess_path is not None,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api_runs.py -v -k artifacts`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full backend suite to check for regressions**

Run: `uv run pytest`
Expected: all tests pass (no test asserts the exact full shape of `get_run_detail`'s response dict in a way that a new key would break — `test_get_run_includes_bess_metrics` and similar only check specific keys via `resp.json()["metrics"][...]`, not full-dict equality).

- [ ] **Step 6: Commit**

```bash
git add services/api/main.py tests/test_api_runs.py
git commit -m "feat(api): expose artifacts availability on GET /runs/{id}"
```

---

### Task 2: Frontend — types, api-client functions, recharts dependency

**Files:**
- Modify: `frontend/lib/types.ts` (add `DispatchRow`, add `artifacts` to `RunDetail`)
- Modify: `frontend/lib/api-client.ts` (add `getRunDispatch`, `downloadRunArtifact`, `getRunLog`)
- Modify: `frontend/lib/api-client.test.ts` (add tests for the 3 new functions)
- Modify: `frontend/components/run-comparison-table.test.tsx` (its `makeRun` helper builds a full `RunDetail` literal — adding a required field to the type breaks it without this fix)
- Modify: `frontend/package.json` (add `recharts` dependency)

**Interfaces:**
- Consumes: existing `request<T>()` helper and `authHeader()` in `frontend/lib/api-client.ts` (both already private to that file — `authHeader` stays private, reused internally by the two new non-JSON functions).
- Produces (for Tasks 3-6 to consume):
  - `interface DispatchRow { generador: string; datetime: string; dispatch: number }`
  - `RunDetail.artifacts: { dispatch: boolean; prices: boolean; bess: boolean }` (required field, always present since the backend always includes it)
  - `getRunDispatch(id: string): Promise<DispatchRow[]>`
  - `downloadRunArtifact(id: string, artifact: "dispatch" | "prices" | "bess"): Promise<Blob>`
  - `getRunLog(id: string): Promise<string | null>` — resolves `null` on a 404 (no log yet), never throws for that case.

- [ ] **Step 1: Add `DispatchRow` type and `artifacts` field to `RunDetail`**

In `frontend/lib/types.ts`, add this new interface near `RunMetrics`:

```ts
export interface DispatchRow {
  generador: string;
  datetime: string;
  dispatch: number;
}

export interface RunArtifacts {
  dispatch: boolean;
  prices: boolean;
  bess: boolean;
}
```

Then modify the existing `RunDetail` interface to add the new field:

```ts
export interface RunDetail extends RunSummary {
  metrics: RunMetrics | null;
  artifacts: RunArtifacts;
}
```

- [ ] **Step 2: Fix the one existing test fixture that breaks**

`frontend/components/run-comparison-table.test.tsx` has a `makeRun` helper that returns a full `RunDetail` object literal (not a partial/cast). Open it and add `artifacts: { dispatch: false, prices: false, bess: false }` to the object it returns, alongside the existing `metrics: null` line. This is the only test file in the repo with a full, uncast `RunDetail` literal — every other file either uses `RunSummary` (unaffected, no `artifacts` field on that type) or casts with `as never`.

- [ ] **Step 3: Write the failing tests for the new api-client functions**

Add to `frontend/lib/api-client.test.ts`, inside the existing `describe("api-client", ...)` block (add these `it` blocks, and add `getRunDispatch`, `downloadRunArtifact`, `getRunLog` to the existing top-of-file import from `"./api-client"`):

```ts
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
```

Note: the "throws when the response is not ok" test for `downloadRunArtifact` and this "returns null on 404" test for `getRunLog` use the identical mock shape (`{ ok: false, status: 404, statusText: "Not Found" }`) but assert opposite outcomes. That is intentional, not a copy-paste error: a missing download IS an error (nothing to give the user), a missing log is a normal, expected state for a `pending`/`running` run. Don't make these consistent with each other.

- [ ] **Step 4: Run tests to verify they fail**

Run: `pnpm test api-client`
Expected: FAIL — `getRunDispatch`/`downloadRunArtifact`/`getRunLog` are not exported yet.

- [ ] **Step 5: Implement the three functions**

In `frontend/lib/api-client.ts`, add these exports at the end of the file (they reuse the existing private `authHeader()` and `API_BASE_URL` already defined at the top of the file):

```ts
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
```

Add `DispatchRow` to the existing type-only import at the top of the file (it already imports `CreateRunRequest, CreateScenarioRequest, RunDetail, RunSummary, Scenario` from `"./types"` — add `DispatchRow` to that list).

- [ ] **Step 6: Run tests to verify they pass**

Run: `pnpm test api-client`
Expected: PASS (all api-client tests, old and new)

- [ ] **Step 7: Install recharts**

`recharts` was already installed while verifying Task 3's Recharts/jsdom rendering approach before this plan was finalized — check `frontend/package.json` first:

```bash
grep recharts frontend/package.json
```

If it's already listed under `"dependencies"` (as `"recharts": "^3.10.1"` or similar), skip this step — nothing to do. Otherwise run `pnpm add recharts` and verify it landed in `"dependencies"`.

- [ ] **Step 8: Run the whole frontend test suite to check for regressions from the `RunDetail` type change**

Run: `pnpm test`
Expected: all tests pass. If any file besides `run-comparison-table.test.tsx` fails to compile because of the new required `artifacts` field, add `artifacts: { dispatch: false, prices: false, bess: false }` to that fixture too — Step 2 above covers the one file known to need it, but a Vitest/TS run is the authority here, not this plan's grep.

- [ ] **Step 9: Commit**

```bash
git add frontend/lib/types.ts frontend/lib/api-client.ts frontend/lib/api-client.test.ts frontend/components/run-comparison-table.test.tsx frontend/package.json frontend/pnpm-lock.yaml
git commit -m "feat(frontend): DispatchRow type, artifacts field, download/dispatch/log api-client functions, add recharts"
```

---

### Task 3: `DispatchChart` component (top-N generadores + "Otros")

**Files:**
- Modify: `frontend/vitest.setup.ts` (add a `ResizeObserver` stub — required, see Step 5)
- Create: `frontend/lib/dispatch-chart-data.ts`
- Create: `frontend/lib/dispatch-chart-data.test.ts`
- Create: `frontend/components/dispatch-chart.tsx`
- Create: `frontend/components/dispatch-chart.test.tsx`

**Interfaces:**
- Consumes: `DispatchRow` type from `frontend/lib/types.ts` (Task 2).
- Produces: `DispatchChart({ rows: DispatchRow[] })` component, used by Task 6.

**Charting decisions (no dataviz skill was available in this environment — these are applied directly):**
- The dispatch CSV has one row per `(generador, datetime)` pair with no upper bound on distinct `generador` values in real data (the only fixture available, `tests/fixtures/xm_smoke/`, has just 2 — not representative). A line or stacked-area chart with one series per generador does not scale past a handful of series before the legend and colors become unreadable.
- Design: sum total `dispatch` per generador across all rows, take the top 6 by that total as individual series, sum everyone else into one `"Otros"` series. If there are 6 or fewer distinct generadores, no `"Otros"` bucket is created (nothing to aggregate).
- X axis: hour of day (0-23), parsed from the `datetime` string's time-of-day substring — NOT a `Date` object. The spec is explicit that the axis is "indice de hora del dia, no timestamp," and string-splitting sidesteps timezone-conversion bugs entirely (`new Date(...)` would apply the browser's local timezone to a naive datetime string, silently shifting hours).
- Chart type: stacked area (`recharts`'s `<AreaChart>` with every `<Area>` sharing `stackId="dispatch"`) — this is the standard way to show a generation-dispatch composition over time (total height = total dispatched energy that hour, band widths = each generator's contribution), and it's more readable than overlapping lines when there are 6-7 series.

- [ ] **Step 1: Write the failing tests for the data-transform function**

Create `frontend/lib/dispatch-chart-data.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { toHourlyDispatchSeries } from "./dispatch-chart-data";
import type { DispatchRow } from "./types";

describe("toHourlyDispatchSeries", () => {
  it("groups rows by hour-of-day parsed from the datetime string, not a Date object", () => {
    const rows: DispatchRow[] = [
      { generador: "A", datetime: "2024-04-18 00:00:00", dispatch: 10 },
      { generador: "A", datetime: "2024-04-18 01:00:00", dispatch: 20 },
    ];

    const { data } = toHourlyDispatchSeries(rows);

    expect(data).toEqual([
      { hour: 0, A: 10 },
      { hour: 1, A: 20 },
    ]);
  });

  it("keeps every generador as its own series when there are 6 or fewer", () => {
    const rows: DispatchRow[] = [
      { generador: "A", datetime: "2024-04-18 00:00:00", dispatch: 10 },
      { generador: "B", datetime: "2024-04-18 00:00:00", dispatch: 5 },
    ];

    const { data, seriesKeys } = toHourlyDispatchSeries(rows);

    expect(seriesKeys.sort()).toEqual(["A", "B"]);
    expect(data).toEqual([{ hour: 0, A: 10, B: 5 }]);
  });

  it("buckets everything past the top 6 (by total dispatch) into Otros", () => {
    const rows: DispatchRow[] = [
      { generador: "G1", datetime: "2024-04-18 00:00:00", dispatch: 100 },
      { generador: "G2", datetime: "2024-04-18 00:00:00", dispatch: 90 },
      { generador: "G3", datetime: "2024-04-18 00:00:00", dispatch: 80 },
      { generador: "G4", datetime: "2024-04-18 00:00:00", dispatch: 70 },
      { generador: "G5", datetime: "2024-04-18 00:00:00", dispatch: 60 },
      { generador: "G6", datetime: "2024-04-18 00:00:00", dispatch: 50 },
      { generador: "G7", datetime: "2024-04-18 00:00:00", dispatch: 5 },
      { generador: "G8", datetime: "2024-04-18 00:00:00", dispatch: 3 },
    ];

    const { data, seriesKeys } = toHourlyDispatchSeries(rows);

    expect(seriesKeys).toEqual(["G1", "G2", "G3", "G4", "G5", "G6", "Otros"]);
    expect(data[0].Otros).toBe(8);
    expect(data[0].G7).toBeUndefined();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm test dispatch-chart-data`
Expected: FAIL — `./dispatch-chart-data` module doesn't exist.

- [ ] **Step 3: Implement the transform**

Create `frontend/lib/dispatch-chart-data.ts`:

```ts
import type { DispatchRow } from "./types";

export interface HourlyDispatchPoint {
  hour: number;
  [series: string]: number;
}

const TOP_N = 6;
const OTHER_LABEL = "Otros";

function hourOfDay(datetime: string): number {
  const timePart = datetime.split(" ")[1] ?? "00:00";
  return Number(timePart.split(":")[0]);
}

export function toHourlyDispatchSeries(rows: DispatchRow[]): {
  data: HourlyDispatchPoint[];
  seriesKeys: string[];
} {
  const totals = new Map<string, number>();
  for (const row of rows) {
    totals.set(row.generador, (totals.get(row.generador) ?? 0) + row.dispatch);
  }

  const sortedByTotal = [...totals.entries()].sort((a, b) => b[1] - a[1]);
  const topGeneradores = sortedByTotal.slice(0, TOP_N).map(([name]) => name);
  const topSet = new Set(topGeneradores);
  const hasOthers = totals.size > topGeneradores.length;

  const byHour = new Map<number, HourlyDispatchPoint>();
  for (const row of rows) {
    const hour = hourOfDay(row.datetime);
    const point = byHour.get(hour) ?? { hour };
    const key = topSet.has(row.generador) ? row.generador : OTHER_LABEL;
    point[key] = (point[key] ?? 0) + row.dispatch;
    byHour.set(hour, point);
  }

  const data = [...byHour.values()].sort((a, b) => a.hour - b.hour);
  const seriesKeys = hasOthers ? [...topGeneradores, OTHER_LABEL] : topGeneradores;
  return { data, seriesKeys };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm test dispatch-chart-data`
Expected: PASS (3 tests)

- [ ] **Step 5: Add a `ResizeObserver` stub to the shared test setup**

This change was already applied to `frontend/vitest.setup.ts` while verifying Task 3's Recharts/jsdom approach before this plan was finalized — check the file's current contents first. If it already matches the code block below, skip straight to Step 6. Otherwise apply it now.

jsdom has no layout engine, so Recharts' `ResponsiveContainer` (which sizes itself via `ResizeObserver`) never measures a non-zero size and renders nothing — verified empirically before writing this task: without a stub, `container.innerHTML` after rendering a chart is just `<div class="recharts-responsive-container" ...><div style="width: 0px; ..."></div></div>`, no SVG. A no-op stub (`observe() {}`) is not enough either — `ResponsiveContainer` only re-renders once its `ResizeObserver` callback actually fires, so the stub must invoke the callback synchronously with a non-zero `contentRect`.

Modify `frontend/vitest.setup.ts` — it currently contains only `import "@testing-library/jest-dom/vitest";`. Replace its full contents with:

```ts
import "@testing-library/jest-dom/vitest";

// jsdom has no layout engine, so Recharts' ResponsiveContainer (which sizes
// itself via ResizeObserver) never measures a non-zero size and renders
// nothing. Stub it to synchronously report a fixed size on observe().
class ResizeObserverStub {
  private cb: ResizeObserverCallback;
  constructor(cb: ResizeObserverCallback) {
    this.cb = cb;
  }
  observe(target: Element) {
    this.cb(
      [{ target, contentRect: { width: 500, height: 320 } } as ResizeObserverEntry],
      this as unknown as ResizeObserver
    );
  }
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver;
```

This is global (applies to every test file), but harmless — nothing else in the suite uses `ResizeObserver`, confirmed by running the full `pnpm test` suite with this change in place (36/36 pass, same count as before the stub).

- [ ] **Step 6: Write the failing test for the chart component**

Create `frontend/components/dispatch-chart.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DispatchChart } from "./dispatch-chart";
import type { DispatchRow } from "@/lib/types";

describe("DispatchChart", () => {
  it("renders a chart when there are rows", () => {
    const rows: DispatchRow[] = [
      { generador: "A", datetime: "2024-04-18 00:00:00", dispatch: 10 },
      { generador: "A", datetime: "2024-04-18 01:00:00", dispatch: 20 },
    ];

    const { container } = render(<DispatchChart rows={rows} />);

    expect(container.querySelector(".recharts-wrapper, svg")).toBeTruthy();
  });

  it("shows an empty-state message instead of a chart when there are no rows", () => {
    render(<DispatchChart rows={[]} />);

    expect(screen.getByText(/no hay datos de despacho/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 7: Run test to verify it fails**

Run: `pnpm test dispatch-chart.test`
Expected: FAIL — `./dispatch-chart` module doesn't exist.

- [ ] **Step 8: Implement the component**

Create `frontend/components/dispatch-chart.tsx`:

```tsx
"use client";

import { toHourlyDispatchSeries } from "@/lib/dispatch-chart-data";
import type { DispatchRow } from "@/lib/types";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const COLORS = ["#2563eb", "#16a34a", "#d97706", "#dc2626", "#7c3aed", "#0891b2", "#6b7280"];

export function DispatchChart({ rows }: { rows: DispatchRow[] }) {
  if (rows.length === 0) {
    return <p>No hay datos de despacho todavia.</p>;
  }

  const { data, seriesKeys } = toHourlyDispatchSeries(rows);

  return (
    <ResponsiveContainer width="100%" height={320}>
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="hour" label={{ value: "Hora del dia", position: "insideBottom", offset: -5 }} />
        <YAxis label={{ value: "MW", angle: -90, position: "insideLeft" }} />
        <Tooltip />
        <Legend />
        {seriesKeys.map((key, i) => (
          <Area
            key={key}
            type="monotone"
            dataKey={key}
            stackId="dispatch"
            stroke={COLORS[i % COLORS.length]}
            fill={COLORS[i % COLORS.length]}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}
```

- [ ] **Step 9: Run test to verify it passes**

Run: `pnpm test dispatch-chart.test` (NOT `pnpm test dispatch-chart` — that substring matches both `dispatch-chart-data.test.ts` and `dispatch-chart.test.tsx`, since Vitest's positional argument is a filename filter, not a test-name filter; `dispatch-chart.test` matches only the component's test file).
Expected: PASS (2 tests).

- [ ] **Step 10: Run the whole frontend suite to confirm the `vitest.setup.ts` change caused no regressions**

Run: `pnpm test`
Expected: all tests pass (35 pre-existing + 4 new from this task = 39).

- [ ] **Step 11: Commit**

```bash
git add frontend/vitest.setup.ts frontend/lib/dispatch-chart-data.ts frontend/lib/dispatch-chart-data.test.ts frontend/components/dispatch-chart.tsx frontend/components/dispatch-chart.test.tsx
git commit -m "feat(frontend): DispatchChart (top-6 generadores + Otros, stacked area by hour)"
```

---

### Task 4: `ArtifactDownloads` component

**Files:**
- Create: `frontend/components/artifact-downloads.tsx`
- Create: `frontend/components/artifact-downloads.test.tsx`

**Interfaces:**
- Consumes: `downloadRunArtifact(id, artifact)` from `frontend/lib/api-client.ts` (Task 2), `RunArtifacts` type from `frontend/lib/types.ts` (Task 2).
- Produces: `ArtifactDownloads({ runId: string, artifacts: RunArtifacts })` component, used by Task 6.

- [ ] **Step 1: Write the failing tests**

Create `frontend/components/artifact-downloads.test.tsx`:

```tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ArtifactDownloads } from "./artifact-downloads";

vi.mock("@/lib/api-client", () => ({
  downloadRunArtifact: vi.fn(),
}));

import { downloadRunArtifact } from "@/lib/api-client";

beforeEach(() => {
  vi.mocked(downloadRunArtifact).mockReset();
  vi.stubGlobal("URL", {
    ...URL,
    createObjectURL: vi.fn(() => "blob:mock-url"),
    revokeObjectURL: vi.fn(),
  });
});

describe("ArtifactDownloads", () => {
  it("shows an empty-state message when no artifacts are available", () => {
    render(<ArtifactDownloads runId="run-1" artifacts={{ dispatch: false, prices: false, bess: false }} />);

    expect(screen.getByText(/no hay artefactos disponibles/i)).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("renders one button per available artifact and none for unavailable ones", () => {
    render(<ArtifactDownloads runId="run-1" artifacts={{ dispatch: true, prices: true, bess: false }} />);

    expect(screen.getByRole("button", { name: /despacho/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /precios/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /bess/i })).not.toBeInTheDocument();
  });

  it("clicking a download button fetches the blob and triggers a synthetic anchor click", async () => {
    const fakeBlob = new Blob(["csv,data"]);
    vi.mocked(downloadRunArtifact).mockResolvedValue(fakeBlob);
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    render(<ArtifactDownloads runId="run-1" artifacts={{ dispatch: true, prices: false, bess: false }} />);
    fireEvent.click(screen.getByRole("button", { name: /despacho/i }));

    await vi.waitFor(() => {
      expect(downloadRunArtifact).toHaveBeenCalledWith("run-1", "dispatch");
    });
    expect(clickSpy).toHaveBeenCalled();

    clickSpy.mockRestore();
  });

  it("shows an error message when the download fails", async () => {
    vi.mocked(downloadRunArtifact).mockRejectedValue(new Error("500"));

    render(<ArtifactDownloads runId="run-1" artifacts={{ dispatch: true, prices: false, bess: false }} />);
    fireEvent.click(screen.getByRole("button", { name: /despacho/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/no se pudo descargar/i);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm test artifact-downloads`
Expected: FAIL — `./artifact-downloads` module doesn't exist.

- [ ] **Step 3: Implement**

Create `frontend/components/artifact-downloads.tsx`:

```tsx
"use client";

import { downloadRunArtifact } from "@/lib/api-client";
import type { RunArtifacts } from "@/lib/types";
import { useState } from "react";

type ArtifactKey = keyof RunArtifacts;

const ARTIFACT_LABELS: Record<ArtifactKey, string> = {
  dispatch: "Despacho",
  prices: "Precios",
  bess: "BESS",
};

export function ArtifactDownloads({
  runId,
  artifacts,
}: {
  runId: string;
  artifacts: RunArtifacts;
}) {
  const [error, setError] = useState<string | null>(null);

  async function handleDownload(artifact: ArtifactKey) {
    setError(null);
    try {
      const blob = await downloadRunArtifact(runId, artifact);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${artifact}-${runId}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {
      setError("No se pudo descargar el artefacto.");
    }
  }

  const available = (Object.keys(artifacts) as ArtifactKey[]).filter((key) => artifacts[key]);

  if (available.length === 0) {
    return <p>No hay artefactos disponibles todavia.</p>;
  }

  return (
    <div>
      {available.map((artifact) => (
        <button key={artifact} type="button" onClick={() => handleDownload(artifact)}>
          Descargar {ARTIFACT_LABELS[artifact]}
        </button>
      ))}
      {error && <p role="alert">{error}</p>}
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm test artifact-downloads`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add frontend/components/artifact-downloads.tsx frontend/components/artifact-downloads.test.tsx
git commit -m "feat(frontend): ArtifactDownloads (authenticated blob download, no bare anchor)"
```

---

### Task 5: `LogViewer` component

**Files:**
- Create: `frontend/hooks/use-run-log.ts`
- Create: `frontend/hooks/use-run-log.test.tsx`
- Create: `frontend/components/log-viewer.tsx`
- Create: `frontend/components/log-viewer.test.tsx`

**Interfaces:**
- Consumes: `getRunLog(id)` from `frontend/lib/api-client.ts` (Task 2, resolves `string | null`).
- Produces: `useRunLog(id: string)` hook (TanStack Query wrapper), `LogViewer({ runId: string })` component, used by Task 6.

- [ ] **Step 1: Write the failing test for the hook**

Create `frontend/hooks/use-run-log.test.tsx`:

```tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useRunLog } from "./use-run-log";

vi.mock("@/lib/api-client", () => ({
  getRunLog: vi.fn(),
}));

import { getRunLog } from "@/lib/api-client";

function wrapper({ children }: { children: React.ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("useRunLog", () => {
  it("returns the log text once loaded", async () => {
    vi.mocked(getRunLog).mockResolvedValue("line 1");

    const { result } = renderHook(() => useRunLog("run-1"), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toBe("line 1");
  });

  it("returns null (not an error) when there is no log yet", async () => {
    vi.mocked(getRunLog).mockResolvedValue(null);

    const { result } = renderHook(() => useRunLog("run-1"), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toBeNull();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm test use-run-log`
Expected: FAIL — `./use-run-log` module doesn't exist.

- [ ] **Step 3: Implement the hook**

Create `frontend/hooks/use-run-log.ts`:

```ts
import { getRunLog } from "@/lib/api-client";
import { useQuery } from "@tanstack/react-query";

export function useRunLog(runId: string) {
  return useQuery({
    queryKey: ["run-log", runId],
    queryFn: () => getRunLog(runId),
  });
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm test use-run-log`
Expected: PASS (2 tests)

- [ ] **Step 5: Write the failing test for the component**

Create `frontend/components/log-viewer.test.tsx`:

```tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { LogViewer } from "./log-viewer";

vi.mock("@/lib/api-client", () => ({
  getRunLog: vi.fn(),
}));

import { getRunLog } from "@/lib/api-client";

function renderWithClient(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

describe("LogViewer", () => {
  it("shows the log text in a <pre> when available", async () => {
    vi.mocked(getRunLog).mockResolvedValue("worker output line");

    renderWithClient(<LogViewer runId="run-1" />);

    expect(await screen.findByText("worker output line")).toBeInTheDocument();
  });

  it("shows 'sin logs todavia' instead of an error when there is no log yet", async () => {
    vi.mocked(getRunLog).mockResolvedValue(null);

    renderWithClient(<LogViewer runId="run-1" />);

    expect(await screen.findByText(/sin logs todavia/i)).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pnpm test log-viewer`
Expected: FAIL — `./log-viewer` module doesn't exist.

- [ ] **Step 7: Implement the component**

Create `frontend/components/log-viewer.tsx`:

```tsx
"use client";

import { useRunLog } from "@/hooks/use-run-log";

export function LogViewer({ runId }: { runId: string }) {
  const { data, isLoading } = useRunLog(runId);

  if (isLoading) return <p>Cargando log...</p>;
  if (!data) return <p>Sin logs todavia.</p>;

  return <pre style={{ maxHeight: "24rem", overflow: "auto" }}>{data}</pre>;
}
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pnpm test log-viewer`
Expected: PASS (2 tests)

- [ ] **Step 9: Commit**

```bash
git add frontend/hooks/use-run-log.ts frontend/hooks/use-run-log.test.tsx frontend/components/log-viewer.tsx frontend/components/log-viewer.test.tsx
git commit -m "feat(frontend): LogViewer (plain-text log fetch, 404 renders as empty state not error)"
```

---

### Task 6: Wire everything into the run-detail page + final verification

**Files:**
- Modify: `frontend/app/(app)/runs/[id]/page.tsx`

**Interfaces:**
- Consumes: `DispatchChart` (Task 3), `ArtifactDownloads` (Task 4), `LogViewer` (Task 5), `useRunDispatch`-equivalent data fetch via `getRunDispatch` (Task 2) wrapped in a `useQuery` inline in the page (no separate hook file needed — this is the only place it's used), `data.artifacts` from the existing `useRunDetail` hook (already returns the full `RunDetail`, which now includes `artifacts` per Task 2).

This task has no isolated unit test — the run-detail page has never had one (it wasn't tested in fase4a either, per the existing file), and correctness here is "does it compose the pieces correctly," which `pnpm build`'s type-checking plus the already-tested pieces from Tasks 3-5 cover. Verify via `pnpm build` and a read-through, not a new test file.

- [ ] **Step 1: Read the current page**

Read `frontend/app/(app)/runs/[id]/page.tsx` in full before editing — it already renders date/level/status/timestamps/error/partial-metrics. Do not remove or restructure any of that; only add new sections.

- [ ] **Step 2: Add the dispatch chart, artifact downloads, and log viewer sections**

Note: the dispatch query is gated with `enabled: Boolean(data?.artifacts.dispatch)`, so when there's no dispatch artifact `dispatchQuery.data` just stays `undefined` and `DispatchChart` receives `rows={[]}` — `DispatchChart` already renders its own "No hay datos de despacho todavia." for the empty-rows case (Task 3), so there's no need for a second `data.artifacts.dispatch ? ... : ...` branch in the page itself. Don't add one — it would just duplicate the same message from two places.

Replace the full contents of `frontend/app/(app)/runs/[id]/page.tsx` with:

```tsx
"use client";

import { ArtifactDownloads } from "@/components/artifact-downloads";
import { DispatchChart } from "@/components/dispatch-chart";
import { LogViewer } from "@/components/log-viewer";
import { formatBogotaTime } from "@/lib/format-date";
import { useRunDetail } from "@/hooks/use-run-detail";
import { getRunDispatch } from "@/lib/api-client";
import { useQuery } from "@tanstack/react-query";
import { useParams } from "next/navigation";

export default function RunDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { data, isLoading } = useRunDetail(id);

  const dispatchQuery = useQuery({
    queryKey: ["run-dispatch", id],
    queryFn: () => getRunDispatch(id),
    enabled: Boolean(data?.artifacts.dispatch),
  });

  if (isLoading || !data) return <p>Cargando...</p>;

  return (
    <div>
      <h1>Ejecucion {data.run_id}</h1>
      <p>Fecha: {data.dispatch_date}</p>
      <p>Nivel: {data.level}</p>
      <p>Status: {data.status}</p>
      <p>Creado: {formatBogotaTime(data.created_at)}</p>
      <p>Iniciado: {formatBogotaTime(data.started_at)}</p>
      <p>Terminado: {formatBogotaTime(data.finished_at)}</p>
      {data.status === "failed" && data.error && (
        <p role="alert">Error: {data.error}</p>
      )}
      {data.metrics && (
        <dl>
          <dt>RMSE</dt>
          <dd>{data.metrics.rmse}</dd>
          <dt>R2</dt>
          <dd>{data.metrics.r2}</dd>
        </dl>
      )}

      <h2>Despacho por generador</h2>
      <DispatchChart rows={dispatchQuery.data ?? []} />

      <h2>Artefactos</h2>
      <ArtifactDownloads runId={data.run_id} artifacts={data.artifacts} />

      <h2>Log</h2>
      <LogViewer runId={data.run_id} />
    </div>
  );
}
```

- [ ] **Step 3: Run the whole frontend test suite**

Run: `pnpm test`
Expected: all tests pass (fase4a + fase4b + this plan's new tests from Tasks 2-5).

- [ ] **Step 4: Run the whole frontend build**

Run: `pnpm build`
Expected: build succeeds, no type errors. If `NEXT_PUBLIC_SUPABASE_URL`/`NEXT_PUBLIC_SUPABASE_ANON_KEY` env vars are required for the build to complete (they were in fase4b), check that plan's task-8-report or the repo's `.env.example`/README for the exact dummy values used before, and reuse the same pattern.

- [ ] **Step 5: Run the whole backend test suite (Task 1's change plus everything else)**

Run: `uv run pytest`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add frontend/app/\(app\)/runs/\[id\]/page.tsx
git commit -m "feat(frontend): wire dispatch chart, artifact downloads, log viewer into run detail page"
```
