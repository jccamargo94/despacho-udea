# Fase 4b: Configurador de escenario BESS + Comparador — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the BESS scenario configurator (`/scenarios`) and the runs
comparator (`/compare`), both frontend-only, on top of the backend and
frontend foundation Fase 4a already shipped.

**Architecture:** Two new route-group pages under `frontend/app/(app)/`,
each assembled from small presentational components + one TanStack Query
hook where needed, following the exact same decomposition Fase 4a used for
the runs screen (table component + form component + page). No backend
changes — `POST /scenarios`, `GET /scenarios`, and `GET /runs/{id}` already
cover everything this plan needs.

**Tech Stack:** Same as Fase 4a — Next.js App Router (client components),
TypeScript, `@tanstack/react-query` (`useMutation`, `useQuery`,
`useQueries`), Vitest + React Testing Library. `Button`/`Card`/`Input`/
`Label`/`Table`/`Badge` from `frontend/components/ui/*` (shadcn, installed
in Fase 4a Task 9, unused until now).

## Global Constraints

- Frontend commands run via `pnpm` inside `frontend/`. No backend/Python
  work in this plan.
- Spec: `docs/superpowers/specs/2026-08-05-fase4-frontend-operativo-design.md`,
  section "fase4b" (amended 2026-08-06 with the decisions below).
- **`BessMode.generator` has no Pyomo formulation** — confirmed this
  session (`app/model/model.py:310`: `raise NotImplementedError("BESS mode
  'generator' has no Pyomo formulation yet")`). The scenario creation form
  offers only `"arbitrage"` and `"grid_asset"` as options — never
  `"generator"`.
- **Client-side validation is UX only, not authority.** `POST /scenarios`
  already 422s via `BessScenario`'s server-side validator
  (`app/schemas/bess.py`) if `charge_bid`/`discharge_bid` are missing for
  `arbitrage`. The form shows/hides those two fields based on `mode` for a
  better UX, but does not reimplement the validation rule — a server 422
  is surfaced via the existing `mutation.isError` / `mutation.error`
  pattern (see Fase 4a's `create-run-form.tsx`), raw text is acceptable
  (no `detail` JSON parsing in this plan — deferred, not a defect).
- **No shadcn `Select`.** It wraps `@base-ui/react/select`, a headless
  popover (Portal + Trigger/Content/Item) that needs click+portal
  interaction in tests, not `fireEvent.change`. Every `<select>` in this
  plan is a native HTML `<select>`, exactly the pattern already proven in
  `frontend/components/create-run-form.tsx`.
- **`POST /scenarios` returns `{"id": row.id}`** — verified against
  `services/api/main.py`. Not `{"scenario_id": ...}` — don't confuse with
  `POST /runs`'s `{"run_id": ..., "status": ...}` shape.
- **`GET /runs` has no metrics field** (`_run_summary` in
  `services/api/main.py` only returns id/status/case fields/timestamps).
  Only `GET /runs/{id}` (`get_run_detail`) has `metrics`. The comparator
  fetches each selected run's detail individually via `useQueries`.
- **`status === "done"` does not imply `metrics` is non-null.** A
  successful run with no matching XM actuals skips `evaluate` and has
  `metrics: null` in its detail response. The comparator must render this
  case explicitly, not assume every "done" run has numbers.
- **TanStack Query v5's `mutationFn` receives `(variables, context)`.**
  Fase 4a's Task 16 hit this: a bare `mutationFn: someFunction` plus a
  single-argument `toHaveBeenCalledWith` assertion fails because the real
  call has 2 arguments. Every `useMutation` in this plan wraps its
  `mutationFn` as `(variables: X) => someFunction(variables)`.
- Every task: write the failing test first, confirm it fails for the
  right reason, implement, confirm it passes, run the relevant test file,
  commit. Page-assembly tasks (no isolated test, pure composition) are
  verified by `pnpm build` instead — this mirrors Fase 4a's Tasks 15/16/17
  pattern exactly.
- After the last task: `cd frontend && pnpm test && pnpm build` — the full
  suite (Fase 4a's 19 tests + this plan's new ones) must be green and the
  build must succeed.

---

## Task 1: Carry-overs — `BessUnit`/`BessMode` types, `createScenario`, shadcn CLI to devDependencies

**Files:**
- Modify: `frontend/lib/types.ts`
- Modify: `frontend/lib/api-client.ts`
- Modify: `frontend/lib/api-client.test.ts`
- Modify: `frontend/package.json`

**Interfaces:**
- Produces: `BessMode` (`"arbitrage" | "grid_asset" | "generator"`),
  `BessUnit` interface, `Scenario.units: BessUnit[]` (was `unknown[]`),
  `Scenario.mode: BessMode` (was an inline 3-value union — same values,
  now named), `CreateScenarioRequest` interface (`mode` narrowed to
  `"arbitrage" | "grid_asset"` — deliberately excludes `"generator"` at
  the type level), `createScenario(body: CreateScenarioRequest):
  Promise<{ id: string }>`.

- [ ] **Step 1: Write the failing test** — append to
  `frontend/lib/api-client.test.ts` (same file, same `fetchMock`/
  `supabase` mocks already set up at the top):

```ts
import { createScenario } from "./api-client";
```

(add `createScenario` to the existing `import { createRun, listRuns } from
"./api-client";` line instead of a new import line)

```ts
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && pnpm test api-client`
Expected: FAIL (`createScenario` is not exported yet).

- [ ] **Step 3: Update `frontend/lib/types.ts`** — replace the `Scenario`
  interface and add the two new types (insert `BessMode`/`BessUnit`
  before `Scenario`, keep everything else in the file unchanged):

```ts
export type BessMode = "arbitrage" | "grid_asset" | "generator";

export interface BessUnit {
  name: string;
  mwh_nom: number;
  hours_to_deplete: number;
  initial_soc: number;
  min_soc: number;
  max_soc: number;
  efficiency: number;
  charge_bid?: number | null;
  discharge_bid?: number | null;
}

export interface Scenario {
  id: string;
  mode: BessMode;
  penetration_level: string;
  units: BessUnit[];
  created_at: string;
}

export interface CreateScenarioRequest {
  mode: "arbitrage" | "grid_asset";
  penetration_level: string;
  units: BessUnit[];
}
```

(This replaces the existing `Scenario` interface, which had
`mode: "arbitrage" | "grid_asset" | "generator"` inline and
`units: unknown[]`.)

- [ ] **Step 4: Update `frontend/lib/api-client.ts`** — add the import and
  the new export. Change the import line:

```ts
import type { CreateRunRequest, CreateScenarioRequest, RunDetail, RunSummary, Scenario } from "./types";
```

Add after `listScenarios`:

```ts
export function createScenario(body: CreateScenarioRequest): Promise<{ id: string }> {
  return request("/scenarios", { method: "POST", body: JSON.stringify(body) });
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pnpm test api-client`
Expected: PASS.

- [ ] **Step 6: Type-check the whole frontend** (catches any other file
  that relied on `Scenario.units` being `unknown[]`):

Run: `pnpm build`
Expected: succeeds. (`create-run-form.tsx` reads `s.penetration_level`/
`s.mode` only, never `.units`, so this should be a no-op change for it.)

- [ ] **Step 7: Move `shadcn` CLI to devDependencies** — in
  `frontend/package.json`, find `"shadcn"` under `"dependencies"` and move
  that line to `"devDependencies"` (create the key alphabetically among
  the existing devDependencies entries). Run `pnpm install` to regenerate
  the lockfile consistently, then `pnpm build` again to confirm nothing
  broke.

- [ ] **Step 8: Commit**

```bash
git add frontend/lib/types.ts frontend/lib/api-client.ts frontend/lib/api-client.test.ts frontend/package.json frontend/pnpm-lock.yaml
git commit -m "feat(frontend): BessUnit/BessMode types, createScenario, move shadcn CLI to devDependencies"
```

---

## Task 2: `ScenariosTable` component

**Files:**
- Create: `frontend/components/scenarios-table.tsx`
- Create: `frontend/components/scenarios-table.test.tsx`

**Interfaces:**
- Consumes: `Scenario` type (Task 1).
- Produces: `ScenariosTable({ scenarios: Scenario[] })`.

- [ ] **Step 1: Write the failing test** — create
  `frontend/components/scenarios-table.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ScenariosTable } from "./scenarios-table";
import type { Scenario } from "@/lib/types";

const scenarios: Scenario[] = [
  {
    id: "scn-1",
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
    created_at: "2024-04-18T05:00:00Z",
  },
];

describe("ScenariosTable", () => {
  it("renders one row per scenario with mode/penetration/unit count", () => {
    render(<ScenariosTable scenarios={scenarios} />);
    expect(screen.getByText("baseline")).toBeInTheDocument();
    expect(screen.getByText("arbitrage")).toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();
  });

  it("renders an empty state with no scenarios", () => {
    render(<ScenariosTable scenarios={[]} />);
    expect(screen.getByText(/sin escenarios/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm test scenarios-table`
Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Implement** — create
  `frontend/components/scenarios-table.tsx`:

```tsx
import type { Scenario } from "@/lib/types";

export function ScenariosTable({ scenarios }: { scenarios: Scenario[] }) {
  if (scenarios.length === 0) return <p>Sin escenarios todavia.</p>;

  return (
    <table>
      <thead>
        <tr>
          <th>Nivel de penetracion</th>
          <th>Modo</th>
          <th>Unidades</th>
        </tr>
      </thead>
      <tbody>
        {scenarios.map((s) => (
          <tr key={s.id}>
            <td>{s.penetration_level}</td>
            <td>{s.mode}</td>
            <td>{s.units.length}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm test scenarios-table`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/components/scenarios-table.tsx frontend/components/scenarios-table.test.tsx
git commit -m "feat(frontend): ScenariosTable component"
```

---

## Task 3: `CreateScenarioForm` component

**Files:**
- Create: `frontend/components/create-scenario-form.tsx`
- Create: `frontend/components/create-scenario-form.test.tsx`

**Interfaces:**
- Consumes: `createScenario` (Task 1), `BessUnit`/`CreateScenarioRequest`
  types (Task 1).
- Produces: `CreateScenarioForm({ onCreated: () => void })`.

- [ ] **Step 1: Write the failing test** — create
  `frontend/components/create-scenario-form.test.tsx`:

```tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import { CreateScenarioForm } from "./create-scenario-form";

vi.mock("@/lib/api-client", () => ({
  createScenario: vi.fn().mockResolvedValue({ id: "scn-1" }),
}));

import { createScenario } from "@/lib/api-client";

function renderWithQueryClient(ui: React.ReactElement) {
  const client = new QueryClient();
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

describe("CreateScenarioForm", () => {
  it("submits mode/penetration_level/units, including bid fields in arbitrage mode", async () => {
    const onCreated = vi.fn();
    renderWithQueryClient(<CreateScenarioForm onCreated={onCreated} />);

    fireEvent.change(screen.getByLabelText(/nivel de penetracion/i), {
      target: { value: "baseline" },
    });
    fireEvent.change(screen.getByLabelText(/^nombre$/i), { target: { value: "bess-1" } });
    fireEvent.change(screen.getByLabelText(/capacidad/i), { target: { value: "10" } });
    fireEvent.change(screen.getByLabelText(/oferta de carga/i), { target: { value: "50" } });
    fireEvent.change(screen.getByLabelText(/oferta de descarga/i), { target: { value: "200" } });
    fireEvent.click(screen.getByRole("button", { name: /crear escenario/i }));

    await waitFor(() =>
      expect(createScenario).toHaveBeenCalledWith(
        expect.objectContaining({
          mode: "arbitrage",
          penetration_level: "baseline",
          units: [expect.objectContaining({ name: "bess-1", mwh_nom: 10, charge_bid: 50, discharge_bid: 200 })],
        })
      )
    );
    await waitFor(() => expect(onCreated).toHaveBeenCalled());
  });

  it("hides bid fields when mode is grid_asset", () => {
    renderWithQueryClient(<CreateScenarioForm onCreated={vi.fn()} />);

    fireEvent.change(screen.getByLabelText(/^modo$/i), { target: { value: "grid_asset" } });

    expect(screen.queryByLabelText(/oferta de carga/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/oferta de descarga/i)).not.toBeInTheDocument();
  });

  it("adds a second unit row when 'Agregar unidad' is clicked", () => {
    renderWithQueryClient(<CreateScenarioForm onCreated={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: /agregar unidad/i }));

    expect(screen.getAllByLabelText(/^nombre$/i)).toHaveLength(2);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm test create-scenario-form`
Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Implement** — create
  `frontend/components/create-scenario-form.tsx`:

```tsx
"use client";

import { createScenario } from "@/lib/api-client";
import type { BessUnit, CreateScenarioRequest } from "@/lib/types";
import { useMutation } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";

function emptyUnit(): BessUnit {
  return {
    name: "",
    mwh_nom: 0,
    hours_to_deplete: 0,
    initial_soc: 0,
    min_soc: 0,
    max_soc: 0,
    efficiency: 0,
    charge_bid: null,
    discharge_bid: null,
  };
}

export function CreateScenarioForm({ onCreated }: { onCreated: () => void }) {
  const [mode, setMode] = useState<"arbitrage" | "grid_asset">("arbitrage");
  const [penetrationLevel, setPenetrationLevel] = useState("");
  const [units, setUnits] = useState<BessUnit[]>([emptyUnit()]);

  // TanStack Query v5 calls mutationFn(variables, context); the wrapper
  // keeps createScenario's own single-argument signature intact.
  const mutation = useMutation({
    mutationFn: (variables: CreateScenarioRequest) => createScenario(variables),
    onSuccess: onCreated,
  });

  function updateUnit(index: number, patch: Partial<BessUnit>) {
    setUnits((prev) => prev.map((u, i) => (i === index ? { ...u, ...patch } : u)));
  }

  function addUnit() {
    setUnits((prev) => [...prev, emptyUnit()]);
  }

  function removeUnit(index: number) {
    setUnits((prev) => prev.filter((_, i) => i !== index));
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    mutation.mutate({ mode, penetration_level: penetrationLevel, units });
  }

  return (
    <form onSubmit={handleSubmit}>
      <label htmlFor="mode">Modo</label>
      <select
        id="mode"
        value={mode}
        onChange={(e) => setMode(e.target.value as "arbitrage" | "grid_asset")}
      >
        <option value="arbitrage">arbitrage</option>
        <option value="grid_asset">grid_asset</option>
      </select>

      <label htmlFor="penetration_level">Nivel de penetracion</label>
      <input
        id="penetration_level"
        value={penetrationLevel}
        onChange={(e) => setPenetrationLevel(e.target.value)}
        required
      />

      {units.map((unit, i) => (
        <fieldset key={i}>
          <legend>Unidad {i + 1}</legend>

          <label htmlFor={`unit-${i}-name`}>Nombre</label>
          <input
            id={`unit-${i}-name`}
            value={unit.name}
            onChange={(e) => updateUnit(i, { name: e.target.value })}
            required
          />

          <label htmlFor={`unit-${i}-mwh_nom`}>Capacidad (MWh)</label>
          <input
            id={`unit-${i}-mwh_nom`}
            type="number"
            value={unit.mwh_nom}
            onChange={(e) => updateUnit(i, { mwh_nom: Number(e.target.value) })}
            required
          />

          <label htmlFor={`unit-${i}-hours_to_deplete`}>Horas para agotar</label>
          <input
            id={`unit-${i}-hours_to_deplete`}
            type="number"
            value={unit.hours_to_deplete}
            onChange={(e) => updateUnit(i, { hours_to_deplete: Number(e.target.value) })}
            required
          />

          <label htmlFor={`unit-${i}-initial_soc`}>SOC inicial</label>
          <input
            id={`unit-${i}-initial_soc`}
            type="number"
            value={unit.initial_soc}
            onChange={(e) => updateUnit(i, { initial_soc: Number(e.target.value) })}
            required
          />

          <label htmlFor={`unit-${i}-min_soc`}>SOC minimo</label>
          <input
            id={`unit-${i}-min_soc`}
            type="number"
            value={unit.min_soc}
            onChange={(e) => updateUnit(i, { min_soc: Number(e.target.value) })}
            required
          />

          <label htmlFor={`unit-${i}-max_soc`}>SOC maximo</label>
          <input
            id={`unit-${i}-max_soc`}
            type="number"
            value={unit.max_soc}
            onChange={(e) => updateUnit(i, { max_soc: Number(e.target.value) })}
            required
          />

          <label htmlFor={`unit-${i}-efficiency`}>Eficiencia</label>
          <input
            id={`unit-${i}-efficiency`}
            type="number"
            value={unit.efficiency}
            onChange={(e) => updateUnit(i, { efficiency: Number(e.target.value) })}
            required
          />

          {mode === "arbitrage" && (
            <>
              <label htmlFor={`unit-${i}-charge_bid`}>Oferta de carga</label>
              <input
                id={`unit-${i}-charge_bid`}
                type="number"
                value={unit.charge_bid ?? ""}
                onChange={(e) => updateUnit(i, { charge_bid: Number(e.target.value) })}
                required
              />

              <label htmlFor={`unit-${i}-discharge_bid`}>Oferta de descarga</label>
              <input
                id={`unit-${i}-discharge_bid`}
                type="number"
                value={unit.discharge_bid ?? ""}
                onChange={(e) => updateUnit(i, { discharge_bid: Number(e.target.value) })}
                required
              />
            </>
          )}

          <button type="button" onClick={() => removeUnit(i)}>
            Quitar unidad
          </button>
        </fieldset>
      ))}

      <button type="button" onClick={addUnit}>
        Agregar unidad
      </button>

      <button type="submit" disabled={mutation.isPending}>
        Crear escenario
      </button>

      {mutation.isError && <p role="alert">{(mutation.error as Error).message}</p>}
    </form>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm test create-scenario-form`
Expected: PASS (all 3 tests).

- [ ] **Step 5: Commit**

```bash
git add frontend/components/create-scenario-form.tsx frontend/components/create-scenario-form.test.tsx
git commit -m "feat(frontend): CreateScenarioForm with dynamic BESS unit rows"
```

---

## Task 4: Scenarios page + nav link

**Files:**
- Create: `frontend/app/(app)/scenarios/page.tsx`
- Modify: `frontend/app/(app)/layout.tsx`

**Interfaces:**
- Consumes: `ScenariosTable` (Task 2), `CreateScenarioForm` (Task 3),
  `listScenarios` (existing, Fase 4a Task 14).

No isolated test for this task — pure composition of already-tested
pieces plus a one-line nav addition, verified by `pnpm build`. This
mirrors Fase 4a's Task 16 (page assembly) pattern.

- [ ] **Step 1: Assemble the page** — create
  `frontend/app/(app)/scenarios/page.tsx`:

```tsx
"use client";

import { CreateScenarioForm } from "@/components/create-scenario-form";
import { ScenariosTable } from "@/components/scenarios-table";
import { listScenarios } from "@/lib/api-client";
import { useQuery, useQueryClient } from "@tanstack/react-query";

export default function ScenariosPage() {
  const queryClient = useQueryClient();
  const scenariosQuery = useQuery({ queryKey: ["scenarios"], queryFn: listScenarios });

  return (
    <div>
      <h1>Escenarios BESS</h1>
      <CreateScenarioForm
        onCreated={() => queryClient.invalidateQueries({ queryKey: ["scenarios"] })}
      />
      {scenariosQuery.isLoading && <p>Cargando...</p>}
      {scenariosQuery.data && <ScenariosTable scenarios={scenariosQuery.data} />}
    </div>
  );
}
```

Note: this invalidates the same `["scenarios"]` query key that
`create-run-form.tsx` (Fase 4a) already uses for its scenario dropdown —
creating a scenario here will also refresh that dropdown next time it's
mounted, no extra wiring needed.

- [ ] **Step 2: Add the nav link** — in `frontend/app/(app)/layout.tsx`,
  add a link to `/scenarios` right after the existing `/runs` link inside
  the `<nav>`:

```tsx
        <Link href="/runs">Ejecuciones</Link>
        <Link href="/scenarios">Escenarios</Link>
```

- [ ] **Step 3: Verify**

Run: `pnpm build`
Expected: succeeds, `/scenarios` listed as a route.

- [ ] **Step 4: Commit**

```bash
git add "frontend/app/(app)/scenarios/page.tsx" "frontend/app/(app)/layout.tsx"
git commit -m "feat(frontend): scenarios page (list + create), nav link"
```

---

## Task 5: `useRunComparisons` hook

**Files:**
- Create: `frontend/hooks/use-run-comparisons.ts`
- Create: `frontend/hooks/use-run-comparisons.test.tsx`

**Interfaces:**
- Consumes: `getRun` (existing, Fase 4a Task 14).
- Produces: `useRunComparisons(ids: string[])` — a thin wrapper over
  `useQueries` returning `UseQueryResult<RunDetail>[]`, one per id, same
  order as the input array. Each entry uses query key `["run", id]` —
  the same key the Fase 4a run-detail page (`use-run-detail.ts`) already
  uses, so navigating to a run's detail page after comparing it reuses
  the cache instead of refetching.

- [ ] **Step 1: Write the failing test** — create
  `frontend/hooks/use-run-comparisons.test.tsx`:

```tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useRunComparisons } from "./use-run-comparisons";

vi.mock("@/lib/api-client", () => ({ getRun: vi.fn() }));
import { getRun } from "@/lib/api-client";

function wrapper({ children }: { children: React.ReactNode }) {
  const client = new QueryClient();
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("useRunComparisons", () => {
  it("returns one query result per id, in order", async () => {
    vi.mocked(getRun).mockImplementation((id: string) =>
      Promise.resolve({ run_id: id, status: "done" } as never)
    );

    const { result } = renderHook(() => useRunComparisons(["r1", "r2"]), { wrapper });

    await waitFor(() => expect(result.current).toHaveLength(2));
    await waitFor(() => expect(result.current[0].data?.run_id).toBe("r1"));
    await waitFor(() => expect(result.current[1].data?.run_id).toBe("r2"));
  });

  it("returns an empty array for an empty id list", () => {
    const { result } = renderHook(() => useRunComparisons([]), { wrapper });
    expect(result.current).toEqual([]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm test use-run-comparisons`
Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Implement** — create
  `frontend/hooks/use-run-comparisons.ts`:

```ts
import { getRun } from "@/lib/api-client";
import { useQueries } from "@tanstack/react-query";

export function useRunComparisons(ids: string[]) {
  return useQueries({
    queries: ids.map((id) => ({
      queryKey: ["run", id],
      queryFn: () => getRun(id),
    })),
  });
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm test use-run-comparisons`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/hooks/use-run-comparisons.ts frontend/hooks/use-run-comparisons.test.tsx
git commit -m "feat(frontend): useRunComparisons hook (useQueries over selected run ids)"
```

---

## Task 6: `RunSelector` component

**Files:**
- Create: `frontend/components/run-selector.tsx`
- Create: `frontend/components/run-selector.test.tsx`

**Interfaces:**
- Consumes: `RunSummary` type (existing, Fase 4a Task 10).
- Produces: `RunSelector({ runs: RunSummary[], selectedIds: string[],
  onToggle: (id: string) => void })`. Filters `runs` to `status ===
  "done"` internally — only done runs are ever selectable, since only
  they can possibly have metrics.

- [ ] **Step 1: Write the failing test** — create
  `frontend/components/run-selector.test.tsx`:

```tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RunSelector } from "./run-selector";
import type { RunSummary } from "@/lib/types";

function makeRun(overrides: Partial<RunSummary>): RunSummary {
  return {
    run_id: "r1",
    status: "done",
    dispatch_date: "2024-04-18",
    level: "preideal",
    scenario_id: null,
    created_at: "2024-04-18T05:00:00Z",
    started_at: null,
    finished_at: null,
    error: null,
    ...overrides,
  };
}

describe("RunSelector", () => {
  it("only renders checkboxes for done runs", () => {
    const runs = [makeRun({ run_id: "r1", status: "done" }), makeRun({ run_id: "r2", status: "pending" })];
    render(<RunSelector runs={runs} selectedIds={[]} onToggle={vi.fn()} />);
    expect(screen.getAllByRole("checkbox")).toHaveLength(1);
  });

  it("calls onToggle with the run id when its checkbox is clicked", () => {
    const onToggle = vi.fn();
    const runs = [makeRun({ run_id: "r1" })];
    render(<RunSelector runs={runs} selectedIds={[]} onToggle={onToggle} />);
    fireEvent.click(screen.getByRole("checkbox"));
    expect(onToggle).toHaveBeenCalledWith("r1");
  });

  it("checks the box for a run id already in selectedIds", () => {
    const runs = [makeRun({ run_id: "r1" })];
    render(<RunSelector runs={runs} selectedIds={["r1"]} onToggle={vi.fn()} />);
    expect(screen.getByRole("checkbox")).toBeChecked();
  });

  it("renders a message when there are no done runs", () => {
    const runs = [makeRun({ run_id: "r1", status: "pending" })];
    render(<RunSelector runs={runs} selectedIds={[]} onToggle={vi.fn()} />);
    expect(screen.getByText(/no hay ejecuciones completas/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm test run-selector`
Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Implement** — create `frontend/components/run-selector.tsx`:

```tsx
"use client";

import type { RunSummary } from "@/lib/types";

export function RunSelector({
  runs,
  selectedIds,
  onToggle,
}: {
  runs: RunSummary[];
  selectedIds: string[];
  onToggle: (id: string) => void;
}) {
  const doneRuns = runs.filter((r) => r.status === "done");

  if (doneRuns.length === 0) return <p>No hay ejecuciones completas para comparar.</p>;

  return (
    <fieldset>
      <legend>Seleccionar ejecuciones</legend>
      {doneRuns.map((run) => (
        <label key={run.run_id}>
          <input
            type="checkbox"
            checked={selectedIds.includes(run.run_id)}
            onChange={() => onToggle(run.run_id)}
          />
          {run.dispatch_date} ({run.level})
        </label>
      ))}
    </fieldset>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm test run-selector`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/components/run-selector.tsx frontend/components/run-selector.test.tsx
git commit -m "feat(frontend): RunSelector component (done runs only)"
```

---

## Task 7: `RunComparisonTable` component

**Files:**
- Create: `frontend/components/run-comparison-table.tsx`
- Create: `frontend/components/run-comparison-table.test.tsx`

**Interfaces:**
- Consumes: `RunDetail`/`RunMetrics` types (existing, Fase 4a Task 10).
- Produces: `RunComparisonTable({ runs: RunDetail[] })` — metrics as rows,
  one column per run. A run whose `metrics` is `null` gets a
  "(sin metricas)" marker in its header and "—" in every metric cell for
  that column — this is the explicit handling of the "done does not imply
  metrics" case from the Global Constraints.

- [ ] **Step 1: Write the failing test** — create
  `frontend/components/run-comparison-table.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RunComparisonTable } from "./run-comparison-table";
import type { RunDetail } from "@/lib/types";

function makeRun(overrides: Partial<RunDetail>): RunDetail {
  return {
    run_id: "r1",
    status: "done",
    dispatch_date: "2024-04-18",
    level: "preideal",
    scenario_id: null,
    created_at: "2024-04-18T05:00:00Z",
    started_at: null,
    finished_at: null,
    error: null,
    metrics: null,
    ...overrides,
  };
}

describe("RunComparisonTable", () => {
  it("renders one column per run with its metric values", () => {
    const runs = [
      makeRun({
        run_id: "r1",
        metrics: {
          rmse: 1.5,
          mae: 1.2,
          bias: 0.1,
          wape: 5.5,
          smape: 4.4,
          r2: 0.9,
          bess_charge_mwh: 10,
          bess_discharge_mwh: 9,
          bess_avg_soc_mwh: 5,
          bess_net_revenue: 1000,
        },
      }),
    ];
    render(<RunComparisonTable runs={runs} />);
    expect(screen.getByText("1.5")).toBeInTheDocument();
    expect(screen.getByText("1000")).toBeInTheDocument();
  });

  it("shows a no-metrics marker and dashes for a run with null metrics", () => {
    const runs = [makeRun({ run_id: "r2", metrics: null })];
    render(<RunComparisonTable runs={runs} />);
    expect(screen.getByText(/sin metricas/i)).toBeInTheDocument();
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm test run-comparison-table`
Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Implement** — create
  `frontend/components/run-comparison-table.tsx`:

```tsx
import type { RunDetail, RunMetrics } from "@/lib/types";

const METRIC_ROWS: { key: keyof RunMetrics; label: string }[] = [
  { key: "rmse", label: "RMSE" },
  { key: "mae", label: "MAE" },
  { key: "bias", label: "Bias" },
  { key: "wape", label: "WAPE" },
  { key: "smape", label: "sMAPE" },
  { key: "r2", label: "R2" },
  { key: "bess_charge_mwh", label: "BESS carga (MWh)" },
  { key: "bess_discharge_mwh", label: "BESS descarga (MWh)" },
  { key: "bess_avg_soc_mwh", label: "BESS SOC promedio (MWh)" },
  { key: "bess_net_revenue", label: "BESS ingreso neto" },
];

export function RunComparisonTable({ runs }: { runs: RunDetail[] }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Metrica</th>
          {runs.map((run) => (
            <th key={run.run_id}>
              {run.dispatch_date} ({run.level})
              {run.metrics === null && " (sin metricas)"}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {METRIC_ROWS.map((row) => (
          <tr key={row.key}>
            <td>{row.label}</td>
            {runs.map((run) => (
              <td key={run.run_id}>{run.metrics ? (run.metrics[row.key] ?? "—") : "—"}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm test run-comparison-table`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/components/run-comparison-table.tsx frontend/components/run-comparison-table.test.tsx
git commit -m "feat(frontend): RunComparisonTable (metrics as rows, runs as columns)"
```

---

## Task 8: Compare page + nav link

**Files:**
- Create: `frontend/app/(app)/compare/page.tsx`
- Modify: `frontend/app/(app)/layout.tsx`

**Interfaces:**
- Consumes: `RunSelector` (Task 6), `RunComparisonTable` (Task 7),
  `useRunComparisons` (Task 5), `listRuns` (existing, Fase 4a Task 14).

No isolated test for this task — pure composition of already-tested
pieces plus a one-line nav addition, verified by `pnpm build`.

- [ ] **Step 1: Assemble the page** — create
  `frontend/app/(app)/compare/page.tsx`:

```tsx
"use client";

import { RunComparisonTable } from "@/components/run-comparison-table";
import { RunSelector } from "@/components/run-selector";
import { useRunComparisons } from "@/hooks/use-run-comparisons";
import { listRuns } from "@/lib/api-client";
import type { RunDetail } from "@/lib/types";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

export default function ComparePage() {
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const runsQuery = useQuery({ queryKey: ["runs"], queryFn: listRuns });
  const comparisons = useRunComparisons(selectedIds);

  function toggle(id: string) {
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }

  const loadedRuns: RunDetail[] = comparisons
    .map((c) => c.data)
    .filter((d): d is RunDetail => d !== undefined);

  return (
    <div>
      <h1>Comparador</h1>
      {runsQuery.isLoading && <p>Cargando ejecuciones...</p>}
      {runsQuery.data && <RunSelector runs={runsQuery.data} selectedIds={selectedIds} onToggle={toggle} />}
      {loadedRuns.length > 0 && <RunComparisonTable runs={loadedRuns} />}
    </div>
  );
}
```

- [ ] **Step 2: Add the nav link** — in `frontend/app/(app)/layout.tsx`,
  add a link to `/compare` right after `/scenarios`:

```tsx
        <Link href="/runs">Ejecuciones</Link>
        <Link href="/scenarios">Escenarios</Link>
        <Link href="/compare">Comparar</Link>
```

- [ ] **Step 3: Verify**

Run: `pnpm build`
Expected: succeeds, `/compare` listed as a route.

- [ ] **Step 4: Full frontend verification** (last task of the plan):

```bash
pnpm test
pnpm build
```

Expected: all tests PASS (Fase 4a's 19 + this plan's new tests), build
succeeds.

- [ ] **Step 5: Commit**

```bash
git add "frontend/app/(app)/compare/page.tsx" "frontend/app/(app)/layout.tsx"
git commit -m "feat(frontend): compare page (run selector + metrics comparison table), nav link"
```

---

## Final verification

- [ ] `cd frontend && pnpm test` — full suite green.
- [ ] `cd frontend && pnpm build` — succeeds, `/scenarios` and `/compare`
  both listed as routes.
- [ ] Manual smoke test (needs real Supabase credentials, not possible in
  a sandbox): `pnpm dev`, log in, create a BESS scenario in `/scenarios`
  with `mode: arbitrage` (verify both bid fields are required and a
  `grid_asset` scenario can be created without them), select it when
  creating a run, then visit `/compare` and select 2+ done runs to confirm
  the table renders correctly including a run with null metrics.
- [ ] Open PR `fase4b-bess-comparador` -> `develop`, following the same
  pattern as PR #7/#9.
