# Fase 5a: Frontend visual design — tokens, nav shell, auth, Ejecuciones Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the AI-Studio-derived design system (`mockups/energy_dispatch/grid_performance_logic/DESIGN.md`) into the real `frontend/` Next.js app — design tokens, nav shell, login/signup/reset-password, and the Ejecuciones (runs) screen — with every field corrected against the real backend contract, not copied from the mockup.

**Architecture:** No new dependencies. Reskin existing components in place using the already-installed shadcn/ui primitives (`Button`, `Input`, `Label`, `Badge`, `Table`, `Card`) + `lucide-react` icons + Tailwind v4 CSS variables. One new route (`/reset-password`) for real Supabase password-recovery, justified as an auth requirement, not a new product surface.

**Tech Stack:** Next.js 16 (App Router), Tailwind v4, shadcn/ui ("base-nova"), `@tanstack/react-query`, `@supabase/supabase-js`, `lucide-react`, Vitest + React Testing Library.

## Global Constraints

- No new npm/pnpm dependencies — everything needed (`lucide-react`, shadcn primitives, Tailwind v4) is already installed.
- Native `<select>` only for dropdowns — never shadcn's `Select` (fase4b decision: it wraps a `@base-ui/react` popover that needs click+open+select simulation in tests instead of `fireEvent.change`; stay with the pattern already proven in `create-run-form.tsx`).
- No new routes this phase except `/reset-password` (real Supabase password-recovery landing page — the one exception the design spec grants).
- UI copy stays ASCII-only (no `ñ`/accented characters), matching the existing convention in every current `.tsx` file (e.g. "Contrasena", not "Contraseña").
- Solver picker offers exactly two options: `cbc` (enabled, default) and `highs` (rendered, `disabled` attribute — "proximamente"). No Gurobi/CPLEX/SCIP anywhere in the UI.
- Every component whose rendered DOM changes must have its `.test.tsx` updated in the same task — do not defer test fixes to a later cleanup task.
- Package manager is `pnpm` (`pnpm test`, `pnpm build`, `pnpm dev`).

---

## File Structure

| File | Responsibility |
|---|---|
| `frontend/app/globals.css` | Design tokens (light + dark CSS vars), fixes a pre-existing self-referential `--font-sans` bug |
| `frontend/app/layout.tsx` | Swaps Geist/Geist Mono for Inter/JetBrains Mono via `next/font/google` |
| `frontend/components/app-sidebar.tsx` (new) | Fixed nav shell: brand, 3 nav links with active state, sign-out |
| `frontend/components/app-sidebar.test.tsx` (new) | Tests for the above |
| `frontend/app/(app)/layout.tsx` | Wires `AppSidebar` into the authenticated shell (replaces the unstyled `<nav>`) |
| `frontend/app/login/page.tsx` | Reskin + real forgot-password (`resetPasswordForEmail`) + link to `/signup` |
| `frontend/app/login/page.test.tsx` | Adds forgot-password + signup-link tests |
| `frontend/app/reset-password/page.tsx` (new) | New route: set new password after the recovery email link |
| `frontend/app/reset-password/page.test.tsx` (new) | Tests for the above |
| `frontend/app/signup/page.tsx` | Reskin + adds Confirm Password field + link to `/login` |
| `frontend/app/signup/page.test.tsx` | Updates for the new field + adds login-link test |
| `frontend/lib/run-status.ts` | Adds `statusLabel()` / `statusBadgeVariant()` (Spanish labels + shadcn Badge variant per `RunStatus`) |
| `frontend/lib/run-status.test.ts` | Tests for the above |
| `frontend/lib/format-date.ts` | Adds `formatDuration(startedAt, finishedAt)` |
| `frontend/lib/format-date.test.ts` | Tests for the above |
| `frontend/components/runs-table.tsx` | Reskin with shadcn `Table`/`Badge`, mono font for data, duration column |
| `frontend/components/runs-table.test.tsx` | Updates for badge label text + new duration column |
| `frontend/app/(app)/runs/page.tsx` | Reskin page chrome (heading, card wrappers) |
| `frontend/components/create-run-form.tsx` | Reskin + solver picker (CBC enabled / HiGHS disabled) |
| `frontend/components/create-run-form.test.tsx` | Adds solver-default + HiGHS-disabled tests |

---

### Task 1: Design tokens (colors, fonts, radius, chart palette)

**Files:**
- Modify: `frontend/app/globals.css`
- Modify: `frontend/app/layout.tsx`

**Interfaces:**
- Produces: CSS variables `--background`, `--foreground`, `--card`, `--card-foreground`, `--popover`, `--popover-foreground`, `--primary`, `--primary-foreground`, `--secondary`, `--secondary-foreground`, `--muted`, `--muted-foreground`, `--accent`, `--accent-foreground`, `--destructive`, `--border`, `--input`, `--ring`, `--chart-1..5`, `--radius` — every later task's Tailwind classes (`bg-primary`, `text-muted-foreground`, `border-border`, `font-sans`, `font-mono`, etc.) resolve through these.

This task also fixes a real, pre-existing bug found while auditing the file: `@theme inline` declares `--font-sans: var(--font-sans);` — a self-referencing custom property, which is invalid CSS and resolves to nothing, meaning the `font-sans` Tailwind utility (applied to `<html>` in `app/layout.tsx`) has never actually pointed at the Geist font variable. Same for `--font-mono`.

Color values are taken directly from `mockups/energy_dispatch/grid_performance_logic/DESIGN.md`. Dark-mode values are **not invented** — they're read from the one screen AI Studio actually rendered dark (`CompararView.tsx`/`Sidebar.tsx` in the cloned `colombian-dispatch` repo), which itself uses `DESIGN.md`'s own `inverse-surface`/`inverse-on-surface`/`primary-fixed-dim`/`secondary-fixed-dim` tokens — confirmed by grep, not guessed.

- [ ] **Step 1: Replace the `:root` (light) color block**

In `frontend/app/globals.css`, replace the entire `:root { ... }` block with:

```css
:root {
  --background: #f8f9ff;
  --foreground: #0b1c30;
  --card: #ffffff;
  --card-foreground: #0b1c30;
  --popover: #ffffff;
  --popover-foreground: #0b1c30;
  --primary: #3525cd;
  --primary-foreground: #ffffff;
  --secondary: #006591;
  --secondary-foreground: #ffffff;
  --muted: #e5eeff;
  --muted-foreground: #464555;
  --accent: #dce9ff;
  --accent-foreground: #0b1c30;
  --destructive: #ba1a1a;
  --border: #c7c4d8;
  --input: #c7c4d8;
  --ring: #3525cd;
  --chart-1: #3525cd;
  --chart-2: #39b8fd;
  --chart-3: #4f46e5;
  --chart-4: #006591;
  --chart-5: #777587;
  --radius: 0.75rem;
  --sidebar: #ffffff;
  --sidebar-foreground: #0b1c30;
  --sidebar-primary: #3525cd;
  --sidebar-primary-foreground: #ffffff;
  --sidebar-accent: #dce9ff;
  --sidebar-accent-foreground: #0b1c30;
  --sidebar-border: #c7c4d8;
  --sidebar-ring: #3525cd;
}
```

- [ ] **Step 2: Replace the `.dark` color block**

Replace the entire `.dark { ... }` block with:

```css
.dark {
  --background: #0b1c30;
  --foreground: #eaf1ff;
  --card: #213145;
  --card-foreground: #eaf1ff;
  --popover: #213145;
  --popover-foreground: #eaf1ff;
  --primary: #c3c0ff;
  --primary-foreground: #0f0069;
  --secondary: #89ceff;
  --secondary-foreground: #001e2f;
  --muted: color-mix(in oklch, #213145, white 8%);
  --muted-foreground: #c7c4d8;
  --accent: color-mix(in oklch, #213145, white 12%);
  --accent-foreground: #eaf1ff;
  --destructive: color-mix(in oklch, #ba1a1a, white 25%);
  --border: oklch(1 0 0 / 10%);
  --input: oklch(1 0 0 / 15%);
  --ring: #c3c0ff;
  --chart-1: #3525cd;
  --chart-2: #39b8fd;
  --chart-3: #4f46e5;
  --chart-4: #006591;
  --chart-5: #777587;
  --sidebar: #213145;
  --sidebar-foreground: #eaf1ff;
  --sidebar-primary: #c3c0ff;
  --sidebar-primary-foreground: #0f0069;
  --sidebar-accent: color-mix(in oklch, #213145, white 12%);
  --sidebar-accent-foreground: #eaf1ff;
  --sidebar-border: oklch(1 0 0 / 10%);
  --sidebar-ring: #c3c0ff;
}
```

Note: `--chart-1..5` are deliberately identical in light and dark — they're already saturated/vivid enough to read on both backgrounds (Recharts series colors), so a second speculative dark chart palette isn't worth maintaining.

- [ ] **Step 3: Fix the self-referential font variables**

In the `@theme inline { ... }` block (top of the same file), change:

```css
  --font-sans: var(--font-sans);
```

to:

```css
  --font-sans: var(--font-inter);
```

and change:

```css
  --font-mono: var(--font-geist-mono);
```

to:

```css
  --font-mono: var(--font-jetbrains-mono);
```

- [ ] **Step 4: Swap Geist for Inter/JetBrains Mono**

In `frontend/app/layout.tsx`, replace:

```tsx
import { Geist, Geist_Mono } from "next/font/google";
```

```tsx
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});
```

with:

```tsx
import { Inter, JetBrains_Mono } from "next/font/google";
```

```tsx
const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
});
```

Then update the `<html>` element's `className` from:

```tsx
className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
```

to:

```tsx
className={`${inter.variable} ${jetbrainsMono.variable} h-full antialiased`}
```

- [ ] **Step 5: Verify nothing broke**

Run: `cd frontend && pnpm test`
Expected: all existing tests still PASS (this task only changes CSS variable values and font source — no component markup changed).

Run: `pnpm build`
Expected: build succeeds (confirms `next/font/google` resolves `Inter`/`JetBrains_Mono` and there are no CSS syntax errors).

- [ ] **Step 6: Commit**

```bash
cd frontend
git add app/globals.css app/layout.tsx
git commit -m "feat(frontend): apply fase5 design tokens (colors, fonts, radius)"
```

---

### Task 2: Nav shell (`AppSidebar`)

**Files:**
- Create: `frontend/components/app-sidebar.tsx`
- Create: `frontend/components/app-sidebar.test.tsx`
- Modify: `frontend/app/(app)/layout.tsx`

**Interfaces:**
- Consumes: `useAuth()` from `@/lib/auth-context` (`{ signOut: () => Promise<void> }`, already exists).
- Produces: `AppSidebar` component (no props) — rendered by `app/(app)/layout.tsx`, wraps nothing itself (siblings, not a wrapper).

- [ ] **Step 1: Write the failing test**

Create `frontend/components/app-sidebar.test.tsx`:

```tsx
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && pnpm vitest run components/app-sidebar.test.tsx`
Expected: FAIL with "Cannot find module './app-sidebar'"

- [ ] **Step 3: Write the implementation**

Create `frontend/components/app-sidebar.tsx`:

```tsx
"use client";

import { Button } from "@/components/ui/button";
import { useAuth } from "@/lib/auth-context";
import { cn } from "@/lib/utils";
import { Gauge, GitCompareArrows, Layers, LogOut, Zap } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/runs", label: "Ejecuciones", icon: Gauge },
  { href: "/scenarios", label: "Escenarios", icon: Layers },
  { href: "/compare", label: "Comparar", icon: GitCompareArrows },
] as const;

export function AppSidebar() {
  const pathname = usePathname();
  const { signOut } = useAuth();

  return (
    <aside className="fixed inset-y-0 left-0 hidden w-60 flex-col border-r border-border bg-card px-4 py-6 md:flex">
      <div className="mb-8 flex items-center gap-2">
        <div className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
          <Zap className="size-4" />
        </div>
        <div>
          <p className="font-heading text-sm font-bold leading-tight">Despacho-UDEA</p>
          <p className="text-xs text-muted-foreground">Technical Dispatch Modeler</p>
        </div>
      </div>
      <nav className="flex flex-1 flex-col gap-1">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = pathname?.startsWith(href) ?? false;
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                active
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )}
            >
              <Icon className="size-4" />
              {label}
            </Link>
          );
        })}
      </nav>
      <Button variant="ghost" className="justify-start gap-2" onClick={() => signOut()}>
        <LogOut className="size-4" />
        Salir
      </Button>
    </aside>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm vitest run components/app-sidebar.test.tsx`
Expected: PASS (3 tests)

- [ ] **Step 5: Wire it into the authenticated layout**

Replace the full contents of `frontend/app/(app)/layout.tsx` with:

```tsx
"use client";

import { AppSidebar } from "@/components/app-sidebar";
import { RequireAuth } from "@/components/require-auth";
import type { ReactNode } from "react";

export default function AppLayout({ children }: { children: ReactNode }) {
  return (
    <RequireAuth>
      <div className="min-h-screen bg-background">
        <AppSidebar />
        <main className="md:pl-60">
          <div className="mx-auto max-w-6xl px-6 py-8">{children}</div>
        </main>
      </div>
    </RequireAuth>
  );
}
```

- [ ] **Step 6: Run the full test suite**

Run: `pnpm test`
Expected: all PASS — `(app)/layout.tsx` has no existing test file, so nothing else references its old markup.

- [ ] **Step 7: Commit**

```bash
git add components/app-sidebar.tsx components/app-sidebar.test.tsx "app/(app)/layout.tsx"
git commit -m "feat(frontend): add styled nav shell (AppSidebar)"
```

---

### Task 3: Login page — reskin + real forgot-password + signup link

**Files:**
- Modify: `frontend/app/login/page.tsx`
- Modify: `frontend/app/login/page.test.tsx`

**Interfaces:**
- Consumes: `supabase.auth.signInWithPassword` (existing), `supabase.auth.resetPasswordForEmail` (Supabase SDK method, not previously called anywhere in this codebase).
- Produces: nothing new consumed by later tasks (leaf screen).

- [ ] **Step 1: Update the mock in the existing test file to add `resetPasswordForEmail`**

In `frontend/app/login/page.test.tsx`, change the `vi.mock("@/lib/supabase", ...)` call from:

```tsx
vi.mock("@/lib/supabase", () => ({
  supabase: { auth: { signInWithPassword: vi.fn() } },
}));
```

to:

```tsx
vi.mock("@/lib/supabase", () => ({
  supabase: { auth: { signInWithPassword: vi.fn(), resetPasswordForEmail: vi.fn() } },
}));
```

- [ ] **Step 2: Add the new failing tests**

Append to the `describe("LoginPage", ...)` block in the same file:

```tsx
  it("sends a password reset email when the forgot-password link is clicked", async () => {
    vi.mocked(supabase.auth.resetPasswordForEmail).mockResolvedValue({
      data: {},
      error: null,
    } as never);

    render(<LoginPage />);
    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: "a@b.com" } });
    fireEvent.click(screen.getByRole("button", { name: /olvidaste tu contrasena/i }));

    await waitFor(() =>
      expect(supabase.auth.resetPasswordForEmail).toHaveBeenCalledWith(
        "a@b.com",
        expect.objectContaining({ redirectTo: expect.stringContaining("/reset-password") })
      )
    );
    await waitFor(() => screen.getByText(/revisa tu correo/i));
  });

  it("shows a validation message when requesting a reset with no email", async () => {
    render(<LoginPage />);
    fireEvent.click(screen.getByRole("button", { name: /olvidaste tu contrasena/i }));

    await waitFor(() => screen.getByText(/ingresa tu correo/i));
    expect(supabase.auth.resetPasswordForEmail).not.toHaveBeenCalled();
  });

  it("links to /signup", () => {
    render(<LoginPage />);
    expect(screen.getByRole("link", { name: /crear cuenta/i })).toHaveAttribute("href", "/signup");
  });
```

- [ ] **Step 3: Run tests to verify the new ones fail**

Run: `pnpm vitest run app/login/page.test.tsx`
Expected: the 2 original tests still PASS (button text "Entrar" and label "Contrasena" are unchanged in Step 4 below); the 3 new tests FAIL (elements don't exist yet).

- [ ] **Step 4: Rewrite the page**

Replace the full contents of `frontend/app/login/page.tsx` with:

```tsx
"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { supabase } from "@/lib/supabase";
import { Lock, Mail, Zap } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [resetMessage, setResetMessage] = useState<string | null>(null);
  const router = useRouter();

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) {
      setError(error.message);
      return;
    }
    router.push("/runs");
  }

  async function handleForgotPassword() {
    setError(null);
    setResetMessage(null);
    if (!email) {
      setError("Ingresa tu correo primero.");
      return;
    }
    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/reset-password`,
    });
    if (error) {
      setError(error.message);
      return;
    }
    setResetMessage("Revisa tu correo para restablecer tu contrasena.");
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="w-full max-w-sm rounded-xl border border-border bg-card p-8 text-card-foreground shadow-sm">
        <div className="mb-6 flex flex-col items-center text-center">
          <div className="mb-3 flex size-12 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <Zap className="size-6" />
          </div>
          <h1 className="font-heading text-xl font-bold">Despacho-UDEA</h1>
          <p className="text-sm text-muted-foreground">Technical Dispatch Modeler</p>
        </div>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="email">Email</Label>
            <div className="relative">
              <Mail className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="pl-8"
              />
            </div>
          </div>
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between">
              <Label htmlFor="password">Contrasena</Label>
              <button
                type="button"
                onClick={handleForgotPassword}
                className="text-xs font-medium text-primary hover:underline"
              >
                Olvidaste tu contrasena?
              </button>
            </div>
            <div className="relative">
              <Lock className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="pl-8"
              />
            </div>
          </div>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          {resetMessage && <p className="text-sm text-muted-foreground">{resetMessage}</p>}
          <Button type="submit">Entrar</Button>
        </form>
        <p className="mt-6 text-center text-sm text-muted-foreground">
          No tienes cuenta?{" "}
          <Link href="/signup" className="font-medium text-primary hover:underline">
            Crear cuenta
          </Link>
        </p>
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pnpm vitest run app/login/page.test.tsx`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add app/login/page.tsx app/login/page.test.tsx
git commit -m "feat(frontend): reskin login + real forgot-password flow"
```

---

### Task 4: Reset-password page (new route)

**Files:**
- Create: `frontend/app/reset-password/page.tsx`
- Create: `frontend/app/reset-password/page.test.tsx`

**Interfaces:**
- Consumes: `supabase.auth.updateUser({ password })` (Supabase SDK method).
- Produces: nothing consumed by later tasks (leaf screen, only reachable via the emailed recovery link built in Task 3).

- [ ] **Step 1: Write the failing test**

Create `frontend/app/reset-password/page.test.tsx`:

```tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ResetPasswordPage from "./page";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

vi.mock("@/lib/supabase", () => ({
  supabase: { auth: { updateUser: vi.fn() } },
}));

import { supabase } from "@/lib/supabase";

describe("ResetPasswordPage", () => {
  it("updates the password and redirects to /login on success", async () => {
    vi.mocked(supabase.auth.updateUser).mockResolvedValue({ error: null } as never);

    render(<ResetPasswordPage />);
    fireEvent.change(screen.getByLabelText(/nueva contrasena/i), { target: { value: "secret123" } });
    fireEvent.change(screen.getByLabelText(/confirmar contrasena/i), { target: { value: "secret123" } });
    fireEvent.click(screen.getByRole("button", { name: /guardar/i }));

    await waitFor(() => expect(supabase.auth.updateUser).toHaveBeenCalledWith({ password: "secret123" }));
    await waitFor(() => expect(push).toHaveBeenCalledWith("/login"));
  });

  it("shows an error when the passwords do not match", async () => {
    render(<ResetPasswordPage />);
    fireEvent.change(screen.getByLabelText(/nueva contrasena/i), { target: { value: "secret123" } });
    fireEvent.change(screen.getByLabelText(/confirmar contrasena/i), { target: { value: "different" } });
    fireEvent.click(screen.getByRole("button", { name: /guardar/i }));

    await waitFor(() => screen.getByText(/no coinciden/i));
    expect(supabase.auth.updateUser).not.toHaveBeenCalled();
  });

  it("shows the server error message on failure", async () => {
    vi.mocked(supabase.auth.updateUser).mockResolvedValue({
      error: { message: "Auth session missing" },
    } as never);

    render(<ResetPasswordPage />);
    fireEvent.change(screen.getByLabelText(/nueva contrasena/i), { target: { value: "secret123" } });
    fireEvent.change(screen.getByLabelText(/confirmar contrasena/i), { target: { value: "secret123" } });
    fireEvent.click(screen.getByRole("button", { name: /guardar/i }));

    await waitFor(() => screen.getByText("Auth session missing"));
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm vitest run app/reset-password/page.test.tsx`
Expected: FAIL with "Cannot find module './page'"

- [ ] **Step 3: Write the implementation**

Create `frontend/app/reset-password/page.tsx`:

```tsx
"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { supabase } from "@/lib/supabase";
import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

export default function ResetPasswordPage() {
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    if (password !== confirmPassword) {
      setError("Las contrasenas no coinciden.");
      return;
    }
    const { error } = await supabase.auth.updateUser({ password });
    if (error) {
      setError(error.message);
      return;
    }
    router.push("/login");
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="w-full max-w-sm rounded-xl border border-border bg-card p-8 text-card-foreground shadow-sm">
        <h1 className="mb-6 font-heading text-xl font-bold">Restablecer contrasena</h1>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="password">Nueva contrasena</Label>
            <Input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={6}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="confirm_password">Confirmar contrasena</Label>
            <Input
              id="confirm_password"
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              minLength={6}
            />
          </div>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <Button type="submit">Guardar contrasena</Button>
        </form>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm vitest run app/reset-password/page.test.tsx`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/reset-password/page.tsx app/reset-password/page.test.tsx
git commit -m "feat(frontend): add /reset-password route for real password recovery"
```

---

### Task 5: Signup page — reskin + Confirm Password + login link

**Files:**
- Modify: `frontend/app/signup/page.tsx`
- Modify: `frontend/app/signup/page.test.tsx`

**Interfaces:**
- Consumes: `supabase.auth.signUp` (existing, unchanged signature).
- Produces: nothing consumed by later tasks (leaf screen).

- [ ] **Step 1: Rewrite the test file**

`getByLabelText(/contrase/i)` in the existing test will become ambiguous once a "Confirmar contrasena" field exists (both labels match `/contrase/i`). Replace the full contents of `frontend/app/signup/page.test.tsx` with:

```tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import SignupPage from "./page";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

vi.mock("@/lib/supabase", () => ({
  supabase: { auth: { signUp: vi.fn() } },
}));

import { supabase } from "@/lib/supabase";

describe("SignupPage", () => {
  it("signs up and redirects to /login on success", async () => {
    vi.mocked(supabase.auth.signUp).mockResolvedValue({ error: null } as never);

    render(<SignupPage />);
    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: "a@b.com" } });
    fireEvent.change(screen.getByLabelText("Contrasena"), { target: { value: "secret123" } });
    fireEvent.change(screen.getByLabelText(/confirmar contrasena/i), { target: { value: "secret123" } });
    fireEvent.click(screen.getByRole("button", { name: /crear cuenta/i }));

    await waitFor(() => expect(push).toHaveBeenCalledWith("/login"));
  });

  it("shows an error when the passwords do not match", async () => {
    render(<SignupPage />);
    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: "a@b.com" } });
    fireEvent.change(screen.getByLabelText("Contrasena"), { target: { value: "secret123" } });
    fireEvent.change(screen.getByLabelText(/confirmar contrasena/i), { target: { value: "different" } });
    fireEvent.click(screen.getByRole("button", { name: /crear cuenta/i }));

    await waitFor(() => screen.getByText(/no coinciden/i));
    expect(supabase.auth.signUp).not.toHaveBeenCalled();
  });

  it("links to /login", () => {
    render(<SignupPage />);
    expect(screen.getByRole("link", { name: /iniciar sesion/i })).toHaveAttribute("href", "/login");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm vitest run app/signup/page.test.tsx`
Expected: FAIL (Confirm Password field, matching error text, and login link don't exist yet).

- [ ] **Step 3: Rewrite the page**

Replace the full contents of `frontend/app/signup/page.tsx` with:

```tsx
"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { supabase } from "@/lib/supabase";
import { Zap } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

export default function SignupPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    if (password !== confirmPassword) {
      setError("Las contrasenas no coinciden.");
      return;
    }
    const { error } = await supabase.auth.signUp({ email, password });
    if (error) {
      setError(error.message);
      return;
    }
    router.push("/login");
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="w-full max-w-sm rounded-xl border border-border bg-card p-8 text-card-foreground shadow-sm">
        <div className="mb-6 flex flex-col items-center text-center">
          <div className="mb-3 flex size-12 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <Zap className="size-6" />
          </div>
          <h1 className="font-heading text-xl font-bold">Despacho-UDEA</h1>
          <p className="text-sm text-muted-foreground">Technical Dispatch Modeler</p>
        </div>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="password">Contrasena</Label>
            <Input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={6}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="confirm_password">Confirmar contrasena</Label>
            <Input
              id="confirm_password"
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              minLength={6}
            />
          </div>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <Button type="submit">Crear cuenta</Button>
        </form>
        <p className="mt-6 text-center text-sm text-muted-foreground">
          Ya tienes cuenta?{" "}
          <Link href="/login" className="font-medium text-primary hover:underline">
            Iniciar sesion
          </Link>
        </p>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm vitest run app/signup/page.test.tsx`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/signup/page.tsx app/signup/page.test.tsx
git commit -m "feat(frontend): reskin signup, add Confirm Password + login link"
```

---

### Task 6: Runs list — status labels/badges + duration + table reskin

**Files:**
- Modify: `frontend/lib/run-status.ts`
- Modify: `frontend/lib/run-status.test.ts`
- Modify: `frontend/lib/format-date.ts`
- Modify: `frontend/lib/format-date.test.ts`
- Modify: `frontend/components/runs-table.tsx`
- Modify: `frontend/components/runs-table.test.tsx`
- Modify: `frontend/app/(app)/runs/page.tsx`

**Interfaces:**
- Produces: `statusLabel(status: RunStatus): string`, `statusBadgeVariant(status: RunStatus): "default" | "secondary" | "outline" | "destructive"` (from `lib/run-status.ts`) and `formatDuration(startedAt: string | null, finishedAt: string | null): string` (from `lib/format-date.ts`) — both will be reused by the run-detail screen in fase5b.

- [ ] **Step 1: Write the failing tests for the new lib functions**

Append to `frontend/lib/run-status.test.ts`:

```tsx
import { statusBadgeVariant, statusLabel } from "./run-status";

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
```

Append to `frontend/lib/format-date.test.ts`:

```tsx
import { formatDuration } from "./format-date";

describe("formatDuration", () => {
  it("formats the difference between started_at and finished_at as Xm Ys", () => {
    expect(formatDuration("2024-04-18T05:00:00Z", "2024-04-18T05:04:12Z")).toBe("4m 12s");
  });

  it("returns a dash when either timestamp is missing", () => {
    expect(formatDuration(null, "2024-04-18T05:04:12Z")).toBe("--");
    expect(formatDuration("2024-04-18T05:00:00Z", null)).toBe("--");
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm vitest run lib/run-status.test.ts lib/format-date.test.ts`
Expected: FAIL — `statusLabel`, `statusBadgeVariant`, `formatDuration` are not exported yet.

- [ ] **Step 3: Implement the lib functions**

Append to `frontend/lib/run-status.ts` (keep the existing `isTerminalStatus` export):

```ts
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
```

Append to `frontend/lib/format-date.ts`:

```ts
export function formatDuration(startedAt: string | null, finishedAt: string | null): string {
  if (!startedAt || !finishedAt) return "--";
  const ms = new Date(finishedAt).getTime() - new Date(startedAt).getTime();
  const totalSeconds = Math.max(0, Math.round(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}m ${seconds}s`;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm vitest run lib/run-status.test.ts lib/format-date.test.ts`
Expected: PASS

- [ ] **Step 5: Update the RunsTable test for the new badge text and duration column**

Replace the full contents of `frontend/components/runs-table.test.tsx` with:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RunsTable } from "./runs-table";
import type { RunSummary } from "@/lib/types";

const runs: RunSummary[] = [
  {
    run_id: "r1",
    status: "done",
    dispatch_date: "2024-04-18",
    level: "preideal",
    scenario_id: null,
    created_at: "2024-04-18T05:00:00Z",
    started_at: "2024-04-18T05:00:00Z",
    finished_at: "2024-04-18T05:04:12Z",
    error: null,
  },
];

describe("RunsTable", () => {
  it("renders one row per run with date/level/status/duration", () => {
    render(<RunsTable runs={runs} />);
    expect(screen.getByText("2024-04-18")).toBeInTheDocument();
    expect(screen.getByText("preideal")).toBeInTheDocument();
    expect(screen.getByText("Completado")).toBeInTheDocument();
    expect(screen.getByText("4m 12s")).toBeInTheDocument();
  });

  it("shows a dash for duration when the run has not finished", () => {
    const running: RunSummary[] = [
      { ...runs[0], run_id: "r2", status: "running", finished_at: null },
    ];
    render(<RunsTable runs={running} />);
    expect(screen.getByText("--")).toBeInTheDocument();
  });

  it("renders an empty state with no runs", () => {
    render(<RunsTable runs={[]} />);
    expect(screen.getByText(/sin ejecuciones/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pnpm vitest run components/runs-table.test.tsx`
Expected: FAIL — "Completado"/"4m 12s" not rendered yet (current code renders raw `run.status` and has no duration column).

- [ ] **Step 7: Rewrite RunsTable**

Replace the full contents of `frontend/components/runs-table.tsx` with:

```tsx
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { formatBogotaTime, formatDuration } from "@/lib/format-date";
import { statusBadgeVariant, statusLabel } from "@/lib/run-status";
import type { RunSummary } from "@/lib/types";
import Link from "next/link";

export function RunsTable({ runs }: { runs: RunSummary[] }) {
  if (runs.length === 0) {
    return <p className="p-6 text-sm text-muted-foreground">Sin ejecuciones todavia.</p>;
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Fecha</TableHead>
          <TableHead>Nivel</TableHead>
          <TableHead>Estado</TableHead>
          <TableHead>Creado</TableHead>
          <TableHead>Duracion</TableHead>
          <TableHead />
        </TableRow>
      </TableHeader>
      <TableBody>
        {runs.map((run) => (
          <TableRow key={run.run_id}>
            <TableCell className="font-mono">{run.dispatch_date}</TableCell>
            <TableCell>{run.level}</TableCell>
            <TableCell>
              <Badge variant={statusBadgeVariant(run.status)}>{statusLabel(run.status)}</Badge>
            </TableCell>
            <TableCell className="font-mono text-muted-foreground">
              {formatBogotaTime(run.created_at)}
            </TableCell>
            <TableCell className="font-mono text-muted-foreground">
              {formatDuration(run.started_at, run.finished_at)}
            </TableCell>
            <TableCell>
              <Link
                href={`/runs/${run.run_id}`}
                className="text-sm font-medium text-primary hover:underline"
              >
                Ver
              </Link>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pnpm vitest run components/runs-table.test.tsx`
Expected: PASS (3 tests)

- [ ] **Step 9: Reskin the page chrome**

Replace the full contents of `frontend/app/(app)/runs/page.tsx` with:

```tsx
"use client";

import { CreateRunForm } from "@/components/create-run-form";
import { RunsTable } from "@/components/runs-table";
import { listRuns } from "@/lib/api-client";
import { useQuery, useQueryClient } from "@tanstack/react-query";

export default function RunsPage() {
  const queryClient = useQueryClient();
  const runsQuery = useQuery({ queryKey: ["runs"], queryFn: listRuns });

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="font-heading text-2xl font-bold">Ejecuciones</h1>
        <p className="text-sm text-muted-foreground">Historial de modelado de despacho</p>
      </div>
      <div className="rounded-xl border border-border bg-card p-6">
        <CreateRunForm onCreated={() => queryClient.invalidateQueries({ queryKey: ["runs"] })} />
      </div>
      <div className="rounded-xl border border-border bg-card">
        {runsQuery.isLoading && <p className="p-6 text-sm text-muted-foreground">Cargando...</p>}
        {runsQuery.data && <RunsTable runs={runsQuery.data} />}
      </div>
    </div>
  );
}
```

(No test file exists for this page — it has no branching logic beyond what `RunsTable`/`CreateRunForm` already cover.)

- [ ] **Step 10: Run the full test suite**

Run: `pnpm test`
Expected: all PASS

- [ ] **Step 11: Commit**

```bash
git add lib/run-status.ts lib/run-status.test.ts lib/format-date.ts lib/format-date.test.ts \
  components/runs-table.tsx components/runs-table.test.tsx "app/(app)/runs/page.tsx"
git commit -m "feat(frontend): reskin Ejecuciones list, add status labels + duration"
```

---

### Task 7: Nueva Ejecucion — solver picker (CBC / HiGHS-disabled) + reskin

**Files:**
- Modify: `frontend/components/create-run-form.tsx`
- Modify: `frontend/components/create-run-form.test.tsx`

**Interfaces:**
- Consumes: `CreateRunRequest` from `@/lib/types` (already has optional `solver?: string` — no type change needed).

- [ ] **Step 1: Add the new failing tests**

Append to the `describe("CreateRunForm", ...)` block in `frontend/components/create-run-form.test.tsx`:

```tsx
  it("submits with the default solver (cbc)", async () => {
    renderWithQueryClient(<CreateRunForm onCreated={vi.fn()} />);

    fireEvent.change(screen.getByLabelText(/fecha/i), { target: { value: "2024-04-18" } });
    fireEvent.click(screen.getByRole("button", { name: /crear/i }));

    await waitFor(() =>
      expect(createRun).toHaveBeenCalledWith(expect.objectContaining({ solver: "cbc" }))
    );
  });

  it("renders the HiGHS solver option as disabled", () => {
    renderWithQueryClient(<CreateRunForm onCreated={vi.fn()} />);
    const highsOption = screen.getByRole("option", { name: /highs/i }) as HTMLOptionElement;
    expect(highsOption.disabled).toBe(true);
  });
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `pnpm vitest run components/create-run-form.test.tsx`
Expected: the 1 original test still PASSES; the 2 new tests FAIL (no solver field exists yet).

- [ ] **Step 3: Rewrite the form**

Replace the full contents of `frontend/components/create-run-form.tsx` with:

```tsx
"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { createRun, listScenarios } from "@/lib/api-client";
import type { CreateRunRequest, DispatchLevel } from "@/lib/types";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";

const SELECT_CLASS =
  "h-8 rounded-lg border border-input bg-transparent px-2.5 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50";

export function CreateRunForm({ onCreated }: { onCreated: () => void }) {
  const [dispatchDate, setDispatchDate] = useState("");
  const [level, setLevel] = useState<DispatchLevel>("preideal");
  const [solver, setSolver] = useState("cbc");
  const [scenarioId, setScenarioId] = useState("");

  const scenariosQuery = useQuery({ queryKey: ["scenarios"], queryFn: listScenarios });
  const mutation = useMutation({
    mutationFn: (variables: CreateRunRequest) => createRun(variables),
    onSuccess: onCreated,
  });

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    mutation.mutate({
      dispatch_date: dispatchDate,
      level,
      solver,
      scenario_id: scenarioId || null,
    });
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-wrap items-end gap-4">
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="dispatch_date">Fecha</Label>
        <Input
          id="dispatch_date"
          type="date"
          value={dispatchDate}
          onChange={(e) => setDispatchDate(e.target.value)}
          required
        />
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="level">Nivel</Label>
        <select
          id="level"
          value={level}
          onChange={(e) => setLevel(e.target.value as DispatchLevel)}
          className={SELECT_CLASS}
        >
          <option value="preideal">preideal</option>
          <option value="ideal">ideal</option>
        </select>
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="solver">Solver</Label>
        <select
          id="solver"
          value={solver}
          onChange={(e) => setSolver(e.target.value)}
          className={SELECT_CLASS}
        >
          <option value="cbc">CBC</option>
          <option value="highs" disabled>
            HiGHS (proximamente)
          </option>
        </select>
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="scenario_id">Escenario BESS (opcional)</Label>
        <select
          id="scenario_id"
          value={scenarioId}
          onChange={(e) => setScenarioId(e.target.value)}
          className={SELECT_CLASS}
        >
          <option value="">Ninguno</option>
          {(scenariosQuery.data ?? []).map((s) => (
            <option key={s.id} value={s.id}>
              {s.penetration_level} ({s.mode})
            </option>
          ))}
        </select>
      </div>
      <Button type="submit" disabled={mutation.isPending}>
        Crear ejecucion
      </Button>
      {mutation.isError && (
        <p role="alert" className="w-full text-sm text-destructive">
          {(mutation.error as Error).message}
        </p>
      )}
    </form>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm vitest run components/create-run-form.test.tsx`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full test suite**

Run: `pnpm test`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add components/create-run-form.tsx components/create-run-form.test.tsx
git commit -m "feat(frontend): reskin Nueva Ejecucion, add solver picker (CBC/HiGHS)"
```

---

### Task 8: Manual verification in a real browser

Automated tests don't catch visual regressions, dark-mode contrast, or auth redirects — this task is the one the system prompt requires before calling UI work done.

**Files:** none (verification only).

- [ ] **Step 1: Start the dev server**

Run: `cd frontend && pnpm dev`

Requires `frontend/.env.local` with real `NEXT_PUBLIC_SUPABASE_URL` / `NEXT_PUBLIC_SUPABASE_ANON_KEY` (see `frontend/.env.local.example` and the `supabase-docker-local-testing` notes) and the API running (`docker compose -f docker/docker-compose.dev.yaml up`, per `docker/Dockerfile.api`, now that Task-preceding `services/` restore is in place).

- [ ] **Step 2: Walk the golden path**

Navigate to `http://localhost:3000`:
1. `/login` — confirm Inter font, indigo primary button, forgot-password link triggers a message (check Supabase Auth logs / inbox for the email), "Crear cuenta" link goes to `/signup`.
2. `/signup` — confirm Confirm Password field, mismatched-password error, submit redirects to `/login`.
3. Log in with a real account, confirm redirect to `/runs`.
4. `/runs` — confirm sidebar renders with "Ejecuciones" highlighted, table shows Spanish status badges and a duration column, "Nueva Ejecucion" form shows CBC selected and HiGHS greyed out/unselectable.
5. Toggle OS/browser dark mode (or manually add `class="dark"` to `<html>` via devtools) and repeat steps 1-4 — confirm text stays legible (no dark-on-dark or invisible borders) on every screen.

- [ ] **Step 3: Check the browser console**

Confirm no React hydration warnings or uncaught errors on any of the 4 routes above.

- [ ] **Step 4: Report findings**

If everything above holds, fase5a is done. If something looks wrong, fix it in the relevant task's files and re-run that task's test file before moving on — do not silently patch without updating the corresponding test if the fix changes rendered text.

---

## Self-Review Notes

- **Spec coverage**: tokens (Task 1), nav shell (Task 2), login+forgot-password (Task 3), reset-password (Task 4), signup (Task 5), runs list+status+duration (Task 6), solver picker (Task 7) — all fase5a spec items from `docs/superpowers/specs/2026-08-06-fase5-frontend-visual-design-design.md` are covered. Run detail, Escenarios, Comparar are fase5b (separate plan, separate branch).
- **Placeholder scan**: no TBD/TODO; every step has literal code, not descriptions.
- **Type consistency**: `RunStatus`/`DispatchLevel`/`RunSummary`/`CreateRunRequest` used exactly as defined in `frontend/lib/types.ts` (verified against the real file, not assumed) across Tasks 6 and 7. `statusLabel`/`statusBadgeVariant`/`formatDuration` signatures introduced in Task 6 are the only new shared interfaces, and no later task in this plan calls them with different signatures.
- **Existing-test risk audited per task**: Task 3 (login) — original 2 tests keep passing unmodified (button text "Entrar" and label "Contrasena" both untouched); Task 5 (signup) — original test's `getByLabelText(/contrase/i)` would break on ambiguity, so the whole file is replaced, not patched; Task 6 (runs-table) — original `getByText("done")` assertion would fail against the new "Completado" badge, so the whole file is replaced; Task 7 (create-run-form) — original test's `objectContaining` assertion is unaffected by the added `solver` field, kept as-is, only new tests appended.
