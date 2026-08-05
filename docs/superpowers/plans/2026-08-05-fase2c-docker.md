# Fase 2C: Dockerfile + docker-compose + smoke test en contenedor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package `despacho-udea` as a Docker image with the CBC solver, add a `docker-compose.yml` with a working `cli` service and empty `future`-profiled placeholders for the not-yet-built `api`/`worker` services, and prove the Fase 2B XM fixture solves cleanly inside the container — the same command, same result, as it already does on the host.

**Architecture:** Single-stage `python:3.12-slim` image. `uv` binary copied in from the official `ghcr.io/astral-sh/uv` image (no pip bootstrap needed). `uv sync --no-dev --frozen` installs only the base `[project.dependencies]` (no `dev` group, no `notebooks` extra) — this is already "runtime" in this repo's `pyproject.toml`, nothing to restructure there. `coinor-cbc` installed via `apt-get`, same package the host dev environment already uses. `data/` is never baked into the image (git-ignored, mounted as a bind volume at runtime); `tests/fixtures/xm_smoke/` **is** version-controlled and gets `COPY`'d explicitly, since it's the smoke-test payload.

**Tech Stack:** Docker, Docker Compose (Compose Spec, no `version:` key), `uv` 0.11.15, `python:3.12-slim`, `coinor-cbc`.

## Global Constraints

- Branch for this work is `fase2c-docker`, based on `origin/develop` (NOT `origin/main` — verify the base ref before/after creating the worktree; PR #4 (Fase 2B) and the `chore/agents-rules` PR #5 are both already merged into `develop`, so `develop` already has the fixture and `.agents/rules/`).
- Solver default stays `cbc`. Do not add `appsi_highs` as a usable alternative anywhere in this plan (see `docs/superpowers/specs/2026-08-05-fase2-docker-design.md` section 3 — the legacy Pyomo wrapper crashes on every solve; `highspy` still gets installed as a dependency, it's just not documented as usable).
- The image must never `COPY` `data/` (git-ignored, may contain a developer's real local downloads).
- No PR co-authorship / "Generated with" lines in any commit message (repo-wide rule, `AGENTS.md`).
- No unrequested abstractions in `Dockerfile`/`docker-compose.yml` — no multi-stage build (not needed at this size), no `version:` key (deprecated in Compose Spec), no build args that aren't used.

---

### Task 0: Carry forward the pending spec correction

There's an uncommitted correction sitting in the working tree of the current session against `docs/superpowers/specs/2026-08-05-fase2-docker-design.md` (verified during this plan's research, not part of Fase 2C itself — it corrects the OFEI parsing description in section 2 from "fixed-width" to "comma-separated, substring-matched, no explicit encoding"). It must land as its own attributable commit, not get silently folded into a Docker commit or lost when a fresh worktree is created from `develop` (a new worktree checks out `develop` cleanly — it does **not** inherit uncommitted changes from another checkout).

**Files:**
- Modify: `docs/superpowers/specs/2026-08-05-fase2-docker-design.md` (section 2, "Que necesita el fixture, verificado contra el codigo real")

**Interfaces:** None — documentation only, no code.

- [ ] **Step 1: Apply the correction**

Replace this paragraph in section 2:

```markdown
- `data/oferta_inicial/OFEI{MMDD}.txt` — formato fixed-width/latin1 propio,
  parseado por `app/data/ofei.py::parse_ofei`. Debe producir
  `precio_arranque` (con al menos una fila `type` conteniendo `"C"` por
  generador termico, o `case_builder.py:322` revienta con `IndexError` al
  indexar `.values[0]` en un resultado vacio), `minimo_operativo`, y puede
  dejar `cc`/`cc_price`/`cc_dispo` vacios (`{}`) sin problema — **verificado
  por ejecucion directa**: `pd.DataFrame({}).stack().reset_index()` seguido
  del resto del bloque de sintesis de CC (case_builder.py:194-214) no
  revienta con diccionarios vacios. Esto significa que el fixture **no
  necesita ningun recurso de ciclo combinado** — 2-3 generadores termicos
  simples bastan, evitando toda la rama `CC_MAP`/`dcondIniPlant` (y con
  ella, los tres mapeos de nombres hardcodeados en `case_builder.py`
  ["FLORES IV", "TSIERRA", "GUAJIR21"], que son fallbacks `.get(x, x)` sin
  efecto si esos nombres no aparecen en los datos — no hay ninguna razon
  para reproducirlos en un fixture sintetico).
```

with:

```markdown
- `data/oferta_inicial/OFEI{MMDD}.txt` — texto plano separado por comas,
  parseado linea por linea con matching de substrings (`app/data/
  ofei.py::parse_ofei`, no es fixed-width), abierto sin encoding explicito
  (`open(path, "r")` — locale por defecto, utf-8 en este entorno; distinto
  de `PrId`, que si usa `encoding="latin1"` explicito — no asumir el mismo
  encoding para los dos archivos, son formatos y aperturas independientes).
  Lineas reconocidas por contenido: `"PAP" in line` -> precio de arranque
  (`resource,type,price`, filtra `"usd" in line.lower()`); `"MO" in line`
  con `mo_line[1]` conteniendo `"MO"` -> perfil de minimo operativo
  (`resource,type,` + 24 columnas horarias); lineas con patron `P(\d+)` y
  `"CC" in line` -> precio de ciclo combinado; patron `DISCONF(\d+)` y `"CC"
  in line` -> disponibilidad de ciclo combinado; lineas con exactamente 3
  campos, `" P" in campo[1]`, sin `"u"`/`"a"` en `campo[1].lower()` -> precio
  de oferta simple. El fixture debe producir al menos una fila `precio_
  arranque` con `type` conteniendo `"C"` por generador termico (o
  `case_builder.py:322` revienta con `IndexError` al indexar `.values[0]`
  en un resultado vacio) y filas `MO` para `minimo_operativo`. Puede dejar
  `cc`/`cc_price`/`cc_dispo` vacios (`{}`) sin problema — **verificado por
  ejecucion directa**: `pd.DataFrame({}).stack().reset_index()` seguido del
  resto del bloque de sintesis de CC (case_builder.py:194-214) no revienta
  con diccionarios vacios. Esto significa que el fixture **no necesita
  ningun recurso de ciclo combinado** — 2-3 generadores termicos simples
  bastan, evitando toda la rama `CC_MAP`/`dcondIniPlant` (y con ella, los
  tres mapeos de nombres hardcodeados en `case_builder.py` ["FLORES IV",
  "TSIERRA", "GUAJIR21"], que son fallbacks `.get(x, x)` sin efecto si esos
  nombres no aparecen en los datos — no hay ninguna razon para reproducirlos
  en un fixture sintetico).
```

(If the working tree the worktree was created from doesn't have this uncommitted diff — e.g. a different session — skip this task, it's already been applied upstream. Check with `git diff docs/superpowers/specs/2026-08-05-fase2-docker-design.md` first.)

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-08-05-fase2-docker-design.md
git commit -m "docs: correct OFEI parsing description in Fase 2 spec"
```

---

### Task 1: Dockerfile + .dockerignore

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

**Interfaces:**
- Produces: a buildable image tagged `despacho-udea:latest`, entrypoint `["uv", "run", "--no-sync", "python", "-m", "app"]`, default command `["run"]`. Task 2 (`docker-compose.yml`) and Task 3 (smoke test) both build from this Dockerfile and invoke the image the same way.

- [ ] **Step 1: Write `.dockerignore`**

```
.venv
__pycache__
*.pyc
.git
.pytest_cache
.ruff_cache
.worktrees
data
*.ipynb
```

- [ ] **Step 2: Write `Dockerfile`**

```dockerfile
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends coinor-cbc \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY app ./app
COPY tests/fixtures/xm_smoke ./tests/fixtures/xm_smoke

ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "app"]
CMD ["run"]
```

`--no-dev` is required, not optional: `dev` is configured as a default `dependency-group` in `pyproject.toml`, so a bare `uv sync` would silently pull `pytest`/`ruff`/`ty` into the image. `--frozen` refuses to touch `uv.lock`. `--no-sync` on the `ENTRYPOINT`'s `uv run` stops it from re-resolving/re-syncing the environment (and touching the network) on every container start — the env was already built and frozen at image-build time.

- [ ] **Step 3: Build the image**

Run: `docker build -t despacho-udea:latest .`
Expected: build succeeds, final layer list shows 38 installed packages including `pyomo==6.7.3`, `numpy==1.26.4`, `highspy==1.7.2` — and does **not** include `pytest`, `ruff`, or `ty`.

- [ ] **Step 4: Verify CBC is importable and available inside the image**

Run:
```bash
docker run --rm --entrypoint uv despacho-udea:latest run --no-sync python -c \
  "import pyomo.environ as pyo; print('cbc', pyo.SolverFactory('cbc').available())"
```
Expected output: `cbc True`

This is the same check `README.md` section 6 already documents for the host install — reused here instead of inventing a new one, so a regression in either environment is comparable.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "build: add Dockerfile with CBC solver"
```

---

### Task 2: docker-compose.yml

**Files:**
- Create: `docker-compose.yml`

**Interfaces:**
- Consumes: the `Dockerfile` from Task 1 (via `build: .`).
- Produces: `docker compose config` validates cleanly; `docker compose run cli ...` and `docker compose up` (no profile) only ever touch the `cli` service.

- [ ] **Step 1: Write `docker-compose.yml`**

```yaml
services:
  cli:
    build: .
    volumes:
      - ./data:/app/data
    command: ["run"]

  api:
    build: ./services/api
    profiles: ["future"]

  worker:
    build: ./services/worker
    profiles: ["future"]
```

`api`/`worker` point at `services/api/` and `services/worker/`, which don't exist yet (Fase 3, no code). This is fine: `docker compose config` resolves and prints the file without touching the filesystem paths of profile-gated services that aren't active, and `docker compose up`/`build` without `--profile future` never look at them. Confirmed by direct execution against both this design and an `image:`-based alternative — both validate; `build:` was kept because it's the accurate placeholder for what Fase 3 will actually add.

- [ ] **Step 2: Validate**

Run: `docker compose config`
Expected: exit 0, prints a resolved config for the `cli` service only (services gated behind an inactive profile are dropped from the default `config` output).

Run: `docker compose config --profile future`
Expected: exit 0, now also prints `api` and `worker` with their (still-nonexistent) build contexts — proves the placeholders are syntactically real, not silently broken.

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "build: add docker-compose.yml with cli service and future placeholders"
```

---

### Task 3: Container smoke test against the Fase 2B fixture

This is Fase 2C's actual exit criterion: the exact same fixture and command that already runs clean on the host (Fase 2B, `tests/test_xm_smoke_cli.py`) must run clean **inside** the container.

Design note on where `--out` writes: the container runs as root and `./data` is git-ignored (may not exist on the host at all). Mounting `./data` and pointing `--out` into it would make Docker create a root-owned `data/` directory and root-owned result CSVs in the working tree — an annoying cleanup surface for no benefit. Instead this smoke test writes to `/tmp/smoke` *inside* the container (no bind mount at all) and asserts on stdout plus `test -f` inside the same `docker run`, exactly mirroring what `tests/test_xm_smoke_cli.py` already asserts on the host.

**Files:**
- None created — this task is a verification step run against the image built in Task 1, executed and recorded here as evidence.

**Interfaces:**
- Consumes: `despacho-udea:latest` image from Task 1, `tests/fixtures/xm_smoke/` (already `COPY`'d into the image).

- [ ] **Step 1: Run the smoke test**

```bash
docker run --rm --entrypoint sh despacho-udea:latest -c '
uv run --no-sync python -m app run 2024-04-18 -t preideal --data-dir tests/fixtures/xm_smoke --out /tmp/smoke &&
test -f /tmp/smoke/marginal_price-2024-04-18-preideal.csv &&
test -f /tmp/smoke/dispatch_by_gen-2024-04-18-preideal.csv &&
echo SMOKE_TEST_OK
'
```

Expected: exit 0, stdout contains `Done: 1 ok, 0 failed.` followed by `SMOKE_TEST_OK`. No network access attempted (`... files already downloaded. Skipping download` in the output confirms `ensure_data_for_date` treated the fixture as already-present, per the Fase 2B layout guarantee — if this instead tries to reach XM, that's a fixture/layout regression, not a Docker problem).

- [ ] **Step 2: Run the same check via docker-compose, to prove the compose service wires to the same image**

```bash
docker compose run --rm --entrypoint sh cli -c '
uv run --no-sync python -m app run 2024-04-18 -t preideal --data-dir tests/fixtures/xm_smoke --out /tmp/smoke &&
test -f /tmp/smoke/marginal_price-2024-04-18-preideal.csv &&
test -f /tmp/smoke/dispatch_by_gen-2024-04-18-preideal.csv &&
echo SMOKE_TEST_OK
'
```

Expected: same as Step 1 (`docker compose run` rebuilds/reuses the `cli` service's image and applies the same entrypoint override).

No commit for this task — nothing new lands in git. Record the two command outputs in the PR description as the evidence for the exit criterion.

---

### Task 4: README updates

Three sections describe state that Fase 2C changes: the repo map doesn't list the new root files, section 6 has no Docker install path, and section 11 states "no Dockerfile" as a known gap.

**Files:**
- Modify: `README.md` (section 5 "Mapa del repositorio", section 6 "Instalacion local", section 8 "Como ejecutar", section 11 "Brechas conocidas")

**Interfaces:** None — documentation only.

- [ ] **Step 1: Add the new files to the repo map (section 5)**

In the fenced block under `## 5. Mapa del repositorio`, after the closing lines `data/ ... git-ignored` / `solver/ ... git-ignored` and before the closing ` ``` `, add:

```text
Dockerfile           # imagen runtime (CBC + deps, sin data/ ni notebooks)
docker-compose.yml   # servicio cli + placeholders api/worker (Fase 3)
.dockerignore
```

- [ ] **Step 2: Add a Docker subsection to section 6**

After the existing "Verificacion basica" code block (ends `uv run pytest -q` / ` ``` `) and before the `---` that closes section 6, insert:

`````markdown

### Con Docker (alternativa a instalacion local)

Requiere Docker. No necesita `uv` ni CBC instalados en el host — ambos
viven dentro de la imagen.

```bash
docker build -t despacho-udea .
docker run --rm --entrypoint uv despacho-udea run --no-sync python -c \
  "import pyomo.environ as pyo; print('cbc', pyo.SolverFactory('cbc').available())"
```

Para correr contra datos reales, montar `data/` como volumen (ver
`docker-compose.yml`, servicio `cli`):

```bash
docker compose run --rm cli run 2024-04-18 -t preideal
```
`````

- [ ] **Step 3: Add a Docker example to section 8**

After the line `` `run_dispatch.run_dispatch(...)` se conserva para notebooks y compatibilidad. `` and before the `---` that closes section 8, insert:

````markdown

Mismo comando via Docker (ver seccion 6):

```bash
docker compose run --rm cli run 2024-04-18 -t preideal
```
````

- [ ] **Step 4: Remove the stale gap from section 11**

Delete this line from the bullet list under `## 11. Brechas conocidas`:

```markdown
- No existe todavia Dockerfile ni `docker-compose.yml`.
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document Docker install/run path, update repo map and known gaps"
```

---

## Exit criteria (from the spec, section 2/5)

- [ ] `docker build -t despacho-udea:latest .` succeeds.
- [ ] CBC is available inside the image (Task 1 Step 4).
- [ ] `docker compose config` (and `--profile future`) validate without error.
- [ ] The Fase 2B fixture runs clean inside the container via both plain `docker run` and `docker compose run cli`: exit 0, `Done: 1 ok, 0 failed.`, both result CSVs present (Task 3).
- [ ] `uv run pytest -q` still passes on the host, unchanged by this phase's Docker-only additions.
- [ ] README reflects the new files and the closed gap.

After Task 4, open a PR against `develop` (per repo workflow — no direct commits to `develop`). Roadmap Fase 2 (A + B + C) is then complete; next is Fase 3 (backend API + persistencia), which needs its own brainstorming session — its design isn't closed yet.
