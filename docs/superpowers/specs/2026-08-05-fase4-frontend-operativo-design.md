# Fase 4: Frontend operativo — diseno

Fecha: 2026-08-05
Roadmap: `docs/roadmap-aplicacion-despacho.md`, seccion "Fase 4".

## Contexto

Fase 3 completa, PR #7 mergeado a `develop` (merge commit `c85ee0f8`).
Backend FastAPI (`services/api/`) + Supabase Postgres/Auth (`app/db/`) +
worker por polling (`services/worker/`) ya funcionando, 129/129 tests.

El roadmap no trae decisiones tomadas para Fase 4 — solo cinco bullets sin
desarrollar (pantalla de ejecuciones, configurador BESS, comparador,
visualizacion de despacho, explorador de artefactos/logs). Este diseno se
cerro por brainstorming interactivo; las decisiones de mayor costo de
reversion (stack de frontend, monorepo vs repo separado, hosting) se
verificaron con `/advisor` antes de comprometerse.

Un primer borrador de este diseno asumia que Fase 4 era "frontend puro mas
dos fixes chicos de API". Una auditoria campo-por-campo del backend real
contra las cinco pantallas del roadmap (`services/api/main.py`,
`app/db/models.py`, `app/db/queries.py`, `services/worker/main.py`)
encontro que el gap es mas grande: faltan campos en las respuestas
existentes, falta un endpoint completo (`GET /scenarios`), y no existe
ninguna captura de logs. Ver seccion "Backend: brechas encontradas".

## Decisiones tomadas (con el usuario)

- **Stack frontend**: **Next.js (App Router)**, no una SPA con Vite.
  Mismo React, pero con rutas por archivo y la opcion de SSR/API routes
  propias sin reescribir nada si la app crece — el usuario explicito que
  quiere margen para escalar aunque hoy no lo necesite.
- **UI**: TypeScript + Tailwind + shadcn/ui (componentes copiados al repo,
  no dependencia opaca; sus charts usan Recharts por debajo, que tambien
  se usa para series de tiempo/comparativas).
- **Gestor de paquetes**: `pnpm`.
- **Estructura de repo**: `frontend/` en la raiz del repo, monorepo simple
  junto a `app/`/`services/`. **Sin Turborepo**: Turborepo coordina
  build/cache entre *multiples* paquetes JS/TS; hoy solo habria un paquete
  (`frontend/`), y Python ya tiene su propio toolchain (`uv`) que
  Turborepo no orquesta. Se agrega el dia que haya 2+ paquetes JS/TS que
  compartan codigo y valga la pena cachear/orquestar tareas entre ellos.
- **Data fetching**: TanStack Query. Necesario para polling del status de
  un run mientras corre (`refetchInterval` que se apaga en status
  terminal), cache/invalidacion al crear runs/escenarios.
- **Auth**: `supabase-js` **cliente-only** — sesion en `localStorage`
  (default de la libreria), proteccion de rutas via un `AuthProvider` +
  guard por componente que redirige a `/login` si no hay sesion. Sin
  `@supabase/ssr` ni `middleware.ts` de Next.js: toda la app de esta fase
  es client-side hacia una API externa (FastAPI), no hay server components
  tocando datos de Supabase directamente, asi que el patron SSR con
  cookies no aporta nada hoy.
- **Testing**: Vitest + React Testing Library para componentes/hooks. Sin
  Playwright/e2e en esta fase — YAGNI hasta que haya superficie suficiente
  para justificarlo.
- **Hosting**: Vercel (plan Hobby/free), justificado por: repo publico, uso
  academico/no-comercial (encaja con la letra chica del free tier),
  Next.js es de Vercel (cero config), preview deployments por PR. **Fuera
  de alcance de la implementacion de Fase 4** — el criterio de "hecho" de
  esta fase es `next dev` funcionando contra la API corriendo local
  (docker-compose), con `NEXT_PUBLIC_API_BASE_URL` configurable por env.
  El deploy real a Vercel (y la implicacion de que el backend necesita un
  origen HTTPS publico para que el navegador no lo bloquee como mixed
  content) queda para una fase de despliegue posterior.
- **Alcance de "logs"**: el roadmap pide explorador de logs, y hoy no
  existe ninguna captura — el worker imprime a stdout y `Run` solo guarda
  `error` (string). El usuario confirmo agregar captura y persistencia
  real (no solo mostrar `error`) — ver diseno del worker abajo.
- **Division en sub-fases**: igual que Fase 2 (A/B/C), un spec para toda
  Fase 4, un plan por sub-fase, cada una en su propia rama -> PR contra
  `develop`.

## Backend: brechas encontradas (auditoria campo-por-campo)

Para cada pantalla del roadmap, el endpoint/campo que consume, verificado
contra `services/api/main.py` + `app/db/models.py` + `app/db/queries.py`:

| Pantalla | Endpoint necesario | Estado hoy |
|---|---|---|
| Ejecuciones | `GET /runs`, `GET /runs/{id}` | `_run_summary()` (`main.py:50-58`) solo devuelve `run_id/status/created_at/started_at/finished_at/error` — **falta `dispatch_date`, `level`, `scenario_id`**, que viven en `Case`, no en `Run`. Sin esto la tabla de ejecuciones es IDs opacos. |
| Configurador BESS | `POST /scenarios` (existe), `GET /scenarios` | **No existe `GET /scenarios`** — se puede crear un escenario pero nunca listarlo/releerlo. `POST /runs` toma `scenario_id`, asi que el formulario de crear run necesita un selector que tampoco existe hoy. |
| Comparador precios/metricas | `GET /runs/{id}` | Devuelve `rmse/mae/bias/wape/smape/r2` pero **omite los 4 campos BESS ya persistidos** en `MetricSet` (`bess_charge_mwh`, `bess_discharge_mwh`, `bess_avg_soc_mwh`, `bess_net_revenue` — confirmados en `app/db/models.py:73-76`). |
| Despacho por recurso/tecnologia | `GET /runs/{id}/dispatch` | Ya existe, sin cambios. Nota de render: el CSV es series por **hora-del-dia** (periodo), no timestamps — el eje del grafico es indice de periodo, no fecha/hora real. |
| Artefactos | `GET /runs/{id}/download/{artifact}` | Ya existe, sin cambios. |
| Logs | *(no existe)* | El worker imprime a stdout/stderr (`print()` en `app/pipeline/runner.py`, `case_builder.py`); nada se captura ni persiste. `Run.error` solo cubre el caso de fallo, no un log completo. |
| — (transversal) | CORS | **No hay `CORSMiddleware`** en `services/api/main.py`. Un frontend en un origen distinto (`localhost:3000` en dev, Vercel en prod) sera bloqueado por el navegador sin esto. |

Los 5 cambios de backend arriba se agrupan en **fase4a** (ver abajo), en vez
de repartirlos por sub-fase, para que `services/api/`, `services/worker/`,
`app/db/models.py` y las migraciones se toquen una sola vez y 4b/4c queden
frontend puro.

## Sub-fases

### fase4a — Backend prerequisites + setup + pantalla de ejecuciones

**Backend** (`services/api/`, `services/worker/`, `app/db/`, migracion
Alembic nueva):

1. **Primera tarea, bloqueante para todo lo demas**: verificar el algoritmo
   JWT real de un proyecto Supabase (`curl $SUPABASE_JWKS_URL`, decodificar
   el header de un `access_token` real). `services/api/auth.py:23` hoy
   fija `algorithms=["RS256"]` sin haber verificado contra un proyecto
   real — si el proyecto usa otro algoritmo, todo login del frontend falla
   en seco. Ajustar el codigo a lo que el JWKS realmente advierte.
2. `CORSMiddleware` con origen(es) permitido(s) via env var
   `FRONTEND_ORIGIN` (soporta lista separada por comas para incluir
   `http://localhost:3000` en dev + el dominio de Vercel en prod). Sin
   `allow_credentials=True` (auth es un header `Authorization`, no una
   cookie).
3. `_run_summary()` incluye `dispatch_date`, `level`, `scenario_id` — leidos
   de `Case` via `queries.get_case(session, run.case_id)` (sin JOIN
   optimizado; N+1 aceptable al volumen actual, no vale la complejidad de
   eager-loading todavia).
4. `GET /scenarios` — lista escenarios (todos, no solo los del usuario;
   `Scenario.created_by` no se usa hoy como filtro de ownership,
   consistente con que un escenario es reutilizable entre usuarios).
5. `GET /runs/{id}` agrega los 4 campos `bess_*` al dict de `metrics`.
6. Captura y persistencia de logs:
   - Columna nueva `Run.log_path: str | None` + migracion Alembic.
   - `services/worker/main.py`: envolver la llamada a `run_case(...)` en
     `process_once` con `contextlib.redirect_stdout`/`redirect_stderr`
     hacia un buffer combinado; al terminar (ok o failed), escribir el
     buffer via la abstraccion `Storage` existente a
     `{out_dir}/run.log` y guardar la ruta en `run.log_path` (mismo patron
     que `dispatch_path`/`price_path`/`bess_path`).
   - Endpoint nuevo `GET /runs/{id}/log` — texto plano (no encaja en el
     patron CSV-a-JSON de `_ARTIFACT_PATHS`), 404 si `log_path` es `None`
     o el archivo no existe (mismo criterio que `_artifact_path`).

**Frontend** (`frontend/`):

- Scaffold Next.js + TypeScript + Tailwind + shadcn/ui, `pnpm`.
- Cliente API tipado: wrapper `fetch` que agrega `Authorization: Bearer
  <token>` (leido de la sesion de Supabase) y castea las respuestas a
  tipos TS espejo de los schemas pydantic relevantes (`Run`, `Scenario`,
  `MetricSet`, etc).
- Supabase Auth: paginas `/login` y `/signup` (email+password via
  `supabase-js`), `AuthProvider` (contexto con sesion/usuario/loading),
  guard que redirige a `/login` si no hay sesion.
- Layout/navegacion base (header, links a las secciones de 4b/4c aunque
  esas paginas aun no existan — placeholders esta bien).
- Pantalla de ejecuciones: tabla de `GET /runs` (fecha, level, escenario,
  status, timestamps), boton para crear run (`POST /runs`, formulario
  fecha+level+solver+escenario opcional), detalle de un run con polling
  via TanStack Query que **se detiene en status terminal**
  (`refetchInterval: (query) => isTerminal(query.state.data?.status) ?
  false : 3000`), muestra `error` si `status === "failed"`.
- Timestamps: convertir siempre a hora de Colombia (`America/Bogota`) al
  renderizar — es la razon por la que Fase 3 agrego `DateTime(timezone=True)`
  a las columnas (`created_at`/`started_at`/`finished_at`); fijar esta
  convencion una sola vez (un helper de formato) en vez de que 4a y 4c la
  reinventen cada uno a su manera.

### fase4b — Configurador de escenario BESS + Comparador (frontend puro)

Decisiones cerradas 2026-08-06 (sin brainstorming interactivo — usuario no
disponible, decisiones tomadas via `/advisor` con Opus y verificadas contra
el codigo real, mismo criterio que el resto de este spec). Sin cambios de
backend — todo lo que necesita ya quedo cerrado en fase4a.

**Configurador de escenario BESS** (`/scenarios`, ruta nueva bajo `(app)`):
- Lista de escenarios existentes via `GET /scenarios` (fase4a) + formulario
  para crear uno nuevo contra `POST /scenarios`.
- Formulario: `mode` (selector), `penetration_level` (texto), `units:
  BessUnit[]` como lista dinamica de filas (agregar/quitar), cada fila con
  `name/mwh_nom/hours_to_deplete/initial_soc/min_soc/max_soc/efficiency` +
  `charge_bid`/`discharge_bid` opcionales.
- **`mode` solo ofrece `arbitrage` y `grid_asset` — `generator` se omite
  del selector.** Verificado contra el codigo real
  (`app/model/model.py:310`): `BessMode.generator` sigue sin formulacion
  Pyomo, `raise NotImplementedError("BESS mode 'generator' has no Pyomo
  formulation yet")` antes de tocar el solver. Ofrecerlo en el formulario
  produciria una corrida que garantizadamente falla — no vale la pena
  construir UI para un modo que el backend no soporta. Si `generator` se
  implementa en el backend en el futuro, se agrega al selector entonces.
- Validacion: **el servidor es la unica autoridad** (`BessScenario`'s
  `field_validator` en `app/schemas/bess.py` ya exige `charge_bid` en modo
  `arbitrage` y `discharge_bid` en `arbitrage`/`generator`, devuelve 422 si
  falta). El frontend NO reimplementa esa logica de validacion — solo
  refleja del lado cliente cual de los dos campos de bid mostrar/ocultar
  segun el `mode` seleccionado (UX, no validacion), y muestra el mensaje
  de error del 422 si el servidor rechaza el submit.
- El selector de escenario en el formulario de crear run (`/runs`, fase4a)
  se actualiza para usar el mismo tipo `Scenario`/`BessUnit` ya definido
  aqui (ver "Carry-overs" abajo).

**Comparador de precios y metricas** (`/compare`, ruta nueva bajo `(app)`,
separada de `/runs` — no se modifica `RunsTable`):
- Selector multi-run: dropdown/checklist de corridas con `status ===
  "done"` (obtenidas de `GET /runs`, ya cacheada por TanStack Query bajo
  la key `["runs"]`).
- **`GET /runs` (`_run_summary`, `services/api/main.py`) no incluye
  metricas** — solo `GET /runs/{id}` (`get_run_detail`) las tiene. El
  comparador debe pedir el detalle de cada corrida seleccionada
  individualmente (`useQueries` de TanStack Query sobre los ids
  seleccionados), no asumir que las metricas ya estan disponibles desde
  la lista.
- **`status === "done"` no implica que existan metricas.** Confirmado
  contra `tests/test_worker_main.py`: una corrida exitosa contra el
  fixture `xm_smoke` (sin datos reales XM para esa fecha) termina en
  `status: "done"` con `get_metric_set(...) is None` — `evaluate` se
  salta si no hay actuals que comparar. El comparador debe manejar
  `metrics: null` por corrida explicitamente (mostrar "sin metricas" en
  esa columna), no asumir que toda corrida `done` tiene numeros.
- **Presentacion: tabla, no un solo grafico de barras.** Los 10 campos de
  metricas tienen escalas y unidades incompatibles entre si (`rmse/mae/bias`
  en COP/MWh, `wape/smape` en porcentaje, `r2` en 0-1, `bess_charge_mwh`/
  `bess_discharge_mwh`/`bess_avg_soc_mwh` en MWh, `bess_net_revenue` en
  COP) — un grafico de barras agrupado mezclando las diez seria ilegible.
  La tabla (metricas como filas, corridas seleccionadas como columnas) es
  el comparador primario. Si se agregan graficos, son "small multiples"
  (un grafico chico por metrica, no uno solo con las diez) — usar la skill
  `dataviz` al escribir ese codigo si se decide incluirlos en el plan.

**Carry-overs de fase4a a resolver como parte de esta fase** (deuda tecnica
que el configurador BESS es el primer consumidor real de estos tipos, asi
que se cierra aqui en vez de seguir difiriendola):
- `Scenario.units` en `frontend/lib/types.ts` esta tipado `unknown[]`
  (placeholder de fase4a) — tipar correctamente como `BessUnit[]` con un
  tipo `BessUnit` espejo del backend (`app/schemas/bess.py`).
- Los formularios nuevos (configurador BESS, comparador) usan los
  componentes shadcn/ui ya instalados en fase4a
  (`frontend/components/ui/*`: button, card, table, input, label, select,
  badge) en vez de `<input>`/`<select>` crudos — fase4a los instalo pero
  nunca los uso (Task 9 los trajo, Tasks 13/16 usaron HTML plano). No se
  retrofitea fase4a, pero fase4b arranca usando los componentes reales.
- `shadcn` (el CLI) esta en `dependencies` de `frontend/package.json` en
  vez de `devDependencies` — moverlo como parte del primer commit de esta
  fase (cambio de una linea).

### fase4c — Visualizacion de despacho + Explorador de artefactos/logs (frontend puro)

- Visualizacion de despacho por recurso/tecnologia: grafico Recharts de
  series por periodo horario (eje = indice de hora del dia, no
  timestamp), leido de `GET /runs/{id}/dispatch`.
- Explorador de artefactos: lista de artefactos disponibles por run
  (dispatch/prices/bess), boton de descarga via
  `GET /runs/{id}/download/{artifact}`.
- Explorador de logs: texto de `GET /runs/{id}/log` (fase4a) en un visor
  simple (`<pre>` con scroll), mas `run.error` cuando `status ===
  "failed"`.
- Sin cambios de backend — todo lo que necesita ya quedo cerrado en
  fase4a.

## Testing

- Backend (extension de la suite existente): `tests/test_api_*.py` cubre
  los campos nuevos en `_run_summary`/`GET /runs/{id}`, el endpoint
  `GET /scenarios`, el endpoint `GET /runs/{id}/log`, y el header CORS en
  la respuesta. `tests/test_worker_*.py` cubre que `process_once` escribe
  `log_path` y que el archivo contiene la salida esperada.
- Frontend: Vitest + React Testing Library para componentes (formularios,
  tabla de ejecuciones, guard de auth) y hooks de TanStack Query (con el
  cliente API mockeado). Sin e2e en esta fase.

## Brechas conocidas / riesgos que quedan para el plan de implementacion

- El deploy real a Vercel requiere que el backend tenga un origen HTTPS
  publico (mixed content bloquea `http://` desde una pagina `https://`) —
  fuera de alcance de Fase 4, a resolver en una fase de despliegue
  posterior.
- `GET /scenarios` sin filtro de ownership (lista todos los escenarios de
  todos los usuarios) — aceptado como consistente con que un escenario es
  reutilizable; revisar si en algun momento se necesita scoping por
  usuario o equipo.
- La captura de logs redirige stdout/stderr del proceso del worker durante
  el solve; no captura logging estructurado si en el futuro `runner.py`
  migra de `print()` a `logging` con handlers propios (habria que
  redirigir tambien el logger raiz en ese caso).
- El plan `fase4a` debe decidir el mecanismo exacto de captura de log (una
  sola cadena de texto en memoria via `io.StringIO` es suficiente al
  volumen actual — un solve de segundos a ~1-2 minutos — no hace falta
  streaming a archivo en tiempo real).
