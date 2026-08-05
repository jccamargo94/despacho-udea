# Fase 3: Backend API y persistencia — diseno

Fecha: 2026-08-05
Roadmap: `docs/roadmap-aplicacion-despacho.md`, seccion "Fase 3".

## Contexto

Fase 2 (A + B + C) completa, PR #6 mergeado a `develop`. `docker-compose.yml`
ya trae placeholders vacios `api`/`worker` (`profiles: ["future"]`,
`build: ./services/api` y `./services/worker`) esperando este diseno.

A diferencia de Fase 2, el roadmap no traia decisiones tomadas para Fase 3 —
solo cuatro bullets sin desarrollar (framework, DB, "worker asincrono",
forma de exponer resultados). Este diseno se cerro por brainstorming
interactivo con el usuario; las decisiones de mayor costo de reversion
(framework, motor DB, mecanismo de worker) se verificaron con `/advisor`
antes de comprometerse.

`AGENTS.md` advierte explicitamente que este repo no usa FastAPI/Celery/
Redis/Thori — no como prohibicion permanente, sino como aviso de no copiar
convenciones de otro stack sin verificar que aplican aqui. Este diseno
introduce FastAPI (justificado abajo) pero explicitamente evita Celery/Redis.

## Decisiones tomadas (con el usuario)

- **Contexto de uso**: servidor compartido, pocos usuarios conocidos (no
  local-solo-yo, no publico/multiusuario masivo). Implica: auth real (no
  "sin auth"), pero no rate limiting/aislamiento multi-tenant pesado.
- **Duracion de solve real**: no medible en esta sesion (sandbox sin DNS,
  `fetch` a la API de XM fallo con `NameResolutionError` — ver seccion de
  brechas). Segun el usuario, corridas reales historicas tomaron segundos a
  ~1-2 minutos. Aun con ese rango bajo, se decidio worker asincrono real
  (no bloquear el request HTTP) porque: (a) el roadmap lo pide
  explicitamente como entregable de Fase 3, (b) el placeholder de
  `docker-compose.yml` ya anticipa un servicio `worker` separado, y (c) un
  request sincrono de hasta 2 minutos excede el `proxy_read_timeout` por
  defecto de nginx (60s) si el server queda detras de un proxy reverso.
- **Framework API**: **FastAPI**. Los modelos de dominio ya son pydantic v2
  (`app/schemas/`) — se exponen casi directo como request/response sin capa
  de traduccion.
- **Mecanismo de worker**: **cola en la propia DB, sin broker** (no Celery,
  no Redis). Proceso `worker` separado hace polling sobre la tabla `runs`
  (`status='pending'`), reclama con `FOR UPDATE SKIP LOCKED` (Postgres),
  ejecuta `run_case` sin modificarlo, escribe el resultado de vuelta.
  Reutiliza la DB ya elegida en vez de agregar infra nueva (Redis); permite
  >1 replica de worker sin cambios futuros.
- **DB + Auth**: **Supabase (Postgres + Auth) juntos**, sin usar la API
  auto-generada de Supabase (PostgREST) — FastAPI sigue siendo la unica
  API HTTP del backend. Justificacion del brainstorming: el usuario
  confirmo que el servidor compartido tiene salida a internet normal (la
  alternativa de descartar Supabase por posible egress restringido no
  aplica) y que **si** necesita saber que usuario especifico lanzo cada
  corrida (justifica auth con identidad real en vez de una API key
  compartida). El flujo de login vive en el frontend (Fase 4, fuera de
  alcance) usando `supabase-js` directo contra Supabase; este backend
  **nunca** maneja passwords ni emite tokens, solo verifica el JWT entrante
  (JWKS de Supabase) en cada request y lee `sub` como user id. Riesgo
  conocido y aceptado: el proyecto free-tier de Supabase se pausa tras ~1
  semana de inactividad total; uso academico en rafagas puede toparse con
  esto (el worker empezaria a fallar sus polls silenciosamente hasta que
  alguien reactive el proyecto).
- **Series de tiempo vs DB**: **no duplicar** dispatch/precios/BESS en DB.
  El modelo de datos minimo que el propio roadmap ya sugiere (seccion
  "Modelo de datos minimo") separa `MetricSet` (escalares: MAE, RMSE, bias,
  WAPE, sMAPE — estructurado, consultable) de `ResultArtifact` (series de
  tiempo — un puntero a archivo, no contenido en DB). Esta fase sigue esa
  separacion: DB solo para lo estructurado/consultable, CSV via `Storage`
  (sin tocar) para las series. Evita reescribir `case_builder`/`runner`/
  `results.py`, que ya estan validados end-to-end contra el fixture de
  Fase 2B y son la pieza mas fragil del repo (`.agents/rules/overview.mdc`).
- **DB en tests**: SQLite (rapido, sin infra), Supabase Postgres real en
  dev/prod. Riesgo aceptado explicitamente por el usuario: algo que pasa en
  SQLite podria fallar en Postgres real (tipos `jsonb`, `FOR UPDATE SKIP
  LOCKED` es Postgres-only). El plan de implementacion debe evitar SQL
  especifico de Postgres en el codigo de negocio comun y aislarlo donde sea
  estrictamente necesario (el claim del worker).

## Arquitectura

```
Frontend (Fase 4, fuera de alcance)
  --login/signup--> Supabase Auth
  --JWT (Authorization: Bearer)--> FastAPI (services/api/)
                                       |
                             verifica JWT (JWKS de Supabase)
                                       |
                          INSERT runs(status='pending') y responde run_id
                          (no bloquea esperando el solve)
                                       |
                          Supabase Postgres: cases / scenarios / runs /
                          metric_sets
                                       ^
                                       | polling: SELECT ... WHERE
                                       | status='pending' FOR UPDATE SKIP
                                       | LOCKED; UPDATE status='running'
                          Worker (services/worker/, proceso Python simple)
                                       |
                          llama app.pipeline.runner.run_case sin modificar
                                       |
                          CSVs via app.storage (Storage/LocalStorage, sin
                          cambios) + UPDATE runs/metric_sets con el resultado
```

`app/` (dominio: modelo Pyomo, `case_builder`, `runner`, `results`,
`evaluate`, schemas) permanece intacto. API y worker son capas nuevas y
delgadas alrededor de el, en `services/api/` y `services/worker/`, tal como
anticipa `docker-compose.yml` desde Fase 2C.

## Modelo de datos (Supabase Postgres)

- **`scenarios`**: `id`, `mode`, `penetration_level`, `units` (`jsonb`,
  espejo directo de `BessScenario.units`), `created_at`, `created_by`
  (user id).
- **`cases`**: `id`, `dispatch_date`, `level`, `solver`, `compute_prices`,
  `scenario_id` (FK nullable a `scenarios`).
- **`runs`**: `id`, `case_id` (FK), `user_id` (de `sub` del JWT), `status`
  (`pending`/`running`/`done`/`failed`), `created_at`, `started_at`,
  `finished_at`, `error`, `out_dir`, `dispatch_path`, `price_path`,
  `bess_path`.
- **`metric_sets`**: `run_id` (FK unico), columnas escalares de
  `price_metrics()` (mae, rmse, bias, wape, smape) + columnas de
  `_bess_summary()` cuando aplica (`bess_charge_mwh`, `bess_discharge_mwh`,
  `bess_avg_soc_mwh`, `bess_net_revenue`).

`out_dir` por run usa un path unico por `run_id` (p.ej.
`data/results/{run_id}/`), evitando la colision de nombres que ya existe en
el flujo CLI (`marginal_price-{date}-{level}.csv` no incluye escenario, asi
que dos escenarios BESS distintos para la misma fecha/level se pisan entre
si en el CLI hoy — bug preexistente, fuera de alcance de esta fase, pero el
path por-run del worker evita heredarlo).

## Worker

Proceso Python simple (sin Celery/Redis), loop:

1. `SELECT id FROM runs WHERE status='pending' ORDER BY created_at LIMIT 1
   FOR UPDATE SKIP LOCKED` dentro de una transaccion; si hay fila, `UPDATE
   status='running', started_at=now()`.
2. Reconstruir `DispatchCase`/`BessScenario` desde `cases`/`scenarios`.
3. Llamar `run_case(case, evaluate=True, out=f"data/results/{run_id}")` —
   **sin modificar** `app/pipeline/runner.py`.
4. Con el `RunResult`: si `ok`, `UPDATE runs SET status='done', ...paths`,
   `INSERT metric_sets`. Si no, `UPDATE runs SET status='failed',
   error=...`.
5. Poll cada ~5s si no hay pendientes (intervalo fijo, sin backoff — volumen
   bajo esperado, no vale la pena la complejidad de un backoff adaptativo
   todavia).

`FOR UPDATE SKIP LOCKED` es Postgres-only — aislado en la capa de acceso a
`runs` (un modulo, no esparcido), para no romper los tests con SQLite (que
en ese modulo especifico corren contra Postgres real o se saltan ese caso,
a definir en el plan de implementacion).

## API (FastAPI, `services/api/`)

Todas las rutas requieren `Authorization: Bearer <jwt-supabase>`, verificado
contra el JWKS de Supabase (sin login/signup aqui — eso es responsabilidad
del frontend de Fase 4 hablando directo con Supabase Auth).

- `POST /scenarios` — body `BessScenario` -> inserta, devuelve id.
- `POST /runs` — body: fecha, level, solver, `scenario_id` opcional ->
  inserta `cases`+`runs(status='pending')`, devuelve `run_id` de inmediato.
- `GET /runs` — lista runs del usuario (o todos, a decidir en el plan) con
  status+metricas.
- `GET /runs/{id}` — detalle: status, timestamps, metricas (`metric_sets`),
  error si aplica.
- `GET /runs/{id}/dispatch` | `/prices` | `/bess` — parsea el CSV
  correspondiente on-demand con pandas y devuelve JSON (para graficar en el
  frontend). Sin cache ni persistencia adicional — los archivos son de un
  dia, chicos, el parseo por request es barato a este volumen.
- `GET /runs/{id}/download/{artifact}` — `FileResponse` del CSV crudo.

## Brecha `evaluate`/`compare` (alcance confirmado)

`evaluate_saved_run` (`app/pipeline/evaluate.py`) escribe
`metrics-{date}-{level}.csv` pero nunca toca `metrics-summary.csv` (que solo
`run_many` genera, una vez, al final de un batch). Efecto: un flujo CLI
`run --no-eval` -> `evaluate` -> `compare` nunca ve las metricas del
`evaluate` posterior.

Fix acotado: `evaluate_saved_run` hace upsert de su fila en
`metrics-summary.csv` (mismas claves date/type/scenario que usa
`run_many` — `scenario="baseline"`, ya que `evaluate_saved_run` no recibe
escenario hoy) en vez de solo escribir el CSV individual. Cambio contenido
en `evaluate.py`, no toca `case_builder`/`runner`.

Este fix es independiente del path API/DB: el worker escribe metricas
directo en `metric_sets` via `run_case(..., evaluate=True)`, asi que nunca
hereda este bug — el fix de `metrics-summary.csv` es exclusivamente para
usuarios de CLI que no pasan por la API.

## Nuevas dependencias

`fastapi`, `uvicorn`, `sqlalchemy`, `psycopg[binary]`, `alembic`
(migraciones de schema), `pyjwt` (verificacion de JWT contra JWKS de
Supabase). Nada de Celery, Redis, ni ORM mas alla de SQLAlchemy para 4
tablas.

## Testing

- `tests/test_api_*.py`: `TestClient` de FastAPI contra SQLite, cubre
  logica de endpoints (auth, validacion, serializacion).
- `tests/test_worker_*.py`: ejercita el loop claim/ejecutar/actualizar
  contra el fixture `xm_smoke` (Fase 2B) de punta a punta, en la misma
  linea que `tests/test_xm_smoke_run.py` ya valida `run_case` directo.
- El claim con `FOR UPDATE SKIP LOCKED` (Postgres-only) se aisla en su
  propio modulo; su test especifico corre contra Postgres real o se marca
  para saltarse bajo SQLite — a definir el mecanismo exacto en el plan de
  implementacion.

## Brechas conocidas / riesgos que quedan para el plan de implementacion

- No se pudo medir un solve real con datos historicos completos (sandbox de
  esta sesion no tiene DNS/salida a internet; `fetch 2024-04-18` fallo con
  `NameResolutionError` contra la API de XM). El rango "segundos a ~1-2 min"
  es la mejor estimacion del usuario, no un dato medido. Si un solve real
  resulta mucho mas lento, el intervalo de poll del worker (~5s) sigue
  siendo razonable, pero vale remedir apenas haya acceso a datos reales.
- Pausa de Supabase free-tier tras inactividad prolongada — aceptado como
  riesgo conocido, no mitigado en este diseno (ver seccion de decisiones).
- Colision de nombres de archivo CSV por escenario en el flujo CLI (sin
  `scenario` en el nombre de archivo) — preexistente, no introducido ni
  resuelto por esta fase; el worker lo evita usando paths por-`run_id`.
- Mezcla de SQLite (tests) y Postgres (dev/prod) — riesgo aceptado
  explicitamente por el usuario; el plan de implementacion debe minimizar
  SQL especifico de Postgres fuera del modulo de claim del worker.
