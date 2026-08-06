# Fase 5: Diseno visual del frontend — diseno

Fecha: 2026-08-06
Roadmap: `docs/roadmap-aplicacion-despacho.md`.
Precedente directo: `docs/superpowers/specs/2026-08-05-fase4-frontend-operativo-design.md`
(Fase 4a/4b/4c completas — frontend funcional, sin estilos: shadcn "base-nova"
por defecto, `(app)/layout.tsx` sin clases en el `<nav>`).

## Contexto

El usuario genero mockups visuales con Google AI Studio (repo publico separado
`jccamargo94/colombian-dispatch`, copia local en `mockups/energy_dispatch/`:
7 `code.html` + `screen.png` por pantalla, mas `grid_performance_logic/DESIGN.md`
con el sistema de diseno). Se verificaron ambas fuentes antes de escribir este
documento, no solo las capturas:

- El repo clonado es una SPA Vite + React 19 con `mockData.ts` hardcodeado,
  sin auth, sin cliente API, con nombres de color estilo Material Design 3
  (`surface-container`, `on-surface`, etc.) en vez de las variables shadcn ya
  usadas en `frontend/app/globals.css`. Sirve como referencia de lenguaje
  visual, no como codigo portable — stack distinto (Vite vs Next.js App
  Router, sin TanStack Query, sin supabase-js).
- El "modo oscuro" del mockup no es un tema real: `App.tsx:110/118` fuerza
  dark solo cuando `currentView === 'comparar'`. Una sola pantalla, no un
  toggle. Se descarta como fuente para dark mode; el asistente deriva la
  paleta oscura desde los tokens claros de `DESIGN.md` con el mismo metodo
  tonal, sin una segunda vuelta a la herramienta de diseno.
- Verificacion campo-por-campo del mockup contra el contrato real (no desde
  la memoria/prosa del mockup) encontro discrepancias de unidades y campos
  inventados — tabla completa abajo. Se corrigen en este diseno.
- Se encontro y corrigio, fuera del alcance de esta fase, una corrupcion de
  working tree no commiteada: `services/` (API+worker) habia sido movido a
  `notebooks/services/` sin `git mv`, rompiendo `docker/Dockerfile.api`.
  Restaurado antes de este diseno (`git mv notebooks/services services`,
  el arbol quedo identico a `HEAD`, no hizo falta commit).
- Se verifico en vivo (no se asumio) si HiGHS es usable hoy: `pyomo` 6.7.3
  (pin actual) solo expone `appsi_highs` via `SolverFactory`. Con
  `pyomo>=6.9` aparece un nombre `highs` directo respaldado por `highspy`
  (ya es dependencia de `pyproject.toml`). Probado contra
  `tests/fixtures/xm_smoke`: el solve MILP inicial falla con `NoDualsError`
  porque la interfaz nueva de Pyomo intenta cargar el `Suffix` `dual`
  (declarado una vez, `app/model/model.py:138`) despues de cada solve,
  incluso cuando el problema es un MIP sin duales validos — CBC tolera esto
  en silencio, la interfaz nueva no. Diferir la declaracion del `Suffix`
  hasta el resolve de pricing arregla el paso MILP, pero el resolve LP de
  pricing (`_solve_pricing_lp`) **sigue fallando** aun con todas las
  variables enteras fijadas — la interfaz nueva de HiGHS parece exigir
  dominio continuo, no solo valor fijo. Esto es una tarea de backend real
  (relajar dominio antes del resolve de pricing, o extraer duales por otra
  via), no un cambio de una linea. **HiGHS queda fuera de esta fase**: la UI
  lo muestra deshabilitado, la implementacion queda como tarea de backend
  separada.

## Decisiones tomadas (con el usuario)

- **Esencia/vibe**: Modern SaaS (registro Linear/Vercel-dashboard). Audiencia
  primaria: portafolio (se muestra a reclutadores/empleadores como trabajo de
  producto pulido, no solo herramienta academica).
- **Tema**: light + dark, ambos de primera clase. Dark derivado por el
  asistente desde `DESIGN.md`, no copiado del mockup (ver Contexto).
- **IA/navegacion**: reskin + nav polish. Sin rutas nuevas, con una excepcion
  justificada (`/reset-password`, ver abajo). Se descartan del sidebar las
  secciones globales "Logs" y "Soporte" que trae el mockup — Logs sigue
  siendo parte del detalle de una corrida (fase4c), Soporte no es una
  feature real de este proyecto.
- **Fuente de tokens**: `mockups/energy_dispatch/grid_performance_logic/DESIGN.md`.
  Primario Electric Indigo `#3525cd` (`primary-container` `#4f46e5` para
  estados hover/secundarios). Tipografia: Inter (UI, reemplaza Geist) +
  JetBrains Mono (datos numericos/logs, reemplaza Geist Mono). Radio base
  12px en contenedores/cards, 6-8px en botones/inputs. Paleta categorica de
  charts: indigo/cyan/violet/slate.
- **Iconos**: `lucide-react` — ya es dependencia (`frontend/package.json`,
  declarado en `components.json`) pero sin uso real todavia. Reemplaza los
  Material Symbols del mockup, no se agrega una libreria nueva.
- **Selector de solver** (`Nueva Ejecucion`): **CBC habilitado** (unico
  verificado end-to-end). **HiGHS visible pero deshabilitado**
  ("proximamente") — visible porque hay intencion real y trabajo de backend
  en curso, deshabilitado porque no funciona todavia (ver Contexto). Gurobi,
  CPLEX y SCIP **no se muestran** — no instalados, sin intencion de
  agregarlos en el futuro cercano (a diferencia de HiGHS).
- **Filtrar / paginacion en Ejecuciones**: oculto por completo. `GET /runs`
  no soporta query params hoy; no se agrega en esta fase (fuera de alcance,
  puramente visual).
- **"Solicitar cuenta" en login**: enlaza al `/signup` existente (self-serve
  abierto desde fase4a, sin aprobacion). No se construye un flujo de
  aprobacion nuevo.
- **"Olvidaste tu contraseña"**: real, no decorativo. `supabase-js` ya trae
  `resetPasswordForEmail()` (Supabase gestiona el correo, sin backend
  propio nuevo) + una pagina nueva `/reset-password` para fijar la
  contrasena tras volver del link del correo. **Unica ruta nueva de esta
  fase** — se justifica como requisito de autenticacion, no como seccion de
  producto nueva.
- **Testing**: fase4a uso HTML plano a proposito en varios componentes
  (evitar los tests fragiles del `Select` de shadcn sobre `@base-ui/react` —
  decision ya tomada en fase4b, se mantiene: los `<select>` siguen siendo
  nativos). El reskin cambia el output renderizado de esos componentes; el
  `.test.tsx` de cada uno se actualiza en la misma tarea que su reskin, no
  como limpieza separada al final.

## Discrepancias mockup -> contrato real (por que no se copia literal)

| Pantalla | Mockup muestra | Real (verificado contra codigo) |
|---|---|---|
| Escenarios (unidad BESS) | `capacityMW` ("Cap. MW"), un solo precio de oferta, SOC en `%` | `mwh_nom` (energia, no potencia) — `app/schemas/bess.py`; `charge_bid`/`discharge_bid` separados; `min_soc`/`max_soc`; falta `initial_soc` en el mockup |
| Escenarios (modo) | badge `Mixto` | solo `arbitrage` / `grid_asset` existen (`generator` deliberadamente sin formular, `app/model/model.py:310`) |
| Escenarios (penetracion) | slider 0-100% | `penetration_level` es texto libre (`BessScenario.penetration_level: str`) |
| Comparar | omite `bias`/`smape`; `Avg SoC (%)`; `$` revenue | `MetricSet` (`app/db/models.py:63-77`) tiene los 10 campos incl. `bias`/`smape`; `bess_avg_soc_mwh` es MWh; revenue es COP |
| Detalle de ejecucion (chart) | "Generacion **vs Demanda**" | `GET /runs/{id}/dispatch` solo trae `generador, datetime, dispatch` — no hay serie de demanda |
| Detalle de ejecucion | "Alerta de Restriccion / nodo Norte / slack activada", `Precios_Nodales.csv` | inventado — solo existe `Run.error` (string), no hay datos nodales |
| Nueva Ejecucion (solver) | Gurobi / CPLEX / CBC | solo CBC verificado; HiGHS parcial (ver Contexto); Gurobi/CPLEX no instalados |
| Login | "Olvidaste tu contrasena" / "Solicitar cuenta" | ninguno existia antes de este diseno — ver decisiones arriba |
| Sidebar | secciones globales "Logs", "Soporte"; "Filtrar" y paginacion en Ejecuciones; buscador en Escenarios | fuera de alcance (IA=reskin, sin rutas/features nuevas salvo reset-password) |

## Pantallas (7 + 1 modal), plan de reskin

1. **Login** (`app/login/page.tsx`) — restyle con tokens nuevos. Link real
   "Olvidaste tu contrasena" -> dispara `resetPasswordForEmail`. "Solicitar
   cuenta" -> link a `/signup`.
2. **Reset password** (`app/reset-password/page.tsx`, nueva) — formulario de
   nueva contrasena, consume el token de la URL que Supabase agrega al
   volver del link del correo.
3. **Signup** (`app/signup/page.tsx`) — restyle, mantiene "Confirm Password"
   (chequeo solo de cliente, el backend no lo pide).
4. **Nav shell** (`app/(app)/layout.tsx`) — sidebar fijo: Ejecuciones /
   Escenarios / Comparar + boton "Nueva Ejecucion" + salir. Reemplaza el
   `<nav>` sin clases actual.
5. **Runs list** (`app/(app)/runs/page.tsx` + `components/runs-table.tsx`) —
   tabla con badges de estado, tipografia mono para IDs/fechas/duraciones.
   Sin "Filtrar" ni paginacion (oculto).
6. **Nueva Ejecucion** (`components/create-run-form.tsx`, ya existe como
   formulario en pagina — evaluar si pasa a modal siguiendo el patron del
   mockup) — selector de solver con CBC habilitado / HiGHS deshabilitado.
7. **Run detail** (`app/(app)/runs/[id]/page.tsx`) — chart de despacho
   top-N generadores + bucket "Otros" (ya definido en fase4c, invocar skill
   `dataviz` al tocar ese componente), sin serie de demanda ni alertas
   inventadas, `Run.error` real, visor de logs estilo terminal (mono font),
   descargas de artefactos via fetch+blob (ya eran auth-required desde
   fase4c, sin cambio de mecanismo, solo de estilo).
8. **Escenarios** (`app/(app)/scenarios/page.tsx`) — formulario BESS con
   campos reales (`mwh_nom`, `hours_to_deplete`, `initial_soc`, `min_soc`,
   `max_soc`, `efficiency`, `charge_bid`/`discharge_bid` condicionales por
   `mode`), `mode` = `arbitrage`/`grid_asset` (select nativo, sin `Mixto`),
   `penetration_level` como input de texto.
9. **Comparar** (`app/(app)/compare/page.tsx`) — tabla con los 10 campos
   reales de `MetricSet`, "sin metricas" explicito por corrida sin datos.

## Sub-fases

Mismo patron que Fase 4 (A/B/C): un spec para toda la Fase 5, un plan por
sub-fase, cada una en su propia rama -> PR contra `develop`.

### fase5a — Tokens + nav shell + auth + Ejecuciones

- Tokens de diseno (`app/globals.css`, fuentes en `app/layout.tsx`, paleta
  de charts `--chart-1..5`).
- Nav shell (`(app)/layout.tsx`).
- Login + Signup + Reset password (ruta nueva).
- Runs list + Nueva Ejecucion (con selector de solver CBC/HiGHS-deshabilitado).

### fase5b — Detalle de ejecucion + Escenarios + Comparar

- Run detail (chart, logs, descargas).
- Escenarios (formulario BESS con campos reales).
- Comparar (tabla de 10 metricas).

## Testing

- Cada componente reestilizado actualiza su `.test.tsx` en la misma tarea
  (fase4a dejo varios en HTML plano a proposito; el reskin cambia el DOM
  que esos tests consultan).
- Sin Playwright/e2e en esta fase — mismo criterio YAGNI que Fase 4.
- `resetPasswordForEmail` y la pagina `/reset-password` llevan su propio
  test de componente (formulario + manejo de error), mockeando `supabase-js`
  igual que `login`/`signup` ya lo hacen.

## Brechas conocidas / riesgos que quedan para el plan de implementacion

- HiGHS quedara con UI deshabilitada hasta que una fase de backend separada
  resuelva la extraccion de duales en el resolve de pricing (ver Contexto).
  No bloquea esta fase.
- `GET /runs` sin filtros/paginacion sigue sin existir — si en el futuro se
  agrega, la UI de "Filtrar" que hoy se oculta puede reactivarse en vez de
  reconstruirse desde cero (mantener el componente, no borrarlo, solo no
  montarlo).
- El componente `Nueva Ejecucion` hoy es una pagina/formulario, no un modal;
  el plan de implementacion de fase5a decide si migrar a modal (patron del
  mockup) o mantener pagina — no es una decision de diseno visual, es de
  interaccion, y no cambia ningun contrato de datos.
