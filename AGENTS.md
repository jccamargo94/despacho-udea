# AGENTS.md — Agent entrypoint

Indice corto. Contexto detallado y siempre-activo vive en **`.agents/rules/`**
(formato `.mdc`, agnostico de agente/IDE). `CLAUDE.md` es un symlink a este
archivo.

## Donde buscar

| Necesito | Ubicacion |
|---|---|
| Contexto siempre-activo (dominio, layout, toolchain, workflow, gotchas verificados) | [`.agents/rules/overview.mdc`](.agents/rules/overview.mdc) |
| Convenciones Python reales de este repo (pydantic v2, Storage, Pyomo/solver, testing) | [`.agents/rules/python-patterns.mdc`](.agents/rules/python-patterns.mdc) |
| Mantenimiento de documentación y GitHub Pages | [`.agents/rules/documentation.mdc`](.agents/rules/documentation.mdc) y [`.agents/rules/github-pages.mdc`](.agents/rules/github-pages.mdc) |
| Vision, mapa del repo, instalacion, datos requeridos, brechas conocidas | [`README.md`](README.md) (secciones 1-14) |
| Hacia donde va el proyecto, fases | [`docs/roadmap-aplicacion-despacho.md`](docs/roadmap-aplicacion-despacho.md) |
| Documentación pública para GitHub Pages | [docs/index.html](docs/index.html) y [docs/formulacion-matematica.html](docs/formulacion-matematica.html) |
| Diseno + plan de una fase/feature especifica ya implementada | `docs/superpowers/specs/` + `docs/superpowers/plans/` (un par por fase) |

## Reglas basicas (siempre aplican)

- **No commitear directo a `develop`.** Cada fase/feature no trivial: rama
  `fase<N><letra>-<nombre>` (p.ej. `fase2b-fixture`) -> tests verdes -> PR
  contra `develop`. Ver `.agents/rules/overview.mdc` para el flujo completo
  (superpowers: spec -> plan -> subagent-driven-development).
- **`data/` es git-ignored.** No asuma que existe ni que tiene datos reales.
  No afirme que el modelo "funciona con datos reales" solo porque los tests
  con fixture sintetico (`tests/fixtures/xm_smoke/`) pasan.
- **Verifique convenciones de unidades/escala contra el codigo real** antes
  de escribirlas en un spec o plan — no desde memoria/dominio general. Ya
  hubo un bug real (revenue BESS inflado 1000x, Fase 1) por asumir una
  convencion sin verificar.
- **No limpie/refactorice `case_builder.py` sin una prueba dorada** (golden
  fixture) que capture el comportamiento actual primero.
- **NUNCA** agregue una linea de co-autoria con ningun nombre de modelo/IA
  en un mensaje de commit, titulo de PR, o cuerpo de PR (p.ej.
  `Co-Authored-By: ...`, `🤖 Generated with ...`). Esto aplica a **todo**:
  commit, PR body, commit title, commit message.
- Python: ver [`.agents/rules/python-patterns.mdc`](.agents/rules/python-patterns.mdc)
  para el detalle completo (pydantic v2, Storage, Pyomo/solver default,
  testing). Desde Fase 3 este repo si usa FastAPI (`services/api/`); el
  worker sigue siendo un loop de polling sobre la DB (`app/db/claim.py`), no
  Celery. No copie convenciones de otro proyecto/stack — Celery, Redis y
  Thori siguen genuinamente ausentes de este repo.

---

*AGENTS.md reescrito 2026-08-05 para reflejar gridforge (el contenido
previo era una plantilla de otro proyecto — referenciaba `thori-overview.mdc`,
`thori-pr-reviewer` y convenciones de Celery/Redis, que no existen en este
repo). FastAPI si existe desde Fase 3 (`services/api/`); Celery, Redis y
Thori siguen ausentes.*
