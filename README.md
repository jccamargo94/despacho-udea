# despacho-udea — modelo academico de despacho electrico colombiano

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
[![Docs](https://img.shields.io/badge/docs-github%20pages-informational.svg)](https://jccamargo94.github.io/despacho-udea/)

Repositorio academico para aproximar el despacho electrico colombiano, comparar
resultados contra informacion publicada por XM y estudiar el efecto de incorporar
BESS (Battery Energy Storage Systems) bajo distintos modos de participacion.

Este README es el punto de entrada para humanos y agentes de IA. Antes de hacer
cambios, lea especialmente las secciones de estado actual, datos requeridos y
brechas conocidas. La documentación pública del proyecto vive también en
[docs/index.md](docs/index.md) y está preparada para GitHub Pages.

## English summary

Academic Pyomo unit-commitment model that approximates Colombia's electricity
dispatch and marginal price, validated against data published by [XM](https://www.xm.com.co/)
(the Colombian grid operator), plus a study of adding BESS (battery) resources
under different market participation modes. Ships as a Python domain library +
Typer CLI, a FastAPI backend with a polling worker, a Next.js frontend, and a
Dockerized dev setup. Jump to [Quickstart](#quickstart) to run it locally, or
[Repository map](#5-mapa-del-repositorio) for the layout. Full docs (Spanish)
continue below and at the [GitHub Pages site](https://jccamargo94.github.io/despacho-udea/).

## Quickstart

Runs with zero external data or credentials — CLI, model, and tests work
against the repo's synthetic fixture out of the box:

```bash
sudo apt-get install coinor-cbc      # solver used by the Pyomo model
uv sync --group dev
uv run pytest -q                     # 138 tests, no external data needed
```

Running the CLI against a real date (`uv run python -m app run 2024-04-18`)
needs XM input files under `data/` (git-ignored, not included — see
[§7](#7-datos-requeridos)) and either downloads them or reads a local copy.
The backend API/worker need `DATABASE_URL` (Postgres) and the frontend needs
Supabase credentials — see [§8](#8-como-ejecutar).

No local Python/solver install? Use Docker — see [§6](#6-instalacion-local).

## Indice

1. [Vision del proyecto](#1-vision-del-proyecto)
2. [Que existe hoy](#2-que-existe-hoy)
3. [Variantes de despacho implementadas](#3-variantes-de-despacho-implementadas)
4. [Conceptos importantes](#4-conceptos-importantes)
5. [Mapa del repositorio](#5-mapa-del-repositorio)
6. [Instalacion local](#6-instalacion-local)
7. [Datos requeridos](#7-datos-requeridos)
8. [Como ejecutar](#8-como-ejecutar)
9. [Resultados](#9-resultados)
10. [Pruebas](#10-pruebas)
11. [Brechas conocidas](#11-brechas-conocidas)
12. [Hacia donde se quiere llegar](#12-hacia-donde-se-quiere-llegar)
13. [Backtesting y escenarios futuros](#13-backtesting-y-escenarios-futuros)
14. [Notas para agentes de IA](#14-notas-para-agentes-de-ia)

---

## 1. Vision del proyecto

El proyecto parte de tres ideas:

1. Construir un **predespacho ideal** con menos restricciones, que produzca un
   precio marginal comparable con referencias de predespacho publicadas por XM.
2. Construir un **despacho ideal** con mas restricciones tecnicas, como
   aproximacion al proceso que determina el precio real de bolsa.
3. Evaluar como cambia el despacho y el precio cuando se agregan baterias BESS
   en distintos niveles de penetracion y con distintas reglas de participacion.

La meta de largo plazo es convertir los scripts y notebooks actuales en una
aplicacion dockerizada, reproducible y operable, con frontend, backend, worker de
ejecucion, almacenamiento de resultados y configuracion de escenarios.

La hoja de ruta extendida esta en
[docs/roadmap-aplicacion-despacho.md](docs/roadmap-aplicacion-despacho.md) y el
sitio público se publica desde [docs/index.md](docs/index.md) con la
formulación matemática en [docs/formulacion-matematica.md](docs/formulacion-matematica.md).

---

## 2. Que existe hoy

La documentación del repositorio ahora se actualiza en dos capas:

- La documentación operativa y de contexto del repo vive en este README, en
  [AGENTS.md](AGENTS.md) y en las reglas de [.agents/rules](.agents/rules).
- La documentación pública para GitHub Pages vive en [docs/index.md](docs/index.md)
y [docs/formulacion-matematica.md](docs/formulacion-matematica.md).

Hoy el repositorio ya no es solo una coleccion de notebooks. Existe una primera
extraccion hacia una aplicacion Python:

- `app/model/`: modelo Pyomo y restricciones del despacho.
- `app/data/`: carga, descarga y parsing de insumos.
- `app/pipeline/`: construccion de casos, ejecucion, guardado y evaluacion.
- `app/cli.py`: CLI Typer para ejecutar corridas desde terminal.
- `tests/`: pruebas unitarias de parsers, rutas, metricas, resultados, CLI y
  orquestacion.
- `notebooks/*.ipynb`: notebooks exploratorios que todavia contienen analisis,
  ETL, graficas y comparaciones no migradas por completo a la app.

El flujo funcional actual es:

1. Resolver archivos de entrada locales o descargados.
2. Parsear OFEI, condiciones iniciales, demanda, disponibilidad y ofertas.
3. Construir `set_data` y `param_data` para Pyomo.
4. Crear y resolver el modelo.
5. Hacer una segunda corrida LP de pricing cuando se requieren precios
   marginales validos.
6. Guardar despacho, precio marginal y metricas cuando hay datos reales de XM.

---

## 3. Variantes de despacho implementadas

Las variantes viven en `DispatchOptions`, dentro de
[app/model/model.py](app/model/model.py).

| `dispatch_type` | significado |
| --- | --- |
| `preideal` | Predespacho ideal contra demanda pronosticada `PrId`. Es el caso base y mas rapido. |
| `ideal` | Despacho ideal con restricciones termicas adicionales: rampas, minimo tiempo en linea, arranques y apagados. |
| `bess_preideal` | Predespacho ideal con BESS. |
| `bess_ideal` | Despacho ideal con BESS. |
| `bess_preideal_resource` | Predespacho con BESS modelada como recurso/activo del sistema mediante objetivo de bienestar social. |
| `bess_ideal_resource` | Despacho ideal con BESS modelada como recurso/activo del sistema. |

### Modos BESS objetivo

Conceptualmente se quiere soportar tres modos:

| modo objetivo | descripcion | estado actual |
| --- | --- | --- |
| Arbitraje independiente | La bateria oferta precios de carga y descarga. | Parcialmente cubierto por `bess_preideal` y `bess_ideal`. |
| Activo del operador/red | El operador optimiza la bateria como activo del sistema y se remunera por energia cargada/descargada. | Parcialmente cubierto por `*_resource`. |
| Generador | La bateria actua como generador que oferta precio de descarga. | Pendiente de formalizar como modo separado. |

Una meta importante es separar estos modos en una configuracion explicita de
escenario BESS, en vez de depender solo del nombre del `dispatch_type`.

---

## 4. Conceptos importantes

- **MPO / precio marginal:** se obtiene como el dual de la restriccion
  `power_balance`. Es el precio marginal del sistema calculado por el modelo.
- **MILP -> pricing LP:** el modelo tiene variables binarias. El dual de una
  solucion MILP no es un precio marginal valido. Por eso
  `UnitCommitmentModel.solve(..., compute_prices=True)` fija las variables
  enteras despues de resolver el MILP y resuelve un LP de pricing. Los precios
  deben leerse despues de esa segunda corrida.
- **Metricas:** las metricas estan en [app/utils/metrics.py](app/utils/metrics.py).
  Se usan RMSE, MAE, bias, WAPE y sMAPE. Se evita MAPE porque en sistemas con
  precios cercanos a cero puede explotar y dar lecturas poco utiles.
- **Unidades:** parte del codigo hace conversiones de unidades al cargar datos.
  No agregue conversiones nuevas sin revisar [app/data/loaders.py](app/data/loaders.py)
  y las pruebas asociadas.

---

## 5. Mapa del repositorio

```text
app/
  cli.py            # CLI Typer: run, fetch, evaluate, compare
  __main__.py       # habilita python -m app
  dates.py          # parsing de fechas: dia, rango, mes o todo
  model/            # UnitCommitmentModel y restricciones Pyomo
  data/
    download.py     # descarga archivos XM por fecha
    ofei.py         # parser del archivo OFEI
    loaders.py      # carga CSVs base y aplica conversiones
    actuals.py      # carga precios/despacho reales para evaluacion
    paths.py        # resuelve ubicaciones historicas y descargadas
  pipeline/
    case_builder.py # fecha + configuracion -> set_data, param_data, meta
    results.py      # extrae MPO/despacho/BESS y guarda resultados
    runner.py       # orquesta build -> solve -> save -> evaluate
    evaluate.py      # re-score post-hoc de una corrida ya guardada
    scenarios.py     # carga escenarios BESS declarativos (scenarios/bess/*.yaml)
  schemas/           # modelos pydantic v2: DispatchCase, InputPack, RunResult, BessScenario
  storage/           # abstraccion Storage (LocalStorage hoy; GCS a futuro)
  db/                # modelos SQLAlchemy y acceso a datos (runs, metric_sets)
  utils/
    metrics.py      # metricas de evaluacion
    misc.py         # compatibilidad hacia app.data.download

services/
  api/              # backend FastAPI (Fase 3): crea/lista corridas, sirve resultados
  worker/           # worker de polling sobre la DB (app/db/claim.py); no usa Celery

frontend/            # app Next.js (Fase 4): configurar y lanzar corridas, ver resultados
alembic/             # migraciones de la base de datos (alembic upgrade head)

scenarios/bess/      # escenarios BESS declarativos (YAML) usados por --bess-scenario
run_dispatch.py     # runner legado de una fecha; conserva compatibilidad
get_date_results.py # runner legado batch
notebooks/*.ipynb   # notebooks exploratorios y ETL no migrado

docs/
  index.md                        # landing page del sitio (GitHub Pages, Jekyll)
  formulacion-matematica.md       # formulacion matematica publicada
  roadmap-aplicacion-despacho.md  # vision y fases hacia app dockerizada
  superpowers/specs/              # diseno de la CLI actual
  superpowers/plans/              # plan de implementacion de la CLI actual

.github/workflows/pages.yml  # publica docs/ a GitHub Pages en push a main/develop

tests/               # suite pytest
data/                # insumos y resultados; git-ignored
solver/              # artefactos locales del solver; git-ignored

docker/
  Dockerfile.cli            # imagen CLI runtime (CBC + deps, sin data/ ni notebooks)
  Dockerfile.api             # imagen backend FastAPI
  Dockerfile.worker          # imagen worker de polling
  docker-compose.yml         # cli, api y worker (api/worker bajo profile "backend")
  docker-compose.dev.yaml    # api + worker + frontend con hot-reload, para dev local
.dockerignore         # se queda en la raiz: el build context de docker/ sigue siendo la raiz
```

---

## 6. Instalacion local

Requiere Python 3.12 (gestionado via `uv python install 3.12`, ver
`.python-version`) y un solver compatible con Pyomo. El flujo documentado
usa **CBC** en el `PATH`.

En Debian/Ubuntu:

```bash
sudo apt-get install coinor-cbc
```

Entorno Python:

```bash
uv sync --group dev
```

Para trabajar con los notebooks exploratorios de `notebooks/`, agregar el extra:

```bash
uv sync --group dev --extra notebooks
```

Los notebooks asumen que el directorio de trabajo del kernel es la raiz del
repo (leen/escriben rutas como `data/...` relativas a la raiz), no
`notebooks/`. En VS Code esto ya esta configurado via
`jupyter.notebookFileRoot` en `.vscode/settings.json`. Con Jupyter classic/Lab
por fuera de VS Code, lance `jupyter lab` desde la raiz del repo o ejecute
`%cd ..` como primera celda antes de correr el resto del notebook.

Verificacion basica:

```bash
uv run python -c "import pyomo.environ as pyo; print('cbc', pyo.SolverFactory('cbc').available())"
uv run pytest -q
```

### Con Docker (alternativa a instalacion local)

Requiere Docker. No necesita `uv` ni CBC instalados en el host — ambos
viven dentro de la imagen. Todos los Dockerfiles y archivos compose viven en
[`docker/`](docker/); el contexto de build sigue siendo la raiz del repo, por
eso los comandos usan `-f docker/...` y `--project-directory .` (ancla
`./data`, `./.env` y el build context a la raiz en vez de a `docker/`).

```bash
docker build -f docker/Dockerfile.cli -t despacho-udea .
docker run --rm --entrypoint uv despacho-udea run --no-sync python -c \
  "import pyomo.environ as pyo; print('cbc', pyo.SolverFactory('cbc').available())"
```

Para correr contra datos reales, montar `data/` como volumen (ver
`docker/docker-compose.yml`, servicio `cli`):

```bash
docker compose --project-directory . --env-file .env -f docker/docker-compose.yml run --rm cli run 2024-04-18 -t preideal
```

---

## 7. Datos requeridos

`data/` esta ignorado por Git. Un clon limpio no trae datos reales.

El codigo soporta dos layouts:

1. Layout historico/offline organizado.
2. Layout descargado por fecha en `data/{YYYY-MM-DD}/`.

`app/data/paths.py` intenta primero el layout historico y luego el layout
descargado.

### Archivos que cambian por fecha

| tipo | ubicacion historica/offline | ubicacion descargada |
| --- | --- | --- |
| OFEI | `data/oferta_inicial/OFEI{MMDD}.txt` | `data/{YYYY-MM-DD}/OFEI{MMDD}.txt` |
| dCondIniP / dCondIniU | `data/condicion_inicial/{YYYY-MM-DD}/dCondIni*{MMDD}.txt` | `data/{YYYY-MM-DD}/dCondIni*{MMDD}.txt` |
| PrId | `data/predespacho_ideal/PrId{MMDD}_NAL.txt` | `data/{YYYY-MM-DD}/PrId{MMDD}_NAL.txt` |
| iMAR | `data/predespacho_ideal/iMAR{MMDD}_NAL.txt` | `data/{YYYY-MM-DD}/iMAR{MMDD}_NAL.txt` |

### CSVs base

Estos archivos son consumidos por [app/data/loaders.py](app/data/loaders.py):

```text
data/dispo_declarada.csv
data/ofertas.csv
data/demaCome.csv
data/agc_asignado.csv
data/parametros_plantas.csv
data/precio_bolsa/precio_bolsa_2024.csv
data/DispoCome_resource.csv   # requerido para tipos ideal
```

### Archivos auxiliares

```text
data/ramps.json
data/preideal_dispatch_map.json
data/error_map.json
data/Supuestos Modelo de despacho.xlsx
```

### Datos reales para evaluacion

```text
data/preideal_price/{YYYY-MM-DD}.txt
data/preideal_dispatch/{YYYY-MM-DD}.txt
```

### Descarga desde XM

`app/data/download.py` descarga archivos por fecha hacia `data/{YYYY-MM-DD}/`.
La CLI invoca `ensure_data_for_date()` cuando necesita datos por fecha.

Los CSVs base todavia dependen principalmente de `notebooks/data_fetcher.ipynb`;
esa ETL no esta migrada por completo a la aplicacion.

---

## 8. Como ejecutar

La CLI es el camino recomendado. Tiene cuatro comandos:

| comando | que hace |
| --- | --- |
| `run` | Construye y resuelve casos de despacho por fecha/tipo, guarda resultados y (opcionalmente) evalua contra XM. |
| `fetch` | Descarga insumos crudos de XM para una o mas fechas, sin correr el modelo. |
| `evaluate` | Re-calcula metricas de una corrida ya guardada contra XM, sin volver a resolver el modelo. |
| `compare` | Hace un outer join de `metrics-summary.csv` de dos corridas para comparar escenarios. |

```bash
python -m app run 2024-04-18
python -m app run 2024-04-18 -t ideal
python -m app run 2024-04-18:2024-04-30 -t all
python -m app run 2024-04
python -m app run
python -m app fetch 2024-04-18:2024-04-30
python -m app evaluate 2024-04-18 -t preideal
python -m app compare data/results/baseline data/results/20pct_arbitrage
```

Opciones utiles:

| opcion | default | significado |
| --- | --- | --- |
| `-t, --type` | `preideal` | Tipo de despacho. Es repetible. `all` ejecuta todos los tipos. |
| `--solver` | `cbc` | Solver Pyomo. |
| `--eval/--no-eval` | eval | Calcula metricas contra XM cuando hay datos reales. |
| `--prices/--no-prices` | prices | Ejecuta o salta el LP de pricing. Sin pricing, el MPO no es confiable. |
| `--skip-dates` | vacio | Fechas `YYYY-MM-DD` separadas por coma para omitir. |
| `--out` | `data/results` | Directorio de salida. |
| `--data-dir` | `data` | Directorio base de insumos. |

Uso programatico:

```python
from datetime import date
from app.model import DispatchConfig
from app.pipeline.runner import run_case

res = run_case(date(2024, 4, 18), DispatchConfig("preideal"), solver="cbc")
print(res.ok, res.metrics)
```

`run_dispatch.run_dispatch(...)` se conserva para notebooks y compatibilidad.

Mismo comando via Docker (ver seccion 6):

```bash
docker compose --project-directory . --env-file .env -f docker/docker-compose.yml run --rm cli run 2024-04-18 -t preideal
```

### Backend API, worker y migraciones (Fase 3)

Desde Fase 3 el repo tambien incluye un backend HTTP (`services/api/`) y un
worker que ejecuta corridas por polling de la base de datos (`services/worker/`).

Local (requiere `DATABASE_URL` en el entorno; la API ademas requiere
`SUPABASE_JWKS_URL` para verificar JWTs de Supabase):

```bash
uv run uvicorn services.api.main:app --reload
uv run python -m services.worker.main
```

Con Docker (requiere un archivo `.env` en la raiz — copiar `.env.example`):

```bash
docker compose --project-directory . --env-file .env --profile backend -f docker/docker-compose.yml up --build
```

Migraciones (requiere `DATABASE_URL` en el entorno):

```bash
uv run alembic upgrade head
```

Recuperacion manual de una corrida atascada: si una fila `runs` queda en
`running` de forma permanente (p. ej. el worker murio a mitad de un solve),
resetear su `status` a `pending` para que el worker la re-reclame. Si la
corrida ya habia escrito una fila en `metric_sets` antes de atascarse, borrar
esa fila primero — `metric_sets.run_id` es unico, y una segunda escritura
fallara con un error de constraint.

### Frontend (Fase 4)

Desde Fase 4 el repo incluye un frontend Next.js (`frontend/`) construido con
pnpm y TypeScript. Permite configurar parametros de ejecucion, lanzar corridas
y consultar resultados a traves de la interfaz web.

Local (requiere `frontend/.env.local` copiado desde `frontend/.env.local.example`
con URLs/claves reales de Supabase y backend; tambien requiere `FRONTEND_ORIGIN`
en el `.env` de la raiz para que el backend CORS permita el frontend):

```bash
cd frontend
pnpm install
pnpm dev
```

El frontend se conecta al backend FastAPI via `NEXT_PUBLIC_API_BASE_URL`. El backend
requiere que `FRONTEND_ORIGIN` esté configurado (p. ej. `http://localhost:3000`)
en la raiz `.env` para permitir requests CORS desde el navegador.

Con Docker, para levantar api + worker + frontend juntos en modo desarrollo
(hot-reload; requiere `.env` en la raiz y `frontend/.env.local`, ver arriba):

```bash
docker compose --project-directory . --env-file .env -f docker/docker-compose.dev.yaml up --build
```

---

## 9. Resultados

Por cada `(fecha, tipo)` se escriben archivos en `data/results/`:

```text
dispatch_by_gen-{date}-{type}.csv
marginal_price-{date}-{type}.csv
metrics-{date}-{type}.csv
```

Cuando hay evaluacion, tambien se genera:

```text
data/results/metrics-summary.csv
```

El resumen incluye metricas como RMSE, MAE, bias, WAPE, sMAPE y R2.

Para escenarios BESS, una brecha actual es guardar de forma mas completa carga,
descarga, SOC, costos/ingresos y remuneracion. Hoy el pipeline guarda el
despacho y el precio marginal, pero la salida BESS debe fortalecerse.

---

## 10. Pruebas

```bash
pytest -q
```

Las pruebas cubren:

- parsing de fechas;
- parser OFEI;
- loaders;
- actuals;
- resolucion de paths;
- extraccion de resultados;
- pricing fix en un modelo pequeno;
- metricas;
- aislamiento de fallas en `runner`;
- CLI.

Estas pruebas no validan por si solas que `case_builder` reproduzca resultados
historicos con datos reales. Para eso hacen falta fixtures doradas o corridas
comparativas contra los scripts previos.

---

## 11. Brechas conocidas

- `app/pipeline/case_builder.py` fue extraido desde scripts legados, pero
  necesita validacion end-to-end con datos reales.
- La ETL de CSVs base sigue principalmente en notebooks.
- Los modos BESS no estan modelados aun como una interfaz de escenario clara.
- Falta una salida BESS completa: carga, descarga, SOC, pagos/remuneracion y
  comparaciones por escenario.
- Los supuestos para correr semanas futuras no estan formalizados en modulos de
  forecasting (Fase 5, aun sin diseno).

---

## 12. Hacia donde se quiere llegar

La aplicacion objetivo deberia tener:

| componente | responsabilidad |
| --- | --- |
| Libreria de dominio | Mantener modelo, loaders, construccion de casos, evaluacion y resultados. |
| CLI | Ejecutar flujos reproducibles para desarrollo, batch y automatizacion. |
| Backend API | Crear escenarios, lanzar corridas, consultar estado y descargar resultados. |
| Worker | Ejecutar corridas largas, descargas, ETL, solver y evaluacion. |
| Frontend | Configurar fechas, escenarios BESS, modo de participacion y ver resultados. |
| Base de datos | Persistir escenarios, corridas, estados, metricas y metadatos. |
| Storage | Guardar insumos XM, artefactos intermedios y resultados pesados. |
| Docker | Hacer reproducible el entorno con solver, dependencias y volumenes de datos. |

### Interfaces de dominio que conviene estabilizar

Antes de construir API/frontend, conviene cerrar estas piezas:

- `DispatchCase`: fecha, tipo de despacho, solver, fuente de datos.
- `BessScenario`: potencia, energia, eficiencia, SOC, modo de mercado y ofertas.
- `InputPack`: insumos historicos, descargados o pronosticados, con version y
  checksum.
- `RunResult`: precios, despacho, resultados BESS, metricas, logs y errores.

### Fases sugeridas

1. **Estabilizar el core** — hecho parcialmente: `case_builder` validado
   end-to-end contra fixture sintetico (Fase 2B); sigue sin validar contra
   datos historicos reales de XM (ver seccion de brechas).
2. **Dockerizar** — hecho (Fase 2C): imagen con solver, dependencias runtime.
3. **Persistir ejecuciones** — hecho (Fase 3): DB + modelo de runs/metric_sets.
4. **Agregar backend y worker** — hecho (Fase 3): `services/api/`, `services/worker/`.
5. **Agregar frontend** — hecho (Fase 4): `frontend/`, interfaz para configurar,
   lanzar y comparar corridas.
6. **Forecast** — pendiente (Fase 5, sin diseno aun): producir insumos futuros
   con supuestos explicitos de precios de oferta, disponibilidad y demanda.

---

## 13. Backtesting y escenarios futuros

Para medir precision historica:

- comparar precio marginal modelo vs precio publicado por XM;
- comparar despacho por recurso/tecnologia cuando haya datos;
- reportar errores por hora, dia, mes y bloque horario;
- comparar caso base vs escenarios BESS.

Para correr semanas futuras:

- precios de oferta: iniciar con mediana historica por recurso y tipo de dia,
  con fallback por tecnologia;
- disponibilidad: usar un baseline seasonal-naive por recurso/hora/tipo de dia;
- demanda: usar forecast publicado si existe; si no, baseline horario ajustado
  por tendencia reciente.

Estos supuestos deben quedar versionados y separados de los datos historicos
reales para no mezclar corridas observadas con corridas hipoteticas.

---

## 14. Notas para agentes de IA

- No asuma que `data/` existe; esta git-ignored.
- No afirme que el modelo completo funciona solo porque pasan las pruebas.
  `case_builder` necesita validacion con datos reales.
- No lea precios marginales de una solucion MILP sin el pricing LP.
- No limpie o refactorice la logica de `case_builder` sin una prueba dorada.
- Antes de cambios grandes, revise
  [docs/roadmap-aplicacion-despacho.md](docs/roadmap-aplicacion-despacho.md),
  `docs/superpowers/specs/` y `docs/superpowers/plans/`.
