# Fase 2: Docker y ejecucion reproducible — diseno

Fecha: 2026-08-05
Roadmap: `docs/roadmap-aplicacion-despacho.md`, seccion "Fase 2".

## Contexto

Fase 1 (PR #2) mergeada a `develop`: Storage layer, escenarios BESS,
`fetch`/`evaluate`/`compare`. 78/78 tests verdes en Python 3.10.14 local.

El roadmap enumera Fase 2 como cuatro items ("Dockerfile con solver",
"docker-compose con volumen", "separar deps runtime/notebooks", "smoke test
en contenedor"). La investigacion previa a este diseno encontro que cada uno
esconde una decision no trivial que cambia el alcance real. Cada hallazgo
esta verificado contra codigo/ejecucion real, no asumido — ver seccion 6.

Esta sesion se ejecuto sin el usuario presente para las decisiones finales
(delegacion explicita); las decisiones de esta seccion se tomaron con
`/advisor` como segunda opinion en los puntos de mayor riesgo (solver
default, alcance del fixture). Marcadas como tal donde aplica.

## Decomposicion y orden

Cuatro frentes, con una dependencia real entre ellos que no es obvia a
primera vista:

- **A. Toolchain**: `requirement.txt` -> `uv` + `pyproject.toml` +
  `uv.lock`, Python 3.10 -> 3.12, pydantic v1 -> v2, pre-commit (`ruff`
  bloqueante, `ty` no bloqueante).
- **B. Fixture XM real**: dataset sintetico pero con formato real que
  sobrevive `case_builder.build_case` completo, para un smoke test
  end-to-end genuino.
- **C. Dockerfile + compose**: imagen con solver(es), servicio CLI,
  placeholders para `services/api`/`services/worker` (Fase 3).
- **D. Solver default**: decidido durante este diseno — **no cambia** (ver
  seccion 3). Absorbido por A (dos lineas en `model.py`).

Orden obligatorio: **A -> B -> C**. La razon es una restriccion de
atribucion de fallos: el smoke test en Docker (fase C) va a correr el
fixture de B. Si B no esta primero verificado en verde **en el host**, un
smoke test rojo en el contenedor es inatribuible — no se puede saber si
fallo el fixture o el contenedor. Ese es exactamente el tipo de bug
"funciona en mi maquina" que este smoke test existe para atrapar; construir
en el orden equivocado destruye la señal.

## 1. Toolchain: uv + Python 3.12 + pydantic v2

`requirement.txt` es un freeze de 200+ paquetes que incluye el entorno
completo de notebooks (Jupyter, PyQt5, Orange3, catboost, xgboost). Se
reemplaza por `pyproject.toml` gestionado con `uv`, con dos grupos:

- **Runtime** (lo que instala la imagen Docker): `pandas`, `numpy`, `pyomo`,
  `thefuzz`, `typer`, `requests`, `plotly`, `openpyxl`, `xlrd`, `highspy`,
  `pydantic`, `pyyaml`, `pydataxm` — el set que `README.md` seccion 6 ya
  documenta como runtime, mas `highspy`/`pyyaml`/`pydataxm` que el README no
  listaba pero el codigo importa (`app/model/model.py`, `app/pipeline/
  scenarios.py`, `app/data/download.py`).
- **Notebooks** (grupo opcional, fuera de la imagen): derivado leyendo los
  `import` reales de los 11 `.ipynb` en la raiz, no copiado del freeze
  completo — el freeze incluye paquetes de entornos de trabajo previos
  (`Orange3`, `catboost`, `xgboost`, `PyQt5`) que ningun notebook de este
  repo importa. Verificar con `jupyter nbconvert --to script` + grep de
  imports antes de fijar la lista final.
- `pytest` es dev-only, no runtime (el README actual lo lista mal como
  runtime). La imagen Docker no lo instala salvo target de test explicito.

**Pin obligatorio verificado**: `numpy<2`. Se probo instalar el stack en un
venv Python 3.12 limpio; con `numpy` sin pin, `import pyomo.environ` falla
en seco:

```
AttributeError: `np.float_` was removed in the NumPy 2.0 release. Use `np.float64` instead.
```

Es un shim de compatibilidad interno de Pyomo 6.7.3 (`pyomo/common/
dependencies.py`), no un problema de Python 3.12 en si. Con `numpy==1.26.4`
(el mismo pin que ya trae `requirement.txt`), el import funciona limpio.
Confirmado tambien que `pydataxm`, `highspy`, `pyomo`, `thefuzz`, `pydantic`
instalan e importan sin fallos adicionales en Python 3.12.

**Pydantic v1 -> v2**: un solo `@validator` en todo el repo
(`app/schemas/bess.py:29`, `BessScenario._check_bids`). Migracion acotada:

```python
# antes
@validator("units")
def _check_bids(cls, units, values): ...
    mode = values.get("mode")

# despues
@field_validator("units")
@classmethod
def _check_bids(cls, units, info: ValidationInfo): ...
    mode = info.data.get("mode")
```

`mode` esta declarado antes que `units` en `BessScenario`, asi que
`info.data` ya lo tiene disponible en v2 (el orden de validacion de campos
importa en pydantic v2). Unico punto de riesgo de esta migracion; el resto
de los schemas (`InputPack`, `RunResult`, `DispatchCase`, `BessUnit`) son
`BaseModel` planos sin validators ni `Config` custom, migran sin cambios de
comportamiento.

**Pre-commit**: `ruff` (lint + format) como hook bloqueante. `ty` explicito
**no bloqueante** en esta fase — es pre-alpha y este repo nunca paso por un
type-checker; la primera corrida seria una pared de errores que bloquearia
cualquier commit de Fase 2, incluidos los de este mismo trabajo. Se deja
configurado para correr en modo informativo (o en un job de CI separado que
no falla el build), no como gate.

**Criterio de salida de A**: 78/78 tests verdes en Python 3.12 con el nuevo
toolchain, antes de tocar el fixture (B) o Docker (C).

## 2. Fixture XM real + smoke test end-to-end

Decision explicita, tomada dos veces con el costo real por delante: el
smoke test corre `python -m app run <fecha> -t preideal` de verdad, contra
un fixture sintetico que sobrevive `case_builder.build_case` completo — no
un caso de juguete que hace monkeypatch de `build_case` (ese patron ya
existe en `tests/test_runner.py::_toy_case`, pero no ejercita el parsing XM
real, que es la parte que un contenedor puede romper de forma no obvia:
encodings, paths, permisos).

Esto absorbe, con datos sinteticos, el trabajo que la Fase 0 del roadmap
todavia tiene pendiente ("verificar `case_builder` con datos reales") — se
documenta como tal, no como si fuera trabajo nuevo de Fase 2 nada mas.

### Que necesita el fixture, verificado contra el codigo real

`case_builder.build_case` (con `ders=None`, el default de la CLI — evita
necesitar el Excel `Supuestos Modelo de despacho.xlsx`) lee, para `level=
preideal`:

- `data/dispo_declarada.csv` — columnas usadas: `datetime`, `resource_name`,
  `dispo`, `gen_type` (`app/data/loaders.py::load_dispo`,
  `case_builder.py` filtra por `gen_type == "TERMICA"`).
- `data/ofertas.csv` — `Date`, `resource_name`, `Value`.
- `data/demaCome.csv` — `datetime`, `dema`.
- `data/agc_asignado.csv` — `datetime`, `recurso`, `agc`.
- `data/parametros_plantas.csv` — `generador`, `TMG`.
- `data/precio_bolsa/precio_bolsa_2024.csv` — `datetime`, `precio_bolsa`
  (el loader multiplica x1e3; misma leccion de unidades del bug de Fase 1 —
  MPO se maneja en COP/MWh en todo el pipeline, no en COP/kWh).
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
- `data/condicion_inicial/{fecha}/dCondIniP{MMDD}.txt` y `dCondIniU{MMDD}
  .txt` — CSV simple con headers en la primera linea (`open().readlines()`
  + split manual, no `pd.read_csv`).
- `data/predespacho_ideal/PrId{MMDD}_NAL.txt` — CSV sin headers,
  encoding latin1, una fila por generador con 24 columnas horarias.
- `data/ramps.json`, `data/preideal_dispatch_map.json` — configuracion
  estatica en la raiz de `data_dir`, no por fecha.

### Guardas identificadas en `build_case` que el fixture debe respetar

Nombres deben ser consistentes entre los 5+ archivos para que cada
`thefuzz.process.extractOne(..., score_cutoff=70)` no caiga por debajo del
cutoff — si cae, el codigo indexa `[0]` sobre `None` y revienta con
`TypeError` en tres puntos distintos (`price_bid_map`:133, `CC_MAP`:180,
`generators_pap_map`:305). El fixture debe usar el mismo nombre de recurso
(o uno con similitud >70% via `token_sort_ratio`/`partial_token_sort_ratio`)
en `dispo_declarada.csv`, `ofertas.csv`, OFEI, `dCondIniP/U`, y
`parametros_plantas.csv`.

`build_case:70` llama `ensure_data_for_date(fecha, data_dir=dd)`, que
intenta descargar de XM si la carpeta no existe. El fixture debe quedar en
un layout que `ensure_data_for_date` reconozca como "ya presente" (mismo
chequeo que usa `download.py`), para que sea un no-op — si no, el smoke
test en el contenedor falla por aislamiento de red, por una razon que no
tiene nada que ver con Docker.

### Alcance del fixture

Fecha sintetica fija, 2-3 generadores termicos, sin BESS, sin DERS, 24
horas, demanda plana o triangular simple. El esquema exacto de cada archivo
(columnas, tipos, encoding) se deriva en la fase de plan directamente de
los loaders/parsers citados arriba, no se transcribe byte a byte aqui.

**Criterio de salida de B**: `python -m app run <fecha-fixture> -t preideal`
corre limpio **en el host** (Python 3.12, toolchain de A), produce
`metrics-*.csv`/resultados sin error, MPO no vacio. Solo entonces se usa
como smoke test de C.

## 3. Solver: cbc se queda como default global

Punto de mayor riesgo de todo este diseno, y el que mas cambio respecto a
la intencion inicial. Decision tomada en dos pasadas:

**Primera pasada** (antes de verificar): cambiar el default global de
`"cbc"` a `"appsi_highs"` en `DispatchCase.solver` y `--solver` de la CLI —
`highspy` es pip puro (sin capa apt), y el patron fix-and-resolve que ya
existe en `UnitCommitmentModel._solve_pricing_lp` (fija enteros al optimo
del MILP, resuelve de nuevo como LP para duales validos) parecia resolver
de raiz la preocupacion original del usuario sobre duales de HiGHS en MILP.

**Verificacion empirica que revirtio la decision**: se ejecuto el mismo
modelo de juguete de `tests/test_results.py` (`beta A=10, B=50, demand=150
-> MPO esperado 50.0`) con `solver="appsi_highs"` a traves de
`pyo.SolverFactory("appsi_highs")` — el unico camino que usa el codigo hoy
(`app/model/model.py:339,363`, siempre via `pyo.SolverFactory(solver, ...)`
con `solver` como string). Resultado:

```
RuntimeError: Solver does not currently have valid duals. Please check the termination condition.
```

Esto **no es la limitacion de duales-en-MILP** que se sospechaba
originalmente (esa ya esta cubierta por el patron fix-and-resolve, que
convierte el problema en un LP genuino antes de leer duales). Es un bug
distinto y mas grave: el crash ocurre **incluso con `compute_prices=False`,
en el primer solve MILP**, sin llegar nunca al resolve de pricing. Es decir,
`appsi_highs` via el wrapper legacy de Pyomo no es usable hoy para *ningun*
solve de este modelo, ni como default ni como flag opcional — no es un
problema de precios, es una falla total.

**Causa raiz aislada**: el crash es especifico del wrapper legacy
(`pyo.SolverFactory("appsi_highs")`, clase `LegacySolver`), no de HiGHS ni
de `appsi` en general. El mismo LP (post fix-and-resolve, con las variables
enteras fijadas y dominio relajado a `Reals`) resuelto con la interfaz
nativa no-legacy (`pyomo.contrib.appsi.solvers.highs.Highs()` instanciada
directamente, sin pasar por `SolverFactory`) reporta `dual_valid=True` y,
leyendo los duales manualmente y cargandolos en el `Suffix`, produce
`MPO=50.0` — identico al resultado de `cbc`. Confirmado por ejecucion
directa, no inferido.

**Decision**: `cbc` se queda como default global (`DispatchCase.solver`,
`--solver` de la CLI) — sin cambios de comportamiento. `highspy` se instala
en la imagen (dependencia liviana, sin apt) pero **no se documenta como
solver alternativo usable** todavia: usarlo hoy vía el camino normal del
codigo revienta en el primer solve. La correccion real (usar la interfaz
nativa de `appsi` para el caso `highs`, en vez del wrapper legacy) es un
cambio de codigo en `app/model/model.py`, validado y con una via de arreglo
conocida, pero es trabajo de la capa de modelo, no de Docker — queda fuera
de Fase 2, documentado aqui como hallazgo con reproduccion.

**Cambio de una linea que si entra en Fase 2** (parte de A, no de C):
`app/model/model.py:326` y `:352` tienen `solver: str = "appsi_highs"` como
default de parametro — hoy sin efecto porque `runner.py:33` siempre pasa
`case.solver` explicitamente, pero es un default vivo para cualquier
llamador futuro de `model.solve()` sin argumentos (incluyendo tests). Se
cambia a `"cbc"` para que el default de la funcion coincida con el default
real del sistema; cero riesgo numerico, dos lineas.

**Impacto en tests**: ninguno. `tests/test_schemas_case.py:10`
(`assert c.solver == "cbc"`), `tests/test_runner.py`, `tests/test_results.py`
pasan `solver="cbc"` explicito o verifican el default actual — ambos siguen
siendo ciertos sin tocar los tests.

## 4. Dockerfile

- Base: `python:3.12-slim`.
- `apt-get install -y --no-install-recommends coinor-cbc`, mismo paquete
  Debian/Ubuntu que ya documenta `README.md` seccion 6 y que esta instalado
  en este entorno de desarrollo (`coinor-cbc 2.10.11+ds1-2`).
- `uv sync` instala solo el grupo runtime (sin `notebooks`, sin dev/test).
- `highspy` viaja en el grupo runtime via pip normal (parte del `uv sync`,
  sin paso adicional) — instalado pero no usado como default (seccion 3).
- `CMD`/entrypoint invoca `python -m app`, args via `docker run ... <args>`.
- Imagen no incluye `data/` (git-ignored, se monta como volumen).

## 5. docker-compose.yml

- Servicio `cli`: build de la imagen anterior, monta `./data:/app/data`,
  comando por defecto delega a los args del CLI.
- Placeholders **vacios** para `api`/`worker` (Fase 3 del roadmap, todavia
  sin codigo) usando `profiles:` de Compose (ej. `profiles: ["future"]`) de
  forma que `docker compose up` (sin profile) no los levanta y `docker
  compose config` no falla — un service real apuntando a un `Dockerfile` en
  `services/api/` que no existe todavia rompe la validacion del compose
  file. Alternativa evaluada y descartada: bloques comentados en YAML (mismo
  efecto documental, pero `profiles:` es validable por el propio Compose sin
  depender de que nadie descomente a mano correctamente).

## 6. Resumen de hallazgos verificados (no asumidos)

Cada uno de estos se confirmo por lectura de codigo o ejecucion directa
durante este diseno, no se tomo del prompt de continuacion ni se asumio:

1. PR #2 (Fase 1) estaba realmente mergeado a `develop` al momento de
   empezar (el prompt de continuacion lo daba por hecho; se verifico con
   `gh pr view` antes de confiar en el estado del repo).
2. `_solve_pricing_lp` ya implementa fix-and-resolve — la preocupacion
   original sobre duales de HiGHS en MILP ya esta cubierta por diseno,
   independientemente del solver.
3. `pyo.SolverFactory("appsi_highs")` revienta en **todo** solve de este
   modelo, no solo en el resolve de pricing — hallazgo mas grave que la
   preocupacion original, causa raiz aislada al wrapper legacy.
4. Existe una via de arreglo verificada (interfaz nativa de `appsi`) que
   produce el mismo MPO que `cbc` — confirmado por ejecucion, no solo
   plausible.
5. `numpy<2` es un pin obligatorio para Python 3.12 con Pyomo 6.7.3 —
   verificado con un venv 3.12 limpio, no supuesto por compatibilidad de
   version de Python en abstracto.
6. CC vacio (`{}`) sobrevive limpio el bloque de sintesis de recursos de
   ciclo combinado en `case_builder.py` — verificado por ejecucion aislada
   del bloque, no inferido leyendo el codigo. Reduce el fixture de B a solo
   generadores termicos simples.
7. `DERS=None` (default real de la CLI) evita la dependencia del Excel
   `Supuestos Modelo de despacho.xlsx` en el fixture.

## Brechas conocidas, explicitamente fuera de alcance de Fase 2

1. **Gap #1 heredado de Fase 1** (`evaluate` no actualiza `metrics-summary
   .csv`, `compare` no lo ve): se difiere. Es una brecha de comportamiento
   del pipeline, no de empaquetado/Docker. Se retoma cuando Fase 3
   (frontend/API) lo necesite.
2. **Fix del wrapper legacy de `appsi_highs`** (seccion 3): hallazgo
   completo, causa raiz aislada, via de arreglo verificada — pero es un
   cambio en la capa de modelo (`app/model/model.py`), no en Docker/toolchain.
   Se documenta aqui como el punto de partida para cuando se decida
   abordarlo.
3. Los dos bloques `open(resolve_input(...))` sin pasar por `Storage` en
   `case_builder.py` (dCondIniP/dCondIniU) — decision deliberada ya
   documentada en el plan de Fase 1, no reabierta aqui.
