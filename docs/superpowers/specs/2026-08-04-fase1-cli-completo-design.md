# Fase 1: libreria confiable y CLI completo — diseno

Fecha: 2026-08-04
Roadmap: `docs/roadmap-aplicacion-despacho.md`, seccion "Fase 1".

## Contexto

Interfaces de dominio (`DispatchCase`, `BessScenario`, `InputPack`,
`RunResult`) ya congeladas y mergeadas a `develop` (PR #1). Este diseno cubre
el siguiente salto del roadmap: completar el CLI (`fetch`, `run`, `evaluate`,
`compare`), exponer escenarios BESS por archivo, persistir resultados BESS, y
un resumen consolidado por corrida. Se agrega ademas una capa de storage
(local/GCS) descubierta como bloqueante durante el diseno: hoy todo el I/O de
archivos (`app/data/*.py`, `app/pipeline/results.py`, `app/cli.py`) asume
rutas de filesystem local via `data_dir: str` plano.

## 0. Capa de storage (local / GCS)

Todo el I/O de archivos (insumos XM descargados, resultados de corrida,
biblioteca de escenarios BESS) pasa por una interfaz `Storage`, para que
local y GCS sean intercambiables sin tocar el resto del pipeline.

`app/storage/base.py`:

```python
class Storage(Protocol):
    def exists(self, path: str) -> bool: ...
    def open(self, path: str, mode: str = "r") -> ContextManager[IO]: ...
    def list_dir(self, path: str) -> list[str]: ...
```

- `LocalStorage(root: str)`: wrapper delgado sobre `pathlib`/`open()` nativo.
  Cero dependencias nuevas.
- `get_storage(root: str) -> Storage`: factory. Prefijo `gs://` ->
  `raise NotImplementedError("GCS backend not implemented yet")`. Cualquier
  otra cosa -> `LocalStorage(root)`.
- GCS (`GcsStorage`) queda deliberadamente sin implementar en esta fase (no
  hay bucket/credenciales en uso todavia; evita anadir `google-cloud-storage`
  sin caso de uso real). El roadmap ubica "storage de archivos" como servicio
  en fases posteriores (Fase 2/3); aqui solo se deja el contrato listo para
  que un `GcsStorage` futuro implemente el mismo `Protocol` sin cambiar
  llamadores.

Selección de backend reusa los flags existentes (`--data-dir`, `--out`) — no
hay flags nuevos. `gs://bucket/prefix` sera un valor valido el dia que
`GcsStorage` exista.

**Llamadores migrados** (firmas publicas sin cambios — cada funcion sigue
recibiendo `data_dir: str`/`out: str`; internamente resuelve
`storage = get_storage(data_dir)` y opera con rutas relativas a traves de
`storage`):

- `app/data/loaders.py`: `load_dispo`, `load_ofertas`, `load_demanda`,
  `load_agc`, `load_parametros_plantas`, `load_precio_bolsa`,
  `load_dispo_come` — `pd.read_csv(storage.open(path))`.
- `app/data/paths.py::resolve_input` — `storage.exists(...)` por candidato.
- `app/data/actuals.py`: `load_actual_price`, `load_actual_dispatch`.
- `app/data/download.py`: `ensure_data_for_date` (chequeo de carpeta ->
  `storage.list_dir` no vacio), `save_file` (`storage.open(path, "w")`).
- `app/pipeline/case_builder.py`: los dos bloques de `open(...).readlines()`
  para `dCondIniP`/`dCondIniU`.
- `app/pipeline/results.py::save_results` — escritura de los CSV de
  dispatch/price/bess.
- `app/cli.py::_available_dates` — `storage.list_dir` en vez de
  `Path.glob`.
- `app/pipeline/scenarios.py` (nuevo, ver seccion 1) — lectura de YAML de
  escenarios.

Este paso es un refactor puro: mismo comportamiento, misma firma publica,
tests existentes deben seguir pasando sin modificacion de aserciones.

## 1. Escenarios BESS: biblioteca YAML reusable

`scenarios/bess/*.yaml` — un archivo = un `BessScenario`. Nombre de archivo
= nombre del escenario (ej. `scenarios/bess/20pct_arbitrage.yaml`).

`app/pipeline/scenarios.py::load_bess_scenario(name_or_path: str, storage: Storage) -> BessScenario`:
- si `scenarios/bess/{name}.yaml` existe (via `storage.exists`), lo usa;
- si no, trata el argumento como ruta literal.
- parsea con `yaml.safe_load` + `BessScenario.parse_obj(...)` (pydantic v1).

No se incluye ejemplo en modo `generator` (sin formulacion Pyomo aun; lanza
`NotImplementedError` en `model.py`).

`app/cli.py::run` gana `--bess-scenario <name-or-path>`: si se pasa, carga el
escenario y lo asigna a `bess_scenario` en cada `DispatchCase` construido
para la corrida.

## 2. Comandos CLI nuevos

- **`fetch <dates>`**: envoltura delgada sobre `ensure_data_for_date` por
  cada fecha del rango (reusa `parse_dates_arg`). No construye caso ni
  resuelve modelo.
- **`evaluate <dates> -t <level> --out <dir> --data-dir <dir>`**: re-scoring
  post-hoc sin resolver el modelo. Por cada fecha/nivel:
  1. lee `{out}/marginal_price-{date}-{type}.csv`,
  2. **ordena por `datetime`** (el CSV se escribe en el orden de iteracion de
     `model._model.dual`, no garantizado; el path inline usa
     `extract_mpo_sorted`, este debe igualarlo),
  3. compara contra `load_actual_price` (columna `ideal_marginal_price`),
  4. escribe `metrics-{date}-{type}.csv`.

  Criterio de aceptacion: correr `run --eval` para un caso y luego `evaluate`
  sobre el mismo `--out`, y que ambos `metrics-*.csv` sean iguales a
  precision de punto flotante. Una prueba que solo verifique "el CSV se
  genero" no detecta el bug de orden.

- **`compare <out-dir-a> <out-dir-b>`**: outer join de los dos
  `metrics-summary.csv` sobre `(date, type, scenario)`. Filas presentes en un
  solo lado quedan con NaN en el otro (no se descartan). Imprime/guarda el
  delta.

## 3. Persistencia de resultados BESS

`save_results` gana una rama BESS (solo si `case.bess_scenario is not
None`): por unidad x hora, columnas `charge`, `discharge`, `soc`,
`revenue = discharge * mpo`, `cost = charge * mpo`. Precio de liquidacion es
el MPO (precio marginal del sistema), no el bid propio: el bid es un insumo
de optimizacion, no el precio de mercado, y los escenarios `grid_asset`
frecuentemente no tienen bids (`charge_bid`/`discharge_bid` son `None` salvo
en `arbitrage`/`generator`). Se escribe a
`bess_results-{date}-{type}.csv`.

`RunResult` gana `bess_path: str | None = None`.

**Convencion de unidades**: energia en MWh, precio en COP/kWh (ver docstring
de `app/utils/metrics.py` y `load_precio_bolsa`). `revenue`/`cost` en COP
requieren factor `* 1000` (MWh -> kWh). Verificar el orden de magnitud de un
dia real contra un precio conocido antes de confiar en la columna — sigue
siendo el punto abierto de Fase 0 sobre unidades, y este es el primer codigo
que multiplica energia por precio.

**Limitacion documentada, no corregida en Fase 1**: `same_soc_start_and_end`
(`app/model/constraints/bess/soc.py`) solo aplica en modo `grid_asset`. El
SOC de cierre difiere por modo, asi que el revenue neto no es directamente
comparable entre `arbitrage` y `grid_asset` sin tener esto en cuenta.

## 4. Resumen consolidado

`run_many` (app/pipeline/runner.py):

- emite una fila por **cada** corrida con `r.ok`, no solo
  `r.ok and r.metrics` (si no, corridas BESS sin actuals XM, o corridas con
  `--no-eval`, desaparecen del resumen — justo el caso de escenarios de
  penetracion futura que el roadmap quiere resumir). Columnas de metricas
  quedan vacias/NaN cuando no aplican.
- agrega columna `scenario`: `bess_scenario.penetration_level` si esta
  seteado, si no `"baseline"`. Es lo que le da sentido a `compare` (sin esto,
  dos corridas que solo difieren en escenario BESS colisionan en la misma
  llave `(date, type)`).
- si `bess_scenario` esta seteado, agrega: MWh cargados totales, MWh
  descargados totales, SOC promedio, revenue neto (suma de
  `revenue - cost` del CSV BESS de esa corrida).

## Orden de construccion (cada paso enviable/probable por separado)

0. `Storage`/`LocalStorage`/`get_storage` + migrar todos los llamadores de
   la seccion 0 (refactor puro, tests existentes sin cambios de aserciones)
1. loader de escenarios YAML + `--bess-scenario` en `run`
2. CSV de resultados BESS + `RunResult.bess_path`
3. columnas de resumen + columna `scenario`, fix de "toda fila `ok`"
4. `fetch`
5. `evaluate`
6. `compare`

## Fuera de alcance (explicito)

- Reescribir `_dispatch_type` string / matching por substring en
  `model.py`/`soc.py` — se deja igual.
- Implementar `GcsStorage` real (credenciales, bucket, dependencia
  `google-cloud-storage`) — futuro, cuando haya caso de uso.
- Formulacion Pyomo del modo BESS `generator`.
- Validacion end-to-end de `case_builder` contra datos reales (Fase 0,
  brecha conocida separada).
