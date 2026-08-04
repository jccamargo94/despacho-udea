# Diseño: interfaces de dominio (DispatchCase, BessScenario, InputPack, RunResult)

## Contexto

`docs/roadmap-aplicacion-despacho.md` identifica cuatro interfaces de dominio
que deben quedar estables antes de construir API, worker y frontend
(Fase 3-4). Hoy esas piezas existen implícitas y dispersas:

- `DispatchConfig` (`app/model/model.py:54`): solo envuelve
  `dispatch_type: DispatchOptions`, un string enum que mezcla nivel
  (`preideal`/`ideal`) y modo BESS (`bess_ideal_resource`, etc.) en un mismo
  token.
- BESS: un `dict` sin tipar pasado como `bess=` a `build_case`
  (`app/pipeline/case_builder.py:28-495`), con claves por unidad
  (`initial_soc`, `MWh_nom`, `charge_bid`, ...). El modo de participación
  ("generador") mencionado en el roadmap no existe en el modelo Pyomo.
- Insumos: `build_case` lee directo de `data_dir` vía `app/data/loaders.py`
  sin distinguir históricos reales, descarga en vivo, o (a futuro) supuestos
  de pronóstico.
- Resultado: `CaseResult` (`app/pipeline/runner.py:19-26`) y el `dict` que
  devuelve `save_results` (`app/pipeline/results.py:33-51`) son dos formas
  distintas de describir el resultado de una corrida.

Este documento congela las 4 interfaces como modelos **pydantic**
(serializables a JSON/YAML, sin dependencia de Typer/CLI) para que Fase 1
(CLI + escenarios YAML) y Fase 3 (API + persistencia) reutilicen el mismo
contrato sin reescritura.

## Alcance

Incluye:
- Definir los 4 modelos y su validación.
- Adaptar `case_builder.py`, `runner.py`, `results.py`, `model.py`, `cli.py` y
  los 3 tests existentes (`test_cli.py`, `test_runner.py`, `test_results.py`)
  a las nuevas firmas.
- Formalizar los 3 modos BESS (`arbitrage`, `grid_asset`, `generator`) como
  campo explícito de `BessScenario`, reemplazando la convención de nombre en
  `dispatch_type`.

No incluye:
- Implementación Pyomo del modo `generator` (queda como
  `NotImplementedError` explícito en `build_case`/`model.py`).
- Loader real de `InputSource.forecast` (Fase 5). Solo se deja el campo.
- Cambios en `run_dispatch.py` / `get_date_results.py` más allá de seguir la
  firma nueva de las funciones que ya delegan a `app.pipeline`.

Reemplazo in-place, sin shim de compatibilidad: no hay consumidores externos
de `DispatchConfig`/`CaseResult` (proyecto académico, sin API publicada
todavía). Blast radius verificado por grep: `run_dispatch.py`,
`get_date_results.py`, `app/cli.py`, `app/pipeline/*.py`, `app/model/*.py`,
3 archivos de test.

## Los 4 modelos

### `DispatchCase` (`app/schemas/case.py`)

```python
class DispatchLevel(str, Enum):
    preideal = "preideal"
    ideal = "ideal"

class DispatchCase(BaseModel):
    dispatch_date: date
    level: DispatchLevel
    bess_scenario: BessScenario | None = None
    solver: str = "cbc"
    compute_prices: bool = True
```

Reemplaza `DispatchConfig`. `dispatch_type` (string tipo
`"bess_ideal_resource"`) se descompone en `level` (nivel de restricciones) +
`bess_scenario` (presencia y modo de BESS), quedando explícito en vez de
codificado en el nombre — el punto que pide el roadmap en la sección
"Participación BESS".

### `BessScenario` (`app/schemas/bess.py`)

```python
class BessMode(str, Enum):
    arbitrage = "arbitrage"     # hoy: bess_preideal / bess_ideal
    grid_asset = "grid_asset"   # hoy: bess_preideal_resource / bess_ideal_resource
    generator = "generator"     # NUEVO — sin formulación Pyomo aún

class BessUnit(BaseModel):
    name: str
    mwh_nom: float
    hours_to_deplete: float
    initial_soc: float          # fracción 0-1
    min_soc: float
    max_soc: float
    efficiency: float
    charge_bid: float | None = None
    discharge_bid: float | None = None

class BessScenario(BaseModel):
    mode: BessMode
    penetration_level: str      # etiqueta de escenario, ej. "10pct"
    units: list[BessUnit]

    @model_validator(mode="after")
    def _check_bids(self) -> "BessScenario":
        for u in self.units:
            if self.mode in (BessMode.arbitrage,) and u.charge_bid is None:
                raise ValueError(f"{u.name}: charge_bid requerido en modo arbitrage")
            if self.mode in (BessMode.arbitrage, BessMode.generator) and u.discharge_bid is None:
                raise ValueError(f"{u.name}: discharge_bid requerido en modo {self.mode}")
        return self
```

`BessUnit` mapea 1:1 a las claves que hoy arma
`case_builder.py:469-495` (`initial_soc * MWh_nom` → `bess_soc_0`,
`MWh_nom / hours_to_deplete` → `bess_max_charge`/`bess_max_discharge`, etc.)
— mismo cálculo, solo tipado y validado en el borde.

En `build_case`/`model.py`: si `bess_scenario.mode == BessMode.generator`,
lanzar `NotImplementedError("modo generator: formulación Pyomo pendiente")`
antes de tocar el solver.

### `InputPack` (`app/schemas/input_pack.py`)

```python
class InputSource(str, Enum):
    historical = "historical"
    live = "live"
    forecast = "forecast"       # Fase 5 — aceptado ya, sin loader

class InputPack(BaseModel):
    dispatch_date: date
    source: InputSource
    data_dir: str
    checksum: str | None = None
    downloaded_at: datetime | None = None
```

`build_case(case: DispatchCase, inputs: InputPack)` reemplaza la firma
actual `build_case(date, config, bess=..., ders=..., data_dir=...)`.
`inputs.source` es metadata de trazabilidad — no cambia qué loader se llama
hasta que exista Fase 5; sirve para que `RunResult` pueda registrar de dónde
salieron los insumos.

### `RunResult` (`app/schemas/run_result.py`)

```python
class RunResult(BaseModel):
    case: DispatchCase
    ok: bool
    dispatch_path: str | None = None
    price_path: str | None = None
    metrics_path: str | None = None
    metrics: dict[str, float] | None = None
    error: str | None = None
```

Reemplaza `CaseResult` y el `dict` de `save_results`. `run_case` en
`runner.py` construye un solo `RunResult` en vez de mezclar `CaseResult` +
`paths` dict + `metrics` dict.

## Cambios en módulos existentes

| archivo | cambio |
| --- | --- |
| `app/schemas/case.py` (nuevo) | `DispatchLevel`, `DispatchCase` |
| `app/schemas/bess.py` (nuevo) | `BessMode`, `BessUnit`, `BessScenario` |
| `app/schemas/input_pack.py` (nuevo) | `InputSource`, `InputPack` |
| `app/schemas/run_result.py` (nuevo) | `RunResult` |
| `app/model/model.py` | `DispatchConfig`/`DispatchOptions` → usa `DispatchCase`/`DispatchLevel`; `_add_bess_operation` lee `BessScenario.units`; guard `NotImplementedError` para `generator` |
| `app/pipeline/case_builder.py` | firma `build_case(case: DispatchCase, inputs: InputPack)`; bloque BESS (líneas 450-495) itera `case.bess_scenario.units` |
| `app/pipeline/runner.py` | `run_case`/`run_many` devuelven `RunResult`; `CaseResult` eliminado |
| `app/pipeline/results.py` | `save_results` devuelve/rellena `RunResult` en vez de `dict` |
| `app/cli.py` | arma `DispatchCase` por fecha/nivel en vez de `DispatchConfig`; sin BESS todavía desde CLI (queda para Fase 1 - YAML de escenarios) |
| `run_dispatch.py`, `get_date_results.py` | ajustan a nuevas firmas de `app.pipeline` |
| `tests/test_cli.py`, `test_runner.py`, `test_results.py` | actualizar a nuevos tipos |
| `tests/test_bess_scenario.py` (nuevo) | un caso por modo: bids faltantes → `ValidationError`; modo `generator` sin discharge_bid falla; `grid_asset` no exige bids |

## Fuera de este cambio

- CLI no gana flag de BESS todavía (Fase 1 lo agrega vía archivo YAML/JSON
  de escenario, usando `BessScenario.model_validate_json`/`model_validate`).
- No se toca `app/data/*` ni el loader real de insumos.
- No se implementa la formulación Pyomo del modo `generator`.
