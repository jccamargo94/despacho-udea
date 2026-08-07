# Ingesta y storage de insumos XM — diseno

Fecha: 2026-08-06
Roadmap: `docs/roadmap-aplicacion-despacho.md`, seccion "Arquitectura objetivo"
(fila "Storage de archivos") y "Modelo de datos minimo" (`InputArtifact`).

## Contexto

Un run real fallo con `FileNotFoundError: data/dispo_declarada.csv` (ver log
del run `0024de852073456dba940c2c207fe730`). Investigacion (systematic
debugging) encontro la causa raiz: `ensure_data_for_date`
(`app/data/download.py`) solo descarga los 5 archivos de texto por-fecha
(OFEI, dCondIniU, dCondIniP, PrId, iMAR). Los 7 "CSV raiz" que
`app/data/loaders.py` lee directo (`dispo_declarada.csv`, `ofertas.csv`,
`demaCome.csv`, `agc_asignado.csv`, `parametros_plantas.csv`,
`precio_bolsa/*.csv`, `DispoCome_resource.csv`) no tienen ningun mecanismo
de descarga en el codigo — se asumen presentes en `data/` (gitignored, nunca
garantizado).

Este documento cubre unicamente el **mecanismo de obtencion/almacenamiento**
de esos insumos, para que `data/` (local) o un bucket (cloud) se comporten
como un cache: "si el dato no esta, descargarlo; si esta, usarlo", igual
para todos los insumos, no solo los 5 que ya lo hacen.

**Fuera de alcance, deliberadamente**: como se interpretan los parametros
tecnicos de Paratec (`TMG`, rampas) para el modelo de despacho por
configuraciones. Ese es un cambio de modelado (`app/model/`,
`app/pipeline/case_builder.py`), no de plomeria de datos, y necesita su
propio brainstorm — ver seccion 7 y memoria
`project_thermal-configuration-dispatch`. Este diseno cachea el JSON crudo
de Paratec sin interpretarlo.

## 1. Los insumos no son una fuente unica

Investigacion contra endpoints reales (no asumido — ver
`project_xm-data-source-matrix` en memoria) encontro tres sistemas fuente
distintos:

| dataset actual | fuente | mecanismo |
|---|---|---|
| OFEI, dCondIniU, dCondIniP, PrId, iMAR | `api-portalxm.xm.com.co` (blob por fecha) | ya existe: `PARAMS` + `ensure_data_for_date` |
| `agc_asignado.csv` | `dAGCUNIDAD{mmdd}.txt`, mismo blob endpoint, carpeta `DESPACHO` | **nuevo**: agregar `dAGCUNIDAD` a `PARAMS`; agrega AGC por-unidad, requiere agregacion unidad->recurso |
| `dispo_declarada.csv`, `ofertas.csv`, `demaCome.csv`, `precio_bolsa/*.csv`, `DispoCome_resource.csv` | API publica XM via `pydataxm.ReadDB` (`DispoDeclarada`, `PrecOferDesp`, `DemaCome`, `PrecBolsNaci`, `DispoCome`) | **nuevo**: fetch bajo demanda, ver seccion 3 |
| `parametros_plantas.csv` (y datos tecnicos que hoy no se descargan: rampas) | Paratec (`paratecbackend.xm.com.co`) | **nuevo**: cache de JSON crudo, sin interpretar (fuera de alcance la interpretacion) |

`pydataxm` ya era dependencia (`pyproject.toml`), sin usar en `app/`. Se
bumpeo `0.3.6` -> `0.3.18` esta sesion (requiere `pandas>=2.2.3`; se bumpeo
`pandas` `2.2.2` -> `2.2.3` en consecuencia). `uv.lock` resuelto, 140/140
tests verdes.

No hay metrica de `pydataxm` para AGC por-recurso (solo existe
`RespComerAGC`, agregado de sistema) — por eso AGC usa el mecanismo de
blob por-fecha en vez de `pydataxm`, igual que OFEI.

## 2. Manifest en Postgres, datos pesados en Storage

Precedente ya decidido en `2026-08-05-fase3-api-persistencia-design.md`:
*"DB solo para lo estructurado/consultable, CSV via Storage"* para series
pesadas — motivado por riesgo conocido de Supabase free-tier (pausa tras
inactividad, egress restringido). Este diseno reusa esa decision en vez de
revertirla: no se propone InfluxDB ni cargar series completas a Postgres.

Tabla nueva, unica, en el mismo Postgres de Supabase que ya tiene
`runs`/`cases`/`scenarios`:

```
input_datasets
  id            uuid pk
  dataset       text        -- 'dispo_declarada' | 'ofertas' | 'demaCome' |
                             -- 'precio_bolsa' | 'dispo_come' | 'agc_asignado' |
                             -- 'parametros_plantas_raw' (uno por fuente Paratec)
  partition_key text        -- año ('2024') para series bulk; fecha ('2026-08-01')
                             -- para blobs por-fecha; 'latest' para Paratec
  source        text        -- 'pydataxm:DispoDeclarada' | 'xm_blob:OFEI' | 'paratec:ThermalPlant.getAll'
  checksum      text
  row_count     int
  fetched_at    timestamptz
  unique(dataset, partition_key)
```

Los datos en si (CSV, txt, JSON) viven en `Storage` (protocolo existente,
sin cambios — `app/storage/base.py`), rooted en `data_dir` (local hoy,
`gs://...` el dia que `GcsStorage` exista — sigue sin implementarse, sigue
fuera de alcance, ver `2026-08-04-fase1-cli-completo-design.md` seccion 0).

## 3. Layout de particiones (generaliza convencion existente)

`precio_bolsa/precio_bolsa_2024.csv` ya particiona por año. Se generaliza
esa misma convencion a los 4 datasets bulk restantes:

```
data/
  dispo_declarada/dispo_declarada_2024.csv
  dispo_declarada/dispo_declarada_2025.csv
  ofertas/ofertas_2024.csv
  demaCome/demaCome_2024.csv
  precio_bolsa/precio_bolsa_2024.csv          (ya existe con este layout)
  dispo_come/dispo_come_2024.csv
  {date}/dAGCUNIDAD{mmdd}.txt                  (junto a OFEI etc., ya existente)
  paratec/thermal_plants_2026-08-06.json       (snapshot fechado, no particionado por año)
```

Los loaders de `app/data/loaders.py` cambian de "un CSV con toda la
historia" a "leer y concatenar las particiones-año que cubren el rango de
fechas pedido" — mismo patron que ya usa `resolve_input` para elegir entre
layouts candidatos.

## 4. Flujo unificado check-then-fetch

```
build_case(case, inputs)
  -> para cada dataset requerido por (case.dispatch_date, case.level):
       ensure_dataset(dataset, date_or_year, data_dir)
         1. query input_datasets manifest (Postgres) por (dataset, partition_key)
         2. si existe fila Y storage.exists(path esperado) -> usar tal cual
         3. si no:
              - blob por-fecha (mecanismo 1): save_file(...) [ya existe]
              - bulk pydataxm (mecanismo 2): ReadDB.request_data(...) ->
                reshape wide->long -> resolver codigo XM a resource_name ->
                escribir particion -> upsert manifest
              - Paratec (mecanismo 3): GET a endpoint -> escribir JSON crudo
                -> upsert manifest (sin interpretar contenido)
  -> loaders.load_*(data_dir) lee las particiones ya garantizadas presentes
```

`ensure_data_for_date` (mecanismo 1) ya implementa el paso 1-2 a su manera
(chequea carpeta no vacia); se extiende con `dAGCUNIDAD` sin cambiar su
forma. Los mecanismos 2 y 3 son nuevos y comparten la tabla de manifest.

## 5. Riesgos verificados, no resueltos aqui (quedan como tareas)

Verificado contra respuesta real de la API (no asumido, ver memoria):

- **`pydataxm.request_data` devuelve formato ancho** (`Values_Hour01..24`
  por fila) **indexado por un codigo corto** (`Values_code`, ej. `2QEK`),
  no por `resource_name` (`TERMO1`). Se confirmo en `PrecOferDesp` y
  `DispoDeclarada`. Se necesita (a) reshape ancho->largo y (b) un crosswalk
  codigo<->nombre antes de escribir la particion. No se investigo de donde
  sale ese crosswalk (candidato: otra coleccion de `pydataxm`, o
  `elementMRID`/`thermalPlantUnit` de Paratec) — tarea de implementacion.
- **`ReadDB()` hace red en el constructor** (`all_variables()`), y
  `request_data` esta limitado a `MaxDays=31` por llamada — backfill
  multi-año necesita loop mensual (la libreria ya lo hace internamente
  para `request_data`, no para el constructor).
- **Gaps de `Storage` nuevos, no documentados antes**: `case_builder.py:366`
  (`preideal_dispatch_map.json`) y `:388` (`ramps.json`) usan `open()`
  plano, no `storage.open()` — no funcionarian contra `gs://` el dia que
  exista `GcsStorage`. Distinto de los gaps ya documentados y aceptados en
  `2026-08-04-fase1-cli-completo-design.md` (`dCondIniP`/`dCondIniU` via
  `resolve_input`, que devuelve ruta de filesystem plano a proposito).

## 6. Testing

Mismo patron que `test_ensure_data_for_date_is_a_noop`
(`docs/superpowers/plans/2026-08-05-fase2b-fixture.md`): monkeypatch de
`pydataxm.pydataxm.ReadDB`/`requests` y de las llamadas a Paratec para que
la suite corra sin red. El fixture `tests/fixtures/xm_smoke/` necesita las
nuevas particiones (`dispo_declarada/dispo_declarada_2024.csv`, etc.) en el
nuevo layout, o el smoke test rompe.

## 7. Explicitamente diferido

Como reducir los datos por-unidad/por-configuracion de Paratec
(`minGenerationTime`, `uploadSpeed`/`downloadSpeed`, `ramps` anidado por
`configurationNumber`) a los escalares planos que hoy usa `case_builder.py`
(`TMG`, `ramp_up == ramp_down`) — y si el modelo deberia representar
unidades/configuraciones en vez de recursos planos — es un cambio de
modelado, no de esta capa de datos. Requiere su propio
`superpowers:brainstorming` sobre `app/model/` + `case_builder.py`. Ver
memoria `project_thermal-configuration-dispatch`.
