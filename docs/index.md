# gridforge

Repositorio académico para aproximar el despacho eléctrico colombiano, comparar resultados contra referencias de XM y estudiar el efecto de incorporar BESS.

## Estado real del repositorio

- Modelo Pyomo con formulación de unit commitment y restricciones térmicas.
- CLI Typer para `run`, `fetch`, `evaluate` y `compare`.
- Pipeline para construir casos, resolver, guardar resultados y comparar contra datos reales.
- Escenarios declarativos BESS en `scenarios/bess/`.
- Frontend, API y worker en `frontend/`, `services/api/` y `services/worker/`.

## Documentación disponible

- [Formulación matemática del modelo](formulacion-matematica.html)
- [README principal del repositorio](../README.md)
- [Hoja de ruta de la aplicación](roadmap-aplicacion-despacho.md)

## Cómo navegar el proyecto

- `app/model/`: modelo, variables y restricciones.
- `app/pipeline/`: construcción del caso, ejecución y evaluación.
- `app/data/`: carga, descarga y parsing de insumos XM.
- `tests/`: suite pytest para validación del pipeline y del CLI.
