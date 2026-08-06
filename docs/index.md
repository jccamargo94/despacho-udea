---
title: "Despacho UDEA"
layout: default
---

# Despacho Eléctrico — Documentación Técnica

## Bienvenida

Este sitio documenta un modelo académico de **unit commitment** (despacho eléctrico) para el sistema colombiano, desarrollado en la Universidad de Antioquia (UDEA). El proyecto aproxima la operación económica del sistema de potencia incluyendo generadores térmicos y sistemas de almacenamiento en batería (BESS).

**Propósito**: Entender la operación económica diaria del sistema eléctrico, validar decisiones de inversión en almacenamiento y estudiar diferentes modos de operación de BESS.

---

## Contenidos principales

<div class="toc">
  <h3>Documentación disponible en este sitio</h3>
  <ul>
    <li><a href="formulacion-matematica.html"><strong>Formulación matemática</strong></a> — Descripción completa del modelo MILP, variables, restricciones y procedimiento de precio marginal</li>
    <li><a href="roadmap-aplicacion-despacho.md"><strong>Hoja de ruta</strong></a> — Fases de implementación, objetivos y estado actual</li>
    <li><a href="../README.md"><strong>README principal</strong></a> — Instalación, requisitos, datos y estructura del repositorio</li>
  </ul>
</div>

---

## ¿Qué es este modelo?

El **modelo de despacho económico** resuelve la siguiente pregunta:

> _¿Cuál es la combinación de plantas generadoras que minimiza el costo de operación del sistema eléctrico en 24 horas, satisfaciendo demanda y restricciones físicas?_

### Características principales

**Generadores térmicos**
- Decisión binaria: encendido/apagado por cada período horario
- Costos de generación variable y costos de arranque
- Restricciones de rampa, tiempo mínimo en línea, disponibilidad

**Sistemas de almacenamiento en batería (BESS)**
- Modelo dinámico de carga/descarga con eficiencia
- Estado de carga (SOC) con límites de profundidad
- Modos de operación: arbitraje independiente o controlado por operador

**Externalidades económicas**
- Precio marginal de la electricidad (LMP) por período
- Valoración de servicios auxiliares desde duales de restricciones

### Tecnología

| Componente | Tecnología | Uso |
|-----------|-----------|-----|
| **Modelo** | Pyomo | Formulación y resolución MILP |
| **Solucionador** | GLPK, CBC, CPLEX | Optimization solvers |
| **Base de datos** | PostgreSQL | Almacenamiento de resultados y historial |
| **CLI** | Typer | Interfaz de línea de comandos |
| **Frontend** | Next.js + React | Visualización de resultados |
| **Backend API** | FastAPI | Servicio de API REST |

---

## Estructura del repositorio

```
despacho-udea/
├── app/                      # Código principal
│   ├── model/               # Modelo Pyomo y restricciones
│   ├── pipeline/            # Case builder, solver runner, evaluator
│   ├── data/                # Loaders para datos XM
│   └── db/                  # ORM, modelos, queries
├── tests/                   # Suite pytest
├── services/                # Microservicios
│   ├── api/                # FastAPI backend
│   └── worker/             # Worker de polling
├── frontend/               # Next.js app
├── docs/                   # Documentación GitHub Pages (este sitio)
└── scenarios/              # Definiciones YAML de escenarios BESS
```

**Inicio rápido:**
- Modelo: Ver [app/model/model.py](../app/model/model.py)
- Estructura de datos: Ver [app/schemas/](../app/schemas/)
- Tests: Ver [tests/](../tests/)
- CLI: Ver [app/cli.py](../app/cli.py)

---

## Flujo de operación

```
1. Descarga de datos XM
   ↓
2. Construcción del caso (demanda, oferta, disponibilidad)
   ↓
3. Resolución MILP (Fases 1 y 2)
   ↓
4. Cálculo de precio marginal
   ↓
5. Almacenamiento en BD y visualización
```

### Ejemplo de uso (CLI)

```bash
# Descargar datos de XM para una fecha específica
uv run app run --date 2024-01-15

# Comparar contra resultados reales de XM
uv run app compare --date 2024-01-15

# Evaluar costo total y duales de restricciones
uv run app evaluate --scenario bess_ideal
```

---

## Modos de operación de BESS

El modelo soporta múltiples **modos de operación** para sistemas de almacenamiento:

| Modo | Objetivo | Controlador | Aplicación |
|------|----------|-------------|-----------|
| **bess_ideal** | Arbitraje de precios | Operador de batería independiente | Estudio de viabilidad económica |
| **bess_preideal** | Pre-despacho (anticipativo) | Batería con pronóstico de demanda | Maximizar ingresos con información imperfecta |
| **bess_ideal_resource** | Minimizar costo sistema | Operador del sistema | Valuación como recurso de sistema |
| **bess_preideal_resource** | Pre-despacho del sistema | Operador controlador | Operación real anticipada |

---

## Validación y referencias

El modelo se valida comparándolo contra:

- **Predespacho oficial**: Resultados de ABANICO (XM)
- **Métricas de desempeño**: RMSE de precios marginales
- **Datos históricos**: XM 2019-2024

### Brechas conocidas

- No incluye flujos de potencia AC (no representa pérdidas de transmisión)
- Demanda agregada nacional (sin desagregación nodal)
- Horizonte de 24 horas (no incluye mercados forward/futuros)
- Renovables como data histórica (sin pronóstico estocástico)

### Mejoras futuras (Hoja de ruta)

1. **AC-OPF**: Incorporar flujos AC y pérdidas en líneas
2. **Estocástico**: Incertidumbre en demanda y renovables
3. **Security-Constrained**: Criterios N-1 de contingencia
4. **Multi-período**: Coordinación con mercados a plazo

---

## Comenzar

### Requisitos

- Python 3.12+
- `uv` (package manager)
- PostgreSQL (para persistencia)
- Solucionador: GLPK (gratuito) o CBC/CPLEX

### Instalación

```bash
# Clonar y activar entorno
git clone https://github.com/jccamargo94/despacho-udea.git
cd despacho-udea
uv sync

# Ejecutar tests
uv run pytest

# Ver CLI disponible
uv run app --help
```

### Documentación técnica completa

Para detalles matemáticos completos, ver:

**→ [Formulación Matemática](formulacion-matematica.html)**

Esta página contiene:
- Descripción formal de todas las variables de decisión (generadores, BESS)
- Función objetivo con descomposición de costos
- Restricciones principales (balance, capacidad, rampa, etc.)
- Formulación específica de BESS
- Procedimiento de cálculo de precio marginal
- Tablas de parámetros típicos y rangos operacionales

---

## Navegación

| Página | Descripción | Link |
|--------|-----------|------|
| **Inicio** (esta página) | Bienvenida, visión general, estructura del proyecto | — |
| **Formulación matemática** | Modelo MILP completo, variables, restricciones | [Ver →](formulacion-matematica.html) |
| **Hoja de ruta** | Fases, timeline, objetivos del proyecto | [Ver →](roadmap-aplicacion-despacho.md) |
| **README principal** | Instalación, requisitos, datos requeridos | [Ver →](../README.md) |

---

## Recursos y referencias

- **XM (Operador System Colombia)**: [www.xm.com.co](https://www.xm.com.co) — Datos históricos, metodología
- **Pyomo Documentation**: [pyomo.readthedocs.io](https://pyomo.readthedocs.io)
- **Unit Commitment Survey**: Wood, Wollenberg & Sheblé (1996) — Clásico en operación de sistemas
- **BESS Modeling**: Fu et al. (2021) — Almacenamiento en sistemas eléctricos

---

## Contacto y contribuciones

- **Repositorio**: [github.com/jccamargo94/despacho-udea](https://github.com/jccamargo94/despacho-udea)
- **Issues y PRs**: Bienvenidos siguiendo convenciones del proyecto

---

**Última actualización**: 2026-08 | **Tema**: jekyll-theme-primer | **Ecuaciones**: MathJax v3
