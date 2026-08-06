---
title: "Formulación matemática"
layout: default
---

# Formulación matemática del despacho eléctrico

## Tabla de contenidos

- [Introducción](#introducción)
- [Variables de decisión](#variables-de-decisión)
- [Función objetivo](#función-objetivo)  
- [Restricciones principales](#restricciones-principales)
- [Restricciones de BESS](#restricciones-de-bess)
- [Procedimiento de precio marginal](#procedimiento-de-precio-marginal)
- [Parámetros y conjuntos](#parámetros-y-conjuntos)

---

## Introducción

El modelo de despacho eléctrico es un problema de **unit commitment** (UC) con restricciones térmicas que minimiza el costo de operación del sistema. Opcionalmente integra sistemas de almacenamiento en batería (BESS) bajo diferentes modos de operación.

El modelo está formado como un **programa lineal entero mixto (MILP)**:

$$\text{minimizar } c^T x$$
$$\text{sujeto a: } Ax = b, \quad Cx \le d, \quad x_B \in \mathbb{Z}, \quad x_C \in \mathbb{R}$$

donde $x_B$ son variables binarias (on/off, arranque/apagado) y $x_C$ son variables continuas (potencias, energía).

---

## Variables de decisión

### Generadores térmicos

**$p_{g,t} \in [0, \infty)$** — Potencia activa del generador $g$ en período $t$ (MW)
- Variable continua no negativa
- Limitada por la restricción de capacidad cuando el generador está encendido
- Típicamente resuelta entre 0.1 y 1000+ MW

**$z_{g,t} \in \{0, 1\}$** — Variable de estado (on/off) del generador $g$ en período $t$
- Binaria: $z_{g,t} = 1$ si generador está en servicio, $z_{g,t} = 0$ si está apagado
- Decisión fundamental del unit commitment
- Acoplada a $p_{g,t}$ mediante restricción de capacidad

**$z^{up}_{g,t} \in \{0, 1\}$** — Variable de arranque (startup) del generador $g$ en período $t$
- Binaria: $z^{up}_{g,t} = 1$ cuando el generador pasa de apagado a encendido
- Captura la transición $z_{g,t-1}=0 \to z_{g,t}=1$
- Genera costo de arranque $c^{start}_g$ (combustible de ignición, pérdidas térmicas, etc.)

**$z^{down}_{g,t} \in \{0, 1\}$** — Variable de apagado (shutdown) del generador $g$ en período $t$
- Binaria: $z^{down}_{g,t} = 1$ cuando el generador pasa de encendido a apagado
- Captura la transición $z_{g,t-1}=1 \to z_{g,t}=0$
- Ocasionalmente genera costo de apagado (modelos avanzados)

### Variables de BESS

**$c_{b,t} \in [0, C^{max}_b]$** — Potencia de carga de batería $b$ en período $t$ (MW)
- Variable continua, limitada por capacidad de carga $C^{max}_b$
- Solo activa cuando $\delta^{ch}_{b,t} = 1$ (ver restricción de acoplamiento)
- Energía que fluye hacia el almacenamiento

**$d_{b,t} \in [0, D^{max}_b]$** — Potencia de descarga de batería $b$ en período $t$ (MW)
- Variable continua, limitada por capacidad de descarga $D^{max}_b$
- Solo activa cuando $\delta^{dis}_{b,t} = 1$
- Energía inyectada a la red desde el almacenamiento

**$soc_{b,t} \in [SOC^{min}_b, SOC^{max}_b]$** — Estado de carga (SOC) de batería $b$ al final del período $t$ (MWh)
- Variable continua que representa energía acumulada
- Evoluciona según ecuación de balance: $soc_{b,t} = soc_{b,t-1} + \eta^{ch}_b c_{b,t} - d_{b,t}/\eta^{dis}_b$
- Limitada por capacidad instalada y requerimientos de durabilidad

**$\delta^{ch}_{b,t} \in \{0, 1\}$** — Indicador de modo carga para batería $b$ en período $t$
- Binaria: $\delta^{ch}_{b,t} = 1$ cuando batería está en modo carga
- Acoplada a $c_{b,t}$: $c_{b,t} \le C^{max}_b \cdot \delta^{ch}_{b,t}$
- Mutuamente excluyente con descarga: $\delta^{ch}_{b,t} + \delta^{dis}_{b,t} \le 1$

**$\delta^{dis}_{b,t} \in \{0, 1\}$** — Indicador de modo descarga para batería $b$ en período $t$
- Binaria: $\delta^{dis}_{b,t} = 1$ cuando batería está en modo descarga
- Acoplada a $d_{b,t}$: $d_{b,t} \le D^{max}_b \cdot \delta^{dis}_{b,t}$
- Mutuamente excluyente con carga

---

## Función objetivo

Minimizar el costo total de operación:

$$\min \quad \sum_{g \in G} \sum_{t \in T} \left[ \beta_g \cdot p_{g,t} + c^{start}_g \cdot z^{up}_{g,t} \right] + \sum_{b \in B} \sum_{t \in T} \left[ \gamma^{ch}_{b,t} \cdot c_{b,t} - \gamma^{dis}_{b,t} \cdot d_{b,t} \right]$$

### Descomposición de costos

**1. Costo variable de generación térmica**: $\sum_g \sum_t \beta_g \cdot p_{g,t}$
- $\beta_g$ (USD/MWh): Costo incremental de generación (típicamente 20–150 USD/MWh)
- Componentes: costo de combustible + O&M variable
- Ejemplos: Carbón ≈ 30–40, Gas ≈ 60–100, Diesel ≈ 150–250

**2. Costo de arranque**: $\sum_g \sum_t c^{start}_g \cdot z^{up}_{g,t}$
- $c^{start}_g$ (USD/arranque): Costo fijo por cada startup del generador
- Rango típico: 500–5,000 USD
- Incluye: combustible de ignición, pérdidas térmicas, desgaste mecánico

**3. Costos de BESS** (modo bienestar social):
- $\gamma^{ch}_{b,t} \cdot c_{b,t}$: Costo de carga (puede ser negativo ≡ beneficio)
- $-\gamma^{dis}_{b,t} \cdot d_{b,t}$: Beneficio de descarga (valor de energía inyectada)
- En modo arbitraje: estos coeficientes parametrizan precios de mercado esperados

### Modos de operación de BESS

| Modo | Objetivo de BESS | Función de costo | Aplicación |
|------|------------------|------------------|-----------|
| `bess_ideal`, `bess_preideal` | Arbitraje independiente | Parámetros de escenario | Operador de batería independiente |
| `bess_ideal_resource`, `bess_preideal_resource` | Activo del sistema | Contribuye a bajar costo total | Batería propiedad/controlada por operador |

---

## Restricciones principales

### 1. Balance de potencia (Kirchhoff nodal)

$$\sum_{g \in G_n} p_{g,t} + \sum_{b \in B_n} (d_{b,t} - c_{b,t}) = D_{n,t} \quad \forall n \in N, t \in T$$

En modelos agregados nacionales, hay una única restricción de balance:

$$\sum_{g \in G} p_{g,t} + \sum_{b \in B} (d_{b,t} - c_{b,t}) = D_t \quad \forall t \in T$$

**Interpretación**: En cada nodo y período, generación + descarga neta debe igualar demanda.

**Tipo de restricción**: Igualdad (=). Es la restricción de balance de energía fundamental.

**Significado económico**: El precio dual de esta restricción es el **precio marginal** (LMP - Locational Marginal Price) del sistema.

### 2. Límites de capacidad de generadores

$$P^{min}_{g,t} \cdot z_{g,t} \le p_{g,t} \le P^{max}_{g,t} \cdot z_{g,t} \quad \forall g \in G, t \in T$$

**Interpretación**:
- Si $z_{g,t} = 0$: $p_{g,t} = 0$ (generador forzado a cero)
- Si $z_{g,t} = 1$: $P^{min}_{g,t} \le p_{g,t} \le P^{max}_{g,t}$ (rango operativo)

**Parámetros típicos**:
- $P^{max}_{g,t}$ = Capacidad nominal del generador (100–800 MW típicamente)
- $P^{min}_{g,t}$ = Mínimo técnico (10–30% de capacidad)
- En algunos casos, incluyen disponibilidad: $P^{max}_{g,t} = \text{Cap}_g \times \text{Disponibilidad}_{g,t}$

### 3. Rompimientos de rampa

$$p_{g,t} - p_{g,t-1} \le RU_g \quad \forall g, t \ge 2$$
$$p_{g,t-1} - p_{g,t} \le RD_g \quad \forall g, t \ge 2$$

**Interpretación**: Los generadores térmicos no pueden cambiar potencia instantáneamente.
- $RU_g$ (ramp-up): Máxima tasa de aumento (MW/hora)
- $RD_g$ (ramp-down): Máxima tasa de disminución (MW/hora)

**Justificación física**:
- Limitaciones de velocidad de válvulas de entrada de vapor
- Inercia térmica de calderas
- Dinámica de combustión en turbinas

**Valores típicos**: $RU_g = RD_g \approx 20–50\%$ de $P^{max}_g$ por hora

### 4. Tiempo mínimo en línea (Minimum Up Time - UT)

$$\sum_{t'=t-UT_g+1}^{t} z^{up}_{g,t'} \le z_{g,t} \quad \forall g, t \ge UT_g$$

**Interpretación**: Una vez encendido, el generador debe permanecer activo $\ge UT_g$ períodos consecutivos.

**Justificación**:
- Desgaste mecánico por arranques frecuentes
- Estabilidad térmica de calderas
- Economía: No es rentable prender y apagar rápidamente

**Rango típico**: $UT_g \in \{1, 2, 4, 8\}$ horas

### 5. Tiempo mínimo fuera de línea (Minimum Down Time - DT)

$$\sum_{t'=t-DT_g+1}^{t} z^{down}_{g,t'} \le 1 - z_{g,t} \quad \forall g, t \ge DT_g$$

**Interpretación**: Una vez apagado, el generador debe permanecer inactivo $\ge DT_g$ períodos.

**Justificación**:
- Enfriamiento de tuberías y equipos
- Presurización de sistemas antes de reencendido
- Seguridad y durabilidad

**Rango típico**: $DT_g \in \{1, 2, 4\}$ horas

### 6. Lógica de arranque/apagado

$$z^{up}_{g,t} - z^{down}_{g,t} = z_{g,t} - z_{g,t-1} \quad \forall g, t \ge 2$$

**Tabla de estados válidos**:

| $z_{g,t-1}$ | $z_{g,t}$ | $z^{up}_{g,t}$ | $z^{down}_{g,t}$ | Descripción |
|-------------|----------|----------------|-----------------|-------------|
| 0 | 0 | 0 | 0 | Apagado → Apagado |
| 0 | 1 | 1 | 0 | **Arranque** |
| 1 | 0 | 0 | 1 | **Apagado** |
| 1 | 1 | 0 | 0 | Encendido → Encendido |

Esta restricción fuerza coherencia entre las variables de transición y estado.

---

## Restricciones de BESS

### 1. Ecuación de balance de energía (SOC dynamics)

$$soc_{b,t} = soc_{b,t-1} + \eta^{ch}_b \cdot c_{b,t} - \frac{d_{b,t}}{\eta^{dis}_b} \quad \forall b \in B, t \in T$$

**Interpretación**: Estado de carga en período $t$ es estado anterior más energía cargada menos energía descargada:
- Carga: $\eta^{ch}_b \cdot c_{b,t}$ con eficiencia de carga $\eta^{ch}_b$
  - Si $c_{b,t} = 10$ MW y $\eta^{ch}_b = 0.9$, se almacenan 9 MWh
- Descarga: $\frac{d_{b,t}}{\eta^{dis}_b}$ (requiere extraer más energía del banco para entregar $d_{b,t}$)
  - Si queremos descargar 10 MW y $\eta^{dis}_b = 0.9$, sacamos 11.1 MWh del banco

**Condición inicial**: $soc_{b,0}$ es un parámetro (usualmente el SOC histórico del sistema)

**Ejemplo numérico**:
```
Período t-1: soc = 200 MWh
Período t: c = 20 MW, η_ch = 0.9, η_dis = 0.85
soc_t = 200 + 0.9*20 - 0/0.85 = 200 + 18 = 218 MWh
```

### 2. Límites de estado de carga

$$SOC^{min}_b \le soc_{b,t} \le SOC^{max}_b \quad \forall b, t$$

**Parámetros**:
- $SOC^{min}_b$ (MWh): Mínimo profundidad de descarga permitida
  - Típicamente 20–30% de capacidad (para preservar durabilidad)
- $SOC^{max}_b$ (MWh): Máximo estado de carga
  - Típicamente 95–100% de capacidad (considerando ineficiencias de carga)
- $E^{cap}_b$ (MWh): Capacidad nominal de almacenamiento
  - Relación: $SOC^{max}_b \approx 0.95 \cdot E^{cap}_b$

**Justificación técnica**: Profundidad de descarga (DoD) es parámetro de diseño crítico para:
- Vida útil de celdas (típicamente 10–20 años)
- Rendimiento de carga/descarga
- Eficiencia general del sistema

### 3. Acoplamiento de potencia de carga

$$c_{b,t} \le C^{max}_b \cdot \delta^{ch}_{b,t} \quad \forall b, t$$

**Interpretación**: La batería solo puede cargar si su modo de carga está activado.
- Si $\delta^{ch}_{b,t} = 0$: $c_{b,t} \le 0 \cdot C^{max}_b = 0$ → fuerza $c_{b,t} = 0$
- Si $\delta^{ch}_{b,t} = 1$: $c_{b,t} \le C^{max}_b$ → permite cualquier valor hasta el máximo

**Parámetro**: $C^{max}_b$ (MW) = Potencia máxima de carga permitida

### 4. Acoplamiento de potencia de descarga

$$d_{b,t} \le D^{max}_b \cdot \delta^{dis}_{b,t} \quad \forall b, t$$

**Interpretación**: Análoga a carga, pero para descarga.
- Si $\delta^{dis}_{b,t} = 0$: fuerza $d_{b,t} = 0$
- Si $\delta^{dis}_{b,t} = 1$: permite $d_{b,t} \in [0, D^{max}_b]$

**Parámetro**: $D^{max}_b$ (MW) = Potencia máxima de descarga

### 5. Mutualidad carga/descarga

$$\delta^{ch}_{b,t} + \delta^{dis}_{b,t} \le 1 \quad \forall b, t$$

**Interpretación**: La batería no puede cargar Y descargar simultáneamente.

**Estados permitidos**:
- $\delta^{ch}_{b,t} = 0, \delta^{dis}_{b,t} = 0$: Batería en reposo/idle (sin operación)
- $\delta^{ch}_{b,t} = 1, \delta^{dis}_{b,t} = 0$: Cargando
- $\delta^{ch}_{b,t} = 0, \delta^{dis}_{b,t} = 1$: Descargando
- $\delta^{ch}_{b,t} = 1, \delta^{dis}_{b,t} = 1$: **NO PERMITIDO** (violaría restricción)

**Nota**: Esto simplifica la realidad; hay pérdidas de standby (~1-2% diarios) que se ignoran en este modelo.

---

## Procedimiento de precio marginal

### Razón: Lectura de duales en MILP

En modelos MILP puro, los precios duales de restricciones pueden no estar disponibles o no tener interpretación económica clara debido a:
- Presencia de variables binarias (discretas)
- Soluciones en vértices potencialmente degenerados
- Falta de garantía de unicidad de duales

### Procedimiento de dos fases

**Fase 1: Resolución MILP completa**

Se resuelve el modelo completo (todas las variables binarias libres):

$$\text{minimizar } c^T x$$
$$\text{s.a. } Ax = b, \quad Cx \le d, \quad x_B \in \mathbb{Z}, \quad x_C \in \mathbb{R}$$

**Resultado**: Solución óptima $x^* = (x^*_B, x^*_C)$ con valor objetivo $v^*$

**Salidas**:
- Decisiones binarias: $z^*_{g,t}, z^{up*}_{g,t}, z^{down*}_{g,t}, \delta^{ch*}_{b,t}, \delta^{dis*}_{b,t}$
- Potencias: $p^*_{g,t}, c^*_{b,t}, d^*_{b,t}, soc^*_{b,t}$
- Despacho y plan de arranques/apagados

**Fase 2: Resolución LP con binarias fijadas**

Se resuelve nuevamente el modelo como LP puro con todas las variables binarias fijadas a sus valores óptimos de Fase 1:

$$\text{minimizar } c^T x_C$$
$$\text{s.a. } A_C x_C = b - A_B x^*_B, \quad C_C x_C \le d - C_B x^*_B, \quad x_C \in \mathbb{R}$$

**Resultado**: Solución LP $\bar{x}^*_C$ (típicamente igual a $x^*_C$ de Fase 1)

**Salidas cruciales**: Precios duales $\lambda^*_t$ de la restricción de balance de potencia

### Precio marginal

El **precio marginal de la electricidad (MCE o LMP)** en período $t$ se define como:

$$\lambda_t = -\text{dual de restricción de balance en período } t$$

**Interpretación económica**:
- $\lambda_t$ (USD/MWh): Costo incremental de satisfacer 1 MW adicional de demanda
- Si $\lambda_t = 75$: aumentar demanda en 1 MW cuesta 75 USD
- **Este es el precio que se utiliza en subastas y valoración de servicios**

**Ejemplo de lectura**:
- Si dual = -75, entonces $\lambda_t = 75$ USD/MWh
- Si dual = +50, entonces $\lambda_t = -50$ USD/MWh (muy poco común, indicaría error)

### Ventajas de esta aproximación

| Ventaja | Justificación |
|---------|---------------|
| **Robustez** | Precios duales siempre disponibles en solucionadores LP |
| **Economía** | Interpretación clara: costo marginal de producción |
| **Consistencia** | Coincide con metodología de XM (operador colombiano real) |
| **Convergencia** | Valores duales estables si problema LP es no-degenerado |

---

## Parámetros y conjuntos

### Conjuntos

| Símbolo | Descripción | Tamaño típico | Ejemplo |
|---------|-------------|---------------|---------|
| $G$ | Generadores térmicos | 10–100 | {BETANIA, EL QUIMBO, CARTAGO, ...} |
| $B$ | Baterías BESS | 0–5 | {BESS_BOGOTA, BESS_COSTA} |
| $N$ | Nodos | 1–500 | {norte, centro, sur} (modelo agregado: 1 nodo) |
| $T$ | Períodos (horas) | 24–168 | {1, 2, ..., 24} |

### Parámetros de generadores (caso genérico)

| Símbolo | Descripción | Unidad | Rango típico | Notas |
|---------|-------------|--------|--------------|-------|
| $\beta_g$ | Costo variable | USD/MWh | 20–250 | Combustible + O&M |
| $c^{start}_g$ | Costo de arranque | USD | 500–5000 | Apagado → Encendido |
| $P^{max}_g$ | Capacidad nominal | MW | 5–800 | Máxima potencia |
| $P^{min}_g$ | Potencia mínima técnica | MW | 0.1·$P^{max}$ – 0.3·$P^{max}$ | Estabilidad de generador |
| $RU_g$ | Ramp-up máximo | MW/h | 0.2·$P^{max}$ – 0.5·$P^{max}$ | Velocidad aumento |
| $RD_g$ | Ramp-down máximo | MW/h | 0.2·$P^{max}$ – 0.5·$P^{max}$ | Velocidad disminución |
| $UT_g$ | Tiempo mín en línea | h | 1–8 | Después de arrancar |
| $DT_g$ | Tiempo mín fuera | h | 1–4 | Después de apagar |
| $\text{Avail}_{g,t}$ | Disponibilidad | [0,1] | 0.95–1.0 | % capacidad disponible |

### Parámetros de BESS

| Símbolo | Descripción | Unidad | Rango típico | Notas |
|---------|-------------|--------|--------------|-------|
| $C^{max}_b$ | Potencia carga máxima | MW | 10–100 | MW en modo carga |
| $D^{max}_b$ | Potencia descarga máxima | MW | 10–100 | MW en modo descarga |
| $E^{cap}_b$ | Capacidad energía | MWh | 40–400 | Total almacenado |
| $\eta^{ch}_b$ | Eficiencia carga | [0,1] | 0.85–0.95 | Entrada → Almacenamiento |
| $\eta^{dis}_b$ | Eficiencia descarga | [0,1] | 0.85–0.95 | Almacenamiento → Salida |
| $SOC^{min}_b$ | Mínimo SOC | MWh | 0.2·$E^{cap}$ – 0.3·$E^{cap}$ | Durabilidad |
| $SOC^{max}_b$ | Máximo SOC | MWh | 0.95·$E^{cap}$ | Ineficiencia carga |
| $SOC^{init}_b$ | SOC inicial | MWh | Histórico | Condición en $t=0$ |

### Datos de entrada temporal

| Símbolo | Descripción | Fuente | Frecuencia |
|---------|-------------|--------|-----------|
| $D_t$ | Demanda del sistema | XM histórico | Horaria |
| $P^{avail}_{g,t}$ | Disponibilidad gen. | XM OFEI | Horaria |
| $O_{g,t}$ | Oferta de precio | Escenarios | Variable |
| $\text{Wind}_{t}$, $\text{Solar}_{t}$ | Generación renovable | Histórico/pronóstico | Horaria |

---

## Notas técnicas y referencias

### Formación MILP

- Problema estándar en literatura de optimización de potencia
- Complejidad NP-Hard; resolución heurística para problemas grandes
- Horizonte típico: 24–168 horas; resolución: horaria

### Solucionador

El código implementa el modelo en **Pyomo** (Python Optimization Modeling Objects) con solucionadores:
- **GLPK**: Gratuito, bueno para problemas medianos
- **CBC**: Gratuito, mejor performance que GLPK
- **CPLEX/Gurobi**: Comerciales, excelente para grandes instancias

### Validación

- Resultados comparados contra predespacho oficial XM (ABANICO)
- Métrica: RMSE de precios marginales, diferencia en costo total
- Data pública de XM para 2019–2024

### Variantes futuras

- **AC-OPF**: Flujos de potencia óptimos (considera pérdidas, ángulos de voltaje)
- **Estocástico**: Demanda e inyecciones renovables inciertas
- **Security-constrained**: Criterios N-1 de contingencia
- **Multi-período**: Coordinación con mercados forward/futuros
