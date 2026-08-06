# Formulación matemática del despacho

El modelo resuelve un unit commitment con restricciones térmicas y, cuando aplica, operación de BESS.

## Variables de decisión

- $p_{g,t} \ge 0$: generación efectiva del generador $g$ en el intervalo $t$.
- $z_{g,t} \in \{0,1\}$: indicador de si el generador $g$ está en servicio en $t$.
- $z^{up}_{g,t}, z^{down}_{g,t}$: señales de arranque y apagado.
- Para BESS: $c_{b,t}, d_{b,t}, soc_{b,t}, \delta^{ch}_{b,t}, \delta^{dis}_{b,t}$.

## Función objetivo

El objetivo es minimizar el costo de generación y el costo de arranque:

$$\min \sum_{g,t} \beta_g p_{g,t} + \sum_{g,t} c_g^{start} z^{up}_{g,t}$$

En los modos BESS orientados a bienestar social, el problema incorpora también costos o beneficios de carga/descarga del almacenamiento.

## Restricciones principales

- Balance de potencia:

$$\sum_{i} p_{i,t} + \sum_b \left(d_{b,t} - c_{b,t}\right) = D_t$$

- Rango operativo por generador:

$$P^{min}_{g,t} z_{g,t} \le p_{g,t} \le P^{max}_{g,t} z_{g,t}$$

- Rompimientos de rampa y permanencia mínima:

$$p_{g,t} - p_{g,t-1} \le RU_g, \qquad p_{g,t-1} - p_{g,t} \le RD_g$$

y las restricciones de arranque/apagado y tiempo mínimo en línea aseguran consistencia operativa.

## Formulación de BESS

Para cada batería $b$ y periodo $t$, se modela el estado de carga como:

$$soc_{b,t} = soc_{b,t-1} + \eta^{ch}_{b} c_{b,t} - \frac{d_{b,t}}{\eta^{dis}_{b}}$$

y se imponen límites de potencia y de estado de carga:

$$0 \le c_{b,t} \le C^{max}_{b} \delta^{ch}_{b,t}, \qquad 0 \le d_{b,t} \le D^{max}_{b} \delta^{dis}_{b,t}$$

$$SOC^{min}_{b} \le soc_{b,t} \le SOC^{max}_{b}$$

Además, se evita cargar y descargar simultáneamente con $\delta^{ch}_{b,t} + \delta^{dis}_{b,t} \le 1$.

## Precio marginal

El precio marginal de operación se obtiene del dual de la restricción de balance de potencia después de resolver una segunda corrida LP con las variables binarias fijadas. Esta práctica evita leer precios marginales directamente de una solución MILP.
