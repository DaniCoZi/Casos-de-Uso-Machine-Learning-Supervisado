# Regresión Lineal — Actividad explicativa

**Autores:** Luis Fernando González Guevara · Kevin David Peña Ávila · Edwin Daniel Rodríguez Cruz  
**Universidad de Cundinamarca, Extensión Chía — Facultad de Ingeniería**  
**Asignatura:** Machine Learning  
**Lugar y fecha:** Chía, Cundinamarca, Colombia — 12 de septiembre de 2025

---

## ¿De qué trata esta actividad?
Esta actividad explica, de manera sencilla, qué es la **regresión lineal** y por qué es útil. La idea central es que, cuando tenemos dos variables relacionadas (por ejemplo, *precio* y *ingresos*), podemos **trazar una línea** que resuma esa relación. Esa línea nos ayuda a **entender la tendencia** general y a **hacer predicciones** aproximadas.

---

## ¿Para qué sirve la regresión lineal?
- **Entender relaciones**: ver si, en promedio, cuando una variable sube la otra también sube (o baja).
- **Predecir**: estimar un valor aproximado de una variable a partir de otra.
- **Apoyar decisiones**: por ejemplo, cómo podría cambiar el ingreso si aumentamos la publicidad.

> Importante: **explicar** no es lo mismo que **probar causa**. La regresión muestra tendencias; para afirmar causalidad se necesita evidencia adicional (diseño del estudio, controles, etc.).

---

## Ideas clave
- **La línea de regresión**: es una línea que pasa “por el centro” de la nube de puntos. Resume la relación general.
- **Pendiente de la línea**: nos dice si la relación es positiva (sube) o negativa (baja) y con qué fuerza.
- **Variabilidad natural**: los puntos no caen todos sobre la línea. Hay factores no observados, errores de medición y ruido.
- **Predicción con cuidado**: la línea predice bien cerca de los datos observados; lejos de ellos (extrapolar) es arriesgado.

---

## Supuestos en palabras simples
Para que las conclusiones sean confiables, la explicación se apoya en estas ideas básicas:
1. **Relación más o menos recta**: que la tendencia general se parezca a una línea.
2. **Dispersión parecida**: los errores (distancias a la línea) no deberían crecer o encogerse mucho a medida que cambia la variable.
3. **Errores independientes**: un error en una observación no debería depender del error en otra (especialmente en series de tiempo).
4. **Errores “normales”**: cuando miramos todas las diferencias, deberían tener una forma más o menos simétrica (tipo campana).

> Si alguna no se cumple, podemos **contarlo y ajustar expectativas**: quizá la línea explica menos, o conviene transformar datos o probar otro enfoque.

---

## Ejemplo didáctico de la actividad
**Relación entre Ingresos, Publicidad y Precio**  
- Vimos que **más publicidad** normalmente se asocia con **más ingresos** (tendencia positiva).  
- También notamos que **precios muy altos** pueden relacionarse con **menos ingresos** si caen las ventas (tendencia negativa).  
- La línea nos permite **resumir** estas tendencias y **estimar** qué podría pasar si cambiamos publicidad o precio.

---

## ¿Cómo leer los gráficos que acompañan la actividad?
- **Nube de puntos**: cada punto es una observación real. Miramos si forman una **tendencia clara**.
- **La línea**: pasa por el centro de esa nube. Si sube, la relación es **positiva**; si baja, es **negativa**.
- **La dispersión**: si los puntos se alejan mucho de la línea, la **predicción es más incierta**.

---

## Lo que se aprende
- Explicar con palabras la relación entre variables usando una línea.
- Diferenciar **explicación** de **predicción** y de **causalidad**.
- Leer e interpretar gráficos sencillos para comunicar hallazgos.

---

## Recomendaciones para usar la herramienta con criterio
- **No extrapolar**: evitar predecir muy fuera del rango observado.
- **Contextualizar**: considerar otros factores (competencia, estacionalidad, calidad).
- **Comunicar incertidumbre**: reconocer que las predicciones son aproximadas.

---

## Referencias
- Eal, R. L. — *Regresión lineal* (revisión general).  
- Nabi, I. — *Supuestos del modelo clásico de regresión lineal* (explicación de las ideas base).  
- RPubs — *Regresión lineal simple: Supuestos* (visualizaciones).  
- GraphPad — *Linear regression calculator* (herramienta de práctica).  
- GeoGebra — Material didáctico de regresión lineal.

---
