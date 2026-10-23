# Plan de trabajo — Actividad Integradora 1: EDA automatizado

**Asignatura:** Aplicaciones de Machine Learning  
**Fuente principal:** [`Integradora 1/Ejercicio`](Integradora%201/Ejercicio)  
**Referencia para los diez puntos:** [`Unidad 1/README_Practico.md`](Unidad%201/README_Practico.md) y [`Unidad 1/Practica1-EDA.ipynb`](Unidad%201/Practica1-EDA.ipynb)

## 1. Propósito y requisitos

La entrega debe diseñar y programar una herramienta propia en Python que analice un dataset de forma automatizada y produzca un diagnóstico estadístico, visual y de calidad de datos. El análisis debe estar orientado a problemas relevantes para regresión: **multicolinealidad, sesgo y valores atípicos**, e incluir los diez puntos de EDA tratados en clase.

El enunciado permite elegir el dataset. Recomienda revisar `ydata-profiling` o `Sweetviz`, crear una clase reutilizable y usar `%%writefile` para guardar el código. El plan adopta esas recomendaciones para organizar la solución; la comparación formal con ambas librerías no aparece como requisito explícito del archivo `Ejercicio`.

## 2. Entregables propuestos

1. Un dataset apropiado para un análisis orientado a regresión, con target continuo identificado.
2. Un notebook de la actividad que cargue los datos, explique decisiones y muestre resultados y conclusiones.
3. Una clase reutilizable `AutoEDA` exportada a `Integradora 1/herramientas_eda.py`.
4. Una ejecución automatizada con tablas y gráficos para los diez puntos, más diagnósticos de VIF, sesgo y outliers.
5. Una sección final que interprete los hallazgos y recomiende acciones, sin presentar las alertas estadísticas como decisiones automáticas de limpieza.

## 3. Los diez puntos que debe cubrir la herramienta

La secuencia corresponde a la Práctica 1 de la Unidad 1. El punto 1 se implementa como preparación común del análisis; los puntos 2–10 producen diagnósticos sobre el dataset.

| # | Punto de clase | Salida automatizada esperada |
|---|---|---|
| 1 | Importar librerías | Dependencias centralizadas para datos, estadísticas y gráficos; documentar cualquier dependencia adicional, por ejemplo `statsmodels` para VIF. |
| 2 | Conocer el dataset | Dimensiones, columnas, primeras/últimas filas y muestra reproducible. |
| 3 | Tipo de variable | Tipos de datos y grupos numéricos/categóricos; cardinalidad para reconocer posibles identificadores. |
| 4 | Calidad de datos | Duplicados y marcadores de ausencia o valores raros como `?`; identificar espacios sobrantes en texto. |
| 5 | Valores faltantes | Conteo y porcentaje por columna, con recomendación de revisión. |
| 6 | Outliers | Resumen IQR y gráficos boxplot; sumar Z-score como diagnóstico complementario para regresión. |
| 7 | Estadística descriptiva | Resumen numérico y categórico; media, mediana, dispersión y medidas de forma. |
| 8 | Desbalance / distribución del objetivo | Para target continuo, histograma/KDE y resumen de su distribución. `value_counts`/countplot se reserva para objetivos discretos o categóricos. |
| 9 | Análisis univariado | Distribuciones de variables, histogramas y medidas de asimetría (skewness) y curtosis. |
| 10 | Análisis bivariado/multivariado | Relaciones predictor–target, comparaciones por categoría, correlaciones y heatmap. Añadir VIF para evaluar multicolinealidad. |

## 4. Ruta de trabajo

### Fase A — Preparar el caso de análisis

1. Elegir un dataset de Kaggle, UCI o una fuente ya disponible. Preferir un tamaño y una licencia que permitan ejecutar y compartir el notebook.
2. Confirmar que existe una variable objetivo continua y definir su nombre (`target_col`). Si se elige otro tipo de target, explicar cómo se adapta el análisis al objetivo de la actividad.
3. Cargar el dataset en un notebook nuevo, por ejemplo `Integradora 1/ActividadIntegradora_EDA.ipynb`. Registrar fuente, URL, descripción de variables, unidad de medida y cualquier transformación aplicada al cargarlo.
4. Revisar forma, nombres, tipos y primeras filas. Corregir solamente problemas de lectura indispensables para que el DataFrame represente el archivo; conservar el original para el diagnóstico de calidad.

**Salida de fase:** dataset cargado, target declarado y procedencia documentada.

### Fase B — Definir el flujo y la interfaz de `AutoEDA`

5. Centralizar imports de `pandas`, `numpy`, `matplotlib` y `seaborn`; agregar la dependencia de VIF si se usa. Crear `AutoEDA(df, target_col=None)` con validación del tipo de entrada y de la existencia del target.
6. Trabajar sobre una copia del DataFrame para que el análisis no altere silenciosamente los datos del usuario. Clasificar columnas numéricas y categóricas y excluir el target de los cálculos de predictores cuando corresponda.
7. Separar el código en métodos legibles, uno por diagnóstico o grupo afín. Crear un método público `run_analysis()` que ejecute el flujo completo, guarde los resultados principales en un atributo de resultados y presente un resumen final.
8. Definir las salidas esperadas antes de programar: tablas para conteos y estadísticas; gráficos con títulos, ejes y etiquetas; alertas con umbrales declarados y posibilidad de omitir un análisis cuando no aplique.

**Salida de fase:** esqueleto de clase y contrato claro de entrada/salida.

### Fase C — Implementar los diez puntos de EDA

9. Implementar overview (forma, columnas, muestras y memoria) y tipos de variables (dtype, cardinalidad y posibles identificadores).
10. Implementar calidad de datos: duplicados, tokens como `?`, `N/A` y cadenas vacías, además de espacios sobrantes. Reportar lo encontrado; no reemplazar ni borrar datos durante un diagnóstico salvo que se documente como decisión separada.
11. Implementar faltantes con conteo y porcentaje por columna. Presentar recomendaciones de revisión; no asumir que eliminar filas o columnas sea siempre apropiado.
12. Implementar outliers numéricos con límites IQR (`1.5 × IQR`) y Z-score (`|z| > 3`) como criterios de cribado, más boxplots. Manejar columnas constantes, valores no finitos y muestras insuficientes.
13. Implementar estadísticas descriptivas para variables numéricas y categóricas. Incluir skewness y curtosis en el resumen numérico.
14. Implementar distribución del target: histograma/KDE para un target continuo; conteos y gráfico de barras solo si el target es discreto/categórico. No llamar “desbalance de clases” a la distribución de una variable continua.
15. Implementar análisis univariado: histogramas/KDE y medidas de forma numéricas; frecuencias para categóricas, limitando categorías si la cardinalidad es alta.
16. Implementar análisis bivariado/multivariado: scatterplots de predictores numéricos contra el target, boxplots del target por categorías cuando sean interpretables y heatmap de correlación numérica.

**Salida de fase:** todos los puntos pueden ejecutarse desde el método orquestador, incluso si alguna columna o gráfico no aplica.

### Fase D — Añadir el diagnóstico específico para regresión

17. Para multicolinealidad, reportar pares de predictores con correlación absoluta alta y una tabla de VIF. Documentar el umbral empleado; revisar variables constantes, faltantes y redundantes antes del cálculo. Excluir el target del VIF.
18. Para sesgo, reportar skewness y curtosis de target y predictores numéricos. Si la asimetría es marcada, recomendar evaluar transformaciones (por ejemplo, `log1p` cuando los valores lo permitan o Box–Cox cuando se cumplan sus condiciones). No modificar la columna original automáticamente.
19. Consolidar alertas con evidencia: variable, métrica, valor y criterio. Diferenciar “requiere revisión” de “debe eliminarse/transformarse”. Explicar que IQR, Z-score, correlación y VIF son señales diagnósticas dependientes del contexto.

**Salida de fase:** reporte útil para orientar un análisis de regresión, sin decisiones destructivas implícitas.

### Fase E — Guardar e integrar la herramienta

20. En el notebook, escribir la clase en una celda cuya primera línea sea `%%writefile herramientas_eda.py` y ejecutar la celda desde `Integradora 1/` para generar el módulo en esa carpeta.
21. Importar la clase desde el archivo generado y crear una instancia con el dataset y el target. La ejecución final debe requerir una llamada a `run_analysis()` y no depender de celdas manuales intermedias.
22. Mantener el notebook reproducible desde una sesión limpia: cargar datos, importar el módulo, ejecutar el análisis y mostrar conclusiones en orden.

**Salida de fase:** notebook y módulo exportado corresponden a la misma implementación.

### Fase F — Contrastar, interpretar y preparar la entrega

23. Revisar el cuaderno de referencia `Integradora 1/herramientas_EDA_(3).ipynb` y los reportes HTML existentes de ydata-profiling y Sweetviz. Usarlos para contrastar qué vistas automatizan y qué hallazgos específicos de regresión aporta la herramienta propia. Si se afirma una comparación formal, ejecutarlas sobre el mismo dataset y declarar diferencias de configuración.
24. Redactar conclusiones basadas en los resultados observados: calidad y faltantes, forma del target, variables con señales de sesgo/outliers, relaciones con el target y alertas de multicolinealidad. Proponer siguientes pasos y justificar su pertinencia.
25. Cerrar con un mapeo que indique dónde se cubre cada uno de los diez puntos y dónde aparecen multicolinealidad, sesgo y outliers.
26. Revisar que el notebook tenga fuente de datos, explicación, ejecución visible y conclusiones; confirmar que el archivo `herramientas_eda.py` esté junto a él y que las rutas funcionen desde la estructura entregada.

**Salida de fase:** materiales que muestran tanto el funcionamiento de la herramienta como la interpretación del diagnóstico.

## 5. Criterios de aceptación

- [ ] El notebook identifica dataset, fuente, target y propósito del análisis.
- [ ] `AutoEDA` procesa un DataFrame y `run_analysis()` lanza el flujo automatizado.
- [ ] Los diez puntos de la Práctica 1 están cubiertos y son localizables en el resultado.
- [ ] El diagnóstico incluye tablas y gráficos legibles de estructura, calidad, distribuciones y relaciones.
- [ ] Outliers se reportan al menos mediante IQR y se contextualizan; Z-score puede usarse como complemento.
- [ ] El análisis orientado a regresión informa sesgo y curtosis, correlaciones y VIF de predictores.
- [ ] Las recomendaciones de transformación consideran las condiciones de los datos y no sobrescriben automáticamente el dataset.
- [ ] La clase se exporta con `%%writefile` a `Integradora 1/herramientas_eda.py` y se importa desde ese archivo.
- [ ] Las conclusiones conectan métricas observadas con acciones razonadas.
- [ ] La revisión de ydata-profiling/Sweetviz se presenta como contraste de referencia; no se atribuye al enunciado una obligación de benchmarking formal.

## 6. Referencias disponibles en el repositorio

- [`Integradora 1/Ejercicio`](Integradora%201/Ejercicio): objetivo e instrucciones oficiales.
- [`Unidad 1/README_Practico.md`](Unidad%201/README_Practico.md): secuencia y ejemplos de los diez puntos.
- [`Unidad 1/Practica1-EDA.ipynb`](Unidad%201/Practica1-EDA.ipynb): práctica de EDA de referencia.
- [`Integradora 1/herramientas_eda.py`](Integradora%201/herramientas_eda.py): implementación inicial que puede orientar el punto de partida.
- [`Integradora 1/herramientas_EDA_(3).ipynb`](Integradora%201/herramientas_EDA_(3).ipynb), [`diagnostico_ydata.html`](Integradora%201/diagnostico_ydata.html) y [`reporte_adult_sweetviz.html`](Integradora%201/reporte_adult_sweetviz.html): materiales de consulta para las herramientas de referencia.
