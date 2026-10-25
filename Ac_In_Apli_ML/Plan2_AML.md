# Plan de Trabajo — Actividad Integradora 1: Herramienta Propia de EDA Automatizado

> Documento de trabajo paso a paso generado a partir de `Integradora 1/Ejercicio`.
> Asignatura: **Aplicaciones de Machine Learning**.

---

## 1. Enunciado oficial (requisitos obligatorios)

Según `Integradora 1/Ejercicio`, la actividad consiste en:

> Diseñar y programar una herramienta **propia** de EDA automatizado en Python capaz de procesar un dataset y generar de forma **100 % automatizada** un diagnóstico **estadístico, visual y de calidad de datos**, optimizado para identificar problemas clave en **modelos de regresión**: la **multicolinealidad**, el **sesgo** y los **valores atípicos**. Se deben incluir **todos los 10 puntos de EDA** vistos en clase.

Requisitos desglosados:

1. **Automatización total**: la herramienta recibe un `DataFrame` (y opcionalmente el nombre del `target`) y ejecuta todo sin código manual por celda.
2. **Orientación a regresión**:
   - **Multicolinealidad** → matriz de correlación + **VIF** (Factor de Inflación de la Varianza).
   - **Sesgo (bias)** → **skewness** y **curtosis**, con recomendación de transformación (`log`, `Box-Cox`).
   - **Valores atípicos (outliers)** → regla **IQR** y **Z-score**, con visualización.
3. **Los 10 puntos de EDA de clase** (ver sección 3).
4. **Arquitectura reutilizable**: implementar como `CLASS` (POO).
5. **Exportación con `%%writefile`**: guardar la clase en `herramientas_eda.py`.
6. **Benchmarking**: comparar contra `ydata-profiling` y `Sweetviz`.
7. **Libertad de dataset** (Kaggle / UCI), orientado a regresión.

---

## 2. Estado actual del repositorio (diagnóstico previo)

| Elemento | Ubicación | Estado |
|---|---|---|
| Enunciado oficial | `Integradora 1/Ejercicio` | ✅ Completo (fuente de requisitos) |
| Cuaderno de referencia (ydata-profiling + Sweetviz) | `Integradora 1/herramientas_EDA_(3).ipynb` | ✅ Disponible |
| Reportes HTML de referencia | `Integradora 1/diagnostico_ydata.html`, `reporte_adult_sweetviz.html` | ✅ Disponibles |
| Clase `AutoEDA` actual | `Integradora 1/herramientas_eda.py` | ⚠️ **Incompleta** |
| Referencia de los 10 puntos | `Unidad 1/Practica1-EDA.ipynb`, `Unidad 1/README_Practico.md` | ✅ Disponible |

**Gap de la clase actual** (`herramientas_eda.py`): solo implementa **4 bloques**:

1. `_diagnose_missing_and_duplicates` → puntos 4 y 5 parcialmente.
2. `_summary_statistics` → punto 7 parcialmente.
3. `_detect_outliers` (IQR) → punto 6 parcialmente (sin gráficos, sin Z-score).
4. `_correlation_analysis` (heatmap Pearson) → parte del punto 10, sin VIF.

**Faltan**: punto 1 (estructura/librerías), punto 2 (conocer dataset), punto 3 (tipo de variable detallado), detección de valores raros (`?`, `-`, `N/A`), punto 8 (distribución del target / desbalance), punto 9 (análisis univariado: histogramas, skew, curtosis), punto 10 completo (bivariado/multivariado), **VIF (multicolinealidad)** y **sesgo (skewness/curtosis)**.

---

## 3. Los 10 puntos de EDA de la clase (canon)

Lista tomada de `Unidad 1/README_Practico.md` (Práctica 1):

| # | Punto de EDA | Operaciones clave |
|---|---|---|
| 1 | **Importar librerías** | `pandas`, `numpy`, `matplotlib`, `seaborn` (+ `statsmodels` para VIF) |
| 2 | **Conocer el dataset** | `head()`, `tail()`, `shape`, `sample()`, `columns`, `memory_usage` |
| 3 | **Tipo de variable** | `info()`, `dtypes`, `select_dtypes` (numéricas vs. categóricas) |
| 4 | **Calidad de datos** | `duplicated()`, valores raros (`?`, `-`, `N/A`, espacios) |
| 5 | **Valores faltantes** | `isnull().sum()`, `%` de nulos, severidad |
| 6 | **Outliers** | boxplot, cuartiles, IQR |
| 7 | **Estadística descriptiva** | `describe()`, media, mediana, std, skew, kurtosis |
| 8 | **Desbalance de clases / target** | `value_counts()`, porcentajes, countplot |
| 9 | **Análisis univariado** | histogramas, KDE, skew, curtosis por variable |
| 10 | **Análisis bivariado/multivariado** | scatter, boxplot por categoría, heatmap de correlación |

---

## 4. Plan de trabajo paso a paso

### Fase A — Preparación y selección del dataset

- **A1.** Elegir un dataset de **regresión** con variable objetivo continua. Recomendados (fáciles de cargar y que evidencian sesgo/outliers/multicolinealidad):
  - *Medical Cost (Insurance)* → `target = "charges"` (sesgo a la derecha, outliers).
  - *California Housing* → `target = "MedHouseVal"` (multicolinealidad `AveRooms`/`AveBedrms`).
  - *Real Estate Valuation (UCI)* → `target = "Y house price of unit area"`.
- **A2.** Crear el notebook de trabajo `Integradora 1/ActividadIntegradora_EDA.ipynb`.
- **A3.** En la primera celda de código, cargar el dataset con `pd.read_csv(...)` y guardarlo en `df`.
- **Verificación**: `df.shape` y `df.head()` muestran filas/columnas esperadas sin errores.

---

### Fase B — Implementación de la clase `AutoEDA` (10 puntos + regresión)

Trabajar en el notebook, en **una celda que inicie con `%%writefile herramientas_eda.py`** (esto sobrescribe el archivo actual con la versión completa). Estructura objetivo:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

class AutoEDA:
    def __init__(self, df, target_col=None):
        self.df = df
        self.target_col = target_col
        self.num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.report = {}   # diccionario para acumular hallazgos del diagnóstico

    # --- Punto 2: conocer el dataset ---
    def _dataset_overview(self): ...
    # --- Punto 3: tipo de variable ---
    def _variable_types(self): ...
    # --- Punto 4: calidad de datos (duplicados + valores raros) ---
    def _data_quality(self): ...
    # --- Punto 5: valores faltantes ---
    def _missing_values(self): ...
    # --- Punto 6: outliers (IQR + Z-score + boxplot) ---
    def _outliers(self): ...
    # --- Punto 7: estadística descriptiva ---
    def _descriptive_stats(self): ...
    # --- Punto 8: distribución del target / desbalance ---
    def _target_distribution(self): ...
    # --- Punto 9: análisis univariado (hist + skew + curtosis) ---
    def _univariate(self): ...
    # --- Punto 10: análisis bivariado/multivariado ---
    def _bivariate(self): ...
    # --- Diagnóstico específico de regresión: multicolinealidad (VIF) ---
    def _multicollinearity(self): ...
    # --- Diagnóstico específico de regresión: sesgo (skewness/curtosis) ---
    def _bias_skewness(self): ...
    # --- Método principal ---
    def run_analysis(self): ...
```

Pasos de implementación (uno por cada método, en orden):

- **B1. `__init__`**: detectar automáticamente `num_cols`, `cat_cols` y validar `target_col` (si se pasa, debe existir en `df`). Guardar `self.report = {}`.
- **B2. `_dataset_overview`** (punto 2): imprimir `shape`, `columns`, `head(3)`, `tail(3)`, `sample(3)` y `memory_usage(deep=True).sum()`.
- **B3. `_variable_types`** (punto 3): listar numéricas vs. categóricas, cardinalidad (nº de valores únicos) y `dtypes`. Señalar posibles ID de alta cardinalidad.
- **B4. `_data_quality`** (punto 4): contar `duplicated()`; detectar valores centinela `?`, `-`, `N/A`, `None`, `""`; advertir espacios sobrantes en strings (`.str.strip()` vs original).
- **B5. `_missing_values`** (punto 5): `isnull().sum()` y `%`; clasificar severidad (<5 % imputable, 5–30 % avanzado, >30 % descartar).
- **B6. `_outliers`** (punto 6): por cada columna numérica calcular Q1/Q3/IQR y contar outliers; añadir conteo por **Z-score** (`|z| > 3`); graficar boxplots.
- **B7. `_descriptive_stats`** (punto 7): `describe()` de numéricas y categóricas; incluir `skew()` y `kurt()`.
- **B8. `_target_distribution`** (punto 8): si hay `target_col`, `value_counts()` (discreto) o histograma (continuo), porcentajes y `countplot`/`histplot`. Registrar desbalance si aplica.
- **B9. `_univariate`** (punto 9): histograma + KDE y boxplot por cada columna numérica; `barplot` de frecuencias para categóricas; reportar `skew` y `kurt` de cada numérica.
- **B10. `_bivariate`** (punto 10): scatter de cada predictor numérico vs. `target_col`; boxplot del target por categoría; heatmap de correlación entre numéricas.
- **B11. `_multicollinearity`** (regresión): heatmap de Pearson (umbral de alerta |r| > 0.8) y **VIF** con `variance_inflation_factor` sobre las numéricas estandarizadas (rellenar NaN, descartar columnas constantes). Marcar VIF > 10 como multicolinealidad severa.
- **B12. `_bias_skewness`** (regresión): `skew()` y `kurt()` del target y predictores; recomendar transformación `log1p`/`Box-Cox` cuando `|skew| > 1`.
- **B13. `run_analysis`**: invocar los métodos en orden, capturar hallazgos en `self.report` y finalizar con un resumen consolidado de alertas.

**Verificación parcial**: ejecutar `eda = AutoEDA(df, target_col='...'); eda.run_analysis()` y comprobar que **cada uno de los 10 puntos + VIF + sesgo** imprime salida sin excepciones.

---

### Fase C — Exportación del código con `%%writefile`

- **C1.** Asegurar que toda la clase `AutoEDA` esté en una única celda cuya primera línea sea `%%writefile herramientas_eda.py`.
- **C2.** Ejecutar esa celda para (re)generar `Integradora 1/herramientas_eda.py`.
- **C3.** Importar desde el archivo y confirmar que funciona igual que la versión en celda:
  ```python
  from herramientas_eda import AutoEDA
  eda = AutoEDA(df, target_col='...')
  eda.run_analysis()
  ```
- **Verificación**: `Integradora 1/herramientas_eda.py` contiene la clase completa y el import no lanza error.

---

### Fase D — Ejecución automatizada y validación

- **D1.** Ejecutar `eda.run_analysis()` de forma limpia (kernel reiniciado) y capturar la salida completa.
- **D2.** Verificar que se generan los gráficos (boxplots, histogramas, heatmaps, scatter) sin errores.
- **D3.** Anotar en el notebook las **conclusiones** del diagnóstico: qué variables tienen VIF alto, cuáles presentan sesgo y cuántos outliers hay.
- **Verificación**: salida reproducible en una sola pasada, sin intervención manual.

---

### Fase E — Benchmarking contra herramientas de referencia

- **E1.** Abrir `Integradora 1/herramientas_EDA_(3).ipynb` y revisar `ydata-profiling` y `Sweetviz`.
- **E2.** Generar (o reutilizar) `diagnostico_ydata.html` y `reporte_adult_sweetviz.html`.
- **E3.** Redactar una comparativa breve: qué aporta cada herramienta y qué cubre la propia (`AutoEDA`) que ellas no priorizan (p. ej., **VIF** y recomendación de transformación por sesgo).
- **Verificación**: tabla comparativa en el notebook con las 3 herramientas.

---

### Fase F — Reporte interpretativo y entrega

- **F1.** Redactar en el notebook un resumen final con los 3 problemas clave de regresión detectados (multicolinealidad, sesgo, outliers) y acciones recomendadas.
- **F2.** Confirmar que los **10 puntos** están cubiertos y mapeados (usar la tabla de la sección 5).
- **F3.** Limpiar el notebook (celdas de prueba eliminadas) y exportarlo con salidas visibles.
- **F4.** Commit al repositorio con `herramientas_eda.py` actualizado y el notebook de la integradora.
- **Verificación**: `git status` limpio y revisión final del notebook renderizado.

---

## 5. Mapeo requisito → entregable (rúbrica de cumplimiento)

| Requisito del enunciado | Dónde se cumple |
|---|---|
| Herramienta propia en Python | Clase `AutoEDA` en `herramientas_eda.py` |
| 100 % automatizada | Método `run_analysis()` dispara todo el flujo |
| Diagnóstico estadístico | `_descriptive_stats`, `_target_distribution`, `_bias_skewness` |
| Diagnóstico visual | `_univariate`, `_bivariate`, `_outliers`, `_correlation/heatmaps` |
| Calidad de datos | `_data_quality`, `_missing_values` |
| Multicolinealidad | `_multicollinearity` (correlación + VIF) |
| Sesgo | `_bias_skewness` (skewness + curtosis + transformación) |
| Valores atípicos | `_outliers` (IQR + Z-score + boxplot) |
| 10 puntos de EDA | Métodos B2–B10 (tabla sección 3) |
| Reutilización (`CLASS`) | POO con `__init__` y métodos privados |
| `%%writefile` | Fase C |
| Benchmarking vs. referencia | Fase E (ydata-profiling / Sweetviz) |
| Dataset libre (Kaggle/UCI) | Fase A |

---

## 6. Checklist final de aceptación

- [ ] La clase `AutoEDA` se importa desde `herramientas_eda.py` sin errores.
- [ ] `run_analysis()` ejecuta los **10 puntos** + **VIF** + **sesgo** en una sola llamada.
- [ ] Se generan gráficos de outliers, distribuciones y correlación.
- [ ] Se reporta **VIF > 10** (multicolinealidad) si existe.
- [ ] Se reporta **|skew| > 1** con recomendación de transformación.
- [ ] Se reportan outliers por **IQR** y **Z-score**.
- [ ] Existe comparativa con `ydata-profiling` y `Sweetviz`.
- [ ] El notebook incluye conclusiones interpretativas y está limpio para entrega.
