# Plan paso a paso — Actividad Integradora: AutoEDA (10 puntos)

Objetivo (archivo `Ejercicio`): programar una **herramienta propia de EDA automatizado** en Python (clase reutilizable) que, con una sola llamada, genere un diagnóstico **estadístico, visual y de calidad de datos**, orientado a detectar **multicolinealidad, sesgo y outliers** (problemas típicos en regresión). Debe cubrir **los 10 puntos de `Pasos_EDA`**.

## Hallazgo clave de los archivos

La clase `AutoEDA` del cuaderno `herramientas_EDA.ipynb` es solo un punto de partida: cubre 4 bloques (nulos/duplicados, resumen estadístico, outliers IQR, correlación de Pearson). **Faltan** tipos de variable, desbalance de clases, distribuciones (skew/kurtosis), análisis bivariado y multivariado (VIF). Este plan completa esos huecos usando como base el código de `README_Practico.md` (Práctica 1).

## Mapa: punto del EDA → fuente → método de la clase

| # | Punto (`Pasos_EDA`) | Referencia en los archivos | Método a implementar | Estado en `AutoEDA` del cuaderno |
|---|---|---|---|---|
| 1 | Conocer el dataset | Práctica 1: `head`, `tail`, `shape`, `sample`, `columns` | `_conocer_dataset()` | Falta |
| 2 | Tipo de variable | Práctica 1: `info`, `dtypes`, `select_dtypes` | `_tipos_variable()` | Parcial (`num_cols`, `cat_cols`) |
| 3 | Calidad de datos | Práctica 1: `duplicated`, valores `?` | `_calidad_datos()` | Parcial (duplicados) |
| 4 | Valores faltantes | Práctica 1: `isnull().sum()`, `% nulos` | `_valores_faltantes()` | Parcial |
| 5 | Outliers | Práctica 1: boxplot + IQR | `_outliers()` | Parcial (solo conteo, sin gráfico) |
| 6 | Estadística descriptiva | Práctica 1: `describe(include="all")` | `_estadistica_descriptiva()` | Parcial |
| 7 | Desbalance de clases | Práctica 1: `value_counts(normalize=True)`, `countplot` | `_desbalance_clases()` | Falta |
| 8 | Análisis univariado / distribuciones | Práctica 1: `histplot(kde=True)`, `skew()`, `kurt()` | `_analisis_univariado()` | Falta |
| 9 | Análisis bivariado | Correlación de Pearson, variable vs. objetivo | `_analisis_bivariado()` | Parcial (heatmap) |
| 10 | Análisis multivariado | Heatmap, pairplot, **VIF** (multicolinealidad) | `_analisis_multivariado()` | Falta |

---

## Fase 0 — Preparación

1. **Elegir dataset** (Kaggle o UCI; libre, según tu interés). Criterios: ≥ 5 columnas numéricas, ≥ 1 categórica, y una **variable objetivo** (categórica para el punto 7; si es numérica, discretizarla o analizarla con histograma).
2. Crear el entorno: `pip install pandas numpy matplotlib seaborn statsmodels ydata-profiling sweetviz`.
3. **Verificación:** `pd.read_csv(...)` carga sin error y `df.shape` coincide con lo esperado.

## Fase 1 — Referencia con herramientas automáticas

4. Ejecutar `ProfileReport` (ydata-profiling) y `sweetviz.analyze` sobre tu dataset, como en el cuaderno. Sirven como **referencia/benchmark** de lo que debe producir tu herramienta.
5. **Verificación:** se generan los `.html`; anota qué secciones muestran (nulos, correlaciones, distribuciones) para comparar al final.

## Fase 2 — Construir la clase `AutoEDA` (archivo `herramientas_eda.py`)

Trabaja **un método a la vez** y valida cada uno antes de pasar al siguiente.

6. Copiar el esqueleto del cuaderno (`__init__` con `df`, `target_col`, `num_cols`, `cat_cols`). Mejoras recomendadas: `self.df = df.copy()` (no mutar el original) y dtype `category` incluido.
7. Implementar los métodos de los puntos **1 a 4** (conocimiento, tipos, calidad, faltantes).
   - Tratar `"?"` como nulo: `df.replace("?", np.nan)` (o `na_values` al leer).
   - **Verificación:** compara conteos con `df.isnull().sum()` y `df.duplicated().sum()` hechos a mano.
8. Implementar los puntos **5 y 6** (outliers IQR con boxplots; `describe`).
   - Reportar por columna: límites `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`, cantidad y %.
   - **Verificación:** el conteo de una columna coincide con el cálculo manual.
9. Implementar el punto **7** (desbalance): `value_counts(normalize=True)` + `countplot`. Emitir una **alerta** si la clase minoritaria < 20 %.
10. Implementar el punto **8** (distribuciones): histograma + KDE, `skew()` y `kurt()` por variable numérica. Señalar **sesgo** si `|skew| > 1`.
11. Implementar el punto **9** (bivariado): correlación de Pearson (heatmap) y variable numérica vs. objetivo (boxplot por clase). Listar pares con `|r| > 0.8`.
12. Implementar el punto **10** (multivariado): **VIF** con `statsmodels.stats.outliers_influence.variance_inflation_factor` (alerta si VIF > 5–10) y `pairplot` (muestreando filas si el dataset es grande).
13. Armar `run_analysis()` que llame los 10 métodos **en orden** con encabezados `=== N. NOMBRE ===`.
14. Guardar el archivo con `%%writefile herramientas_eda.py` (como en el cuaderno).
15. **Verificación:** `from herramientas_eda import AutoEDA` funciona en una celda nueva, sin errores.

### Esqueleto sugerido

```python
%%writefile herramientas_eda.py
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

class AutoEDA:
    def __init__(self, df, target_col=None):
        self.df = df.copy()
        self.target_col = target_col
        self.num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

    def _conocer_dataset(self): ...        # 1
    def _tipos_variable(self): ...         # 2
    def _calidad_datos(self): ...          # 3
    def _valores_faltantes(self): ...      # 4
    def _outliers(self): ...               # 5
    def _estadistica_descriptiva(self): ...# 6
    def _desbalance_clases(self): ...      # 7
    def _analisis_univariado(self): ...    # 8
    def _analisis_bivariado(self): ...     # 9
    def _analisis_multivariado(self): ...  # 10 (heatmap + VIF)

    def run_analysis(self):
        for paso in (self._conocer_dataset, self._tipos_variable,
                     self._calidad_datos, self._valores_faltantes,
                     self._outliers, self._estadistica_descriptiva,
                     self._desbalance_clases, self._analisis_univariado,
                     self._analisis_bivariado, self._analisis_multivariado):
            paso()
```

> Cada método debe imprimir un **diagnóstico en lenguaje claro** (ej.: "⚠ VIF de `X` = 12.4 → multicolinealidad alta"), no solo tablas. Eso es lo que lo hace "diagnóstico automatizado".

## Fase 3 — Probar y documentar

16. Ejecutar en el cuaderno:
    ```python
    eda = AutoEDA(df, target_col="tu_objetivo")
    eda.run_analysis()
    ```
17. **Verificación final (checklist):**
    - [ ] Aparecen los 10 bloques numerados.
    - [ ] Corre con una sola línea tras instanciar la clase.
    - [ ] Detecta y reporta **multicolinealidad** (VIF/correlaciones), **sesgo** (skew) y **outliers** (IQR).
    - [ ] Funciona sin `target_col` (el punto 7 se omite con aviso en vez de fallar).
    - [ ] Los resultados son coherentes con los de ydata-profiling/Sweetviz (Fase 1).
18. Descargar `herramientas_eda.py` (`files.download`) y probar importándolo en un cuaderno limpio.

## Fase 4 — Entrega

19. Redactar conclusiones breves por cada punto (qué problema encontró tu herramienta en *tu* dataset y qué harías: imputar, eliminar duplicados, transformar con log si hay sesgo, quitar variables con VIF alto, balancear clases).
20. Entregar: cuaderno `.ipynb`, `herramientas_eda.py` y (opcional) los reportes HTML de referencia.

## Errores comunes a evitar

- Asumir que no hay nulos: en datasets tipo Adult vienen como `"?"` (o `" ?"` con espacio).
- Calcular VIF con nulos o columnas no numéricas → filtrar `num_cols` y `dropna()` antes.
- `pairplot` sobre muchas columnas/filas: muestrear (`df.sample(2000)`) y limitar columnas.
- Mutar el DataFrame original dentro de la clase (usar `.copy()`).
