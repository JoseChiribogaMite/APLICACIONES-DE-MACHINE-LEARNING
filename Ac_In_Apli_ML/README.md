# Actividad Integradora 1 — Herramienta propia de EDA automatizado (AutoEDA)

**Asignatura:** Aplicaciones de Machine Learning · **Dataset:** `insurance.csv` (gastos médicos personales) · **Problema:** regresión con target continuo `charges`

## Qué contiene esta carpeta

- **`Reg_Act_Int_uno.ipynb`** — cuaderno principal de la actividad: los 10 puntos de EDA, la ejecución automatizada, el benchmark y la sección final de interpretación.
- **`herramientas_eda.py`** — clase reutilizable `AutoEDA` exportada con `%%writefile` (idéntica a la copia de `Integradora 1/`).
- **`diagnostico_ydata.html`** — reporte de referencia generado con `ydata-profiling`.
- **`reporte_sweetviz.html`** — reporte de referencia generado con `Sweetviz`.

## Qué hace la herramienta

`AutoEDA` procesa un DataFrame y genera, de forma **100 % automatizada**, un diagnóstico **estadístico, visual y de calidad de datos** orientado a modelos de **regresión**, cubriendo los 10 puntos de EDA de clase:

1. Importar librerías (pandas, numpy, matplotlib, seaborn + statsmodels para VIF)
2. Conocer el dataset (`head`, `tail`, `shape`, `sample`, `columns`, `memory_usage`)
3. Tipo de variable (`dtypes`, `select_dtypes`)
4. Calidad de datos (`duplicated`, valores centinela `?` `-` `N/A`, espacios)
5. Valores faltantes (`isnull`, porcentaje y severidad)
6. Outliers (regla IQR de Tukey 1.5×/3.0× + boxplots)
7. Estadística descriptiva (`describe`, media, mediana, std, skew, curtosis)
8. Variable objetivo (distribución en regresión / balance de clases en clasificación)
9. Análisis univariado (histogramas + KDE, skewness, curtosis)
10. Análisis bivariado/multivariado (scatter, boxplot por categoría, heatmap, **VIF**)

**Principio metodológico:** la herramienta **solo diagnostica y reporta**. Ninguna alerta estadística se convierte en decisión automática de limpieza: cada acción (eliminar, imputar, transformar) se propone como **recomendación para decisión humana** en la sección final del cuaderno.

## Cómo ejecutar

**Google Colab:** abrir `Reg_Act_Int_uno.ipynb` (Archivo → Subir notebook) y ejecutar **Entorno de ejecución → Ejecutar todas**.

**Jupyter local:**

```bash
pip install pandas numpy matplotlib seaborn statsmodels ydata-profiling sweetviz ipywidgets
jupyter notebook Reg_Act_Int_uno.ipynb
```

Orden de las celdas: instalación e importaciones (punto 1) → carga del dataset (punto 2) → celda `%%writefile` que genera `herramientas_eda.py` → `eda.run_analysis()` (puntos 2–10) → benchmark (genera los dos HTML) → sección final de interpretación.

## Hallazgos principales del dataset (resumen)

- **Calidad:** 1 fila duplicada (0.07 %), 0 valores nulos, sin valores centinela.
- **Outliers:** `charges` concentra 139 atípicos (10.39 %) y 6 extremos; `bmi` 9 atípicos (0.67 %).
- **Sesgo:** `charges` con skewness 1.52 (cola derecha); media 13 270 vs mediana 9 382.
- **Multicolinealidad:** ausente — sin pares con |r| ≥ 0.80 y VIF ≈ 1.0 en todos los predictores.
- **Relación con el target:** `smoker` domina (gasto medio 8 434 no fumador vs 32 050 fumador); correlaciones: `age` 0.299, `bmi` 0.198, `children` 0.068.

## Requisitos del enunciado cubiertos

- [x] Herramienta propia (`class AutoEDA`, reutilizable) y automatización 100 % en una llamada
- [x] Diagnóstico estadístico, visual y de calidad de datos
- [x] Multicolinealidad (correlación + VIF), sesgo (skewness/curtosis) y outliers (IQR)
- [x] Los 10 puntos de EDA de clase
- [x] `%%writefile` para exportar la clase
- [x] Referencia comparativa con `ydata-profiling` y `Sweetviz`
- [x] Sección final: interpreta los hallazgos y recomienda acciones **sin presentar las alertas estadísticas como decisiones automáticas de limpieza**
