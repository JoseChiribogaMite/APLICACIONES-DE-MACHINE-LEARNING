# Aplicaciones de Machine Learning

Repositorio de prácticas del curso **Aplicaciones de Machine Learning**. Reúne notebooks de dos unidades:

- **Unidad 1** — Fundamentos, Análisis Exploratorio de Datos (EDA) e IA Explicable (XAI).
- **Unidad 2** — Aprendizaje supervisado (clasificación y regresión) y optimización de hiperparámetros.

> 📄 Documentación por unidad: [`Unidad 1/README.md`](Unidad%201/README.md) · [`Unidad 2/README.md`](Unidad%202/README.md)
> 🧠 Evaluación interactiva: [`cuestionario.html`](cuestionario.html)

---

## Estructura del repositorio

```
.
├── README.md                     ← este archivo
├── cuestionario.html             ← quiz interactivo de 20 preguntas
├── Unidad 1/
│   ├── Practica1-EDA.ipynb              → Análisis Exploratorio de Datos (Adult Income)
│   ├── Practica2_1_FeatureImportance.ipynb → XAI: importancia de características
│   ├── Practica2_2_SHAPvalues.ipynb     → XAI: valores SHAP
│   ├── Practica3.ipynb                  → Series de tiempo y detección de cambio
│   ├── adult.csv                         → dataset usado en la Práctica 1
│   └── AML unidad 1.pptx                 → diapositivas de la unidad
└── Unidad 2/
    ├── Practica4.ipynb                   → AdaBoost, curva ROC y AUC
    ├── Practica5.ipynb                   → Regresión + RandomizedSearchCV
    ├── Practica6A_OptBayesiana(1).ipynb  → Optimización Bayesiana con Optuna
    ├── Practica6B_OptBayesiana.ipynb     → BayesSearchCV (scikit-optimize)
    ├── Unid2_ Aplicaciones de ML en Clasificación.pdf
    └── unidad 2 - Aprendizaje supervisado Regresion (4).pptx
```

---

## Análisis y resumen del repositorio

### Unidad 1 — Fundamentos, EDA e IA Explicable (XAI)

| Práctica | Tema | Dataset | Modelos / Librerías clave |
|---|---|---|---|
| `Practica1-EDA` | Análisis Exploratorio de Datos | Adult Income (Censo 1994, ~32k filas, 14 atributos) | pandas, NumPy, Matplotlib, Seaborn |
| `Practica2_1_FeatureImportance` | XAI: importancia de características | California Housing | RandomForestRegressor, `permutation_importance` |
| `Practica2_2_SHAPvalues` | XAI: valores SHAP | California Housing | XGBRegressor, shap |
| `Practica3` | Series de tiempo, tendencia y detección de cambio | Serie sintética (pendiente + ruido) | ARIMA, LinearRegression, `mean_squared_error` |

**Idea central:** antes de predecir hay que *entender* los datos (EDA) y después *explicar* el modelo (XAI). La unidad cierra con una introducción a series de tiempo, donde se aprende a detectar un cambio de tendencia con regresión lineal y ARIMA.

### Unidad 2 — Aprendizaje supervisado y optimización de hiperparámetros

| Práctica | Tema | Dataset | Modelos / Librerías clave |
|---|---|---|---|
| `Practica4` | Clasificación, curva ROC y AUC | Datos sintéticos (1D) | AdaBoostClassifier, `roc_curve`, `roc_auc_score` |
| `Practica5` | Regresión + búsqueda aleatoria | Diabetes | RandomForestRegressor, `RandomizedSearchCV`, RMSE |
| `Practica6A` | Optimización Bayesiana | Breast Cancer | Optuna (`TPESampler`), XGBClassifier, Grid/Randomized |
| `Practica6B` | Búsqueda bayesiana con scikit-optimize | Breast Cancer | `BayesSearchCV`, RandomForestClassifier |

**Idea central:** clasificar con métricas que van más allá de la exactitud (ROC/AUC), regresar con métricas de error (RMSE) y afinar los modelos eligiendo hiperparámetros de forma eficiente: búsqueda exhaustiva (Grid), aleatoria (Randomized) o guiada por un modelo probabilístico (Bayesiana).

### Temas transversales

- Librerías del ecosistema: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `shap`, `optuna`, `scikit-optimize`, `statsmodels`.
- Flujo de trabajo repetido en todas las prácticas: cargar datos → explorar → dividir en train/test → entrenar → evaluar → (opcional) explicar u optimizar.
- Énfasis en la **interpretabilidad** (XAI) y en la **selección de hiperparámetros** como habilidades prácticas del ML aplicado.

---

## Cómo usar este repositorio

1. Abre los notebooks en Google Colab o Jupyter.
2. Sigue el orden de las prácticas (1 → 6) para recorrer la curva de aprendizaje completa.
3. Usa los README de cada unidad para repasar los conceptos clave.
4. Pon a prueba lo aprendido con el [`cuestionario.html`](cuestionario.html).
