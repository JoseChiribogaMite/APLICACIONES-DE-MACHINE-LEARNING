# Unidad 1 — Fundamentos, EDA e IA Explicable (XAI)

Esta unidad introduce los pilares del Machine Learning aplicado: entender los datos mediante el Análisis Exploratorio (EDA) y explicar los modelos mediante técnicas de IA Explicable (XAI). Cierra con una introducción a las series de tiempo.

## Prácticas de la unidad

- **Práctica 1 — EDA:** análisis exploratorio del dataset *Adult Income*.
- **Práctica 2.1 — Feature Importance:** importancia de características con Random Forest.
- **Práctica 2.2 — SHAP Values:** explicación de predicciones con SHAP y XGBoost.
- **Práctica 3 — Series de tiempo:** tendencia, detección de cambio y ARIMA.

---

## 15 temas y conceptos importantes

1. **Machine Learning y tipos de aprendizaje**
   Rama de la IA donde los algoritmos aprenden patrones a partir de datos. Se divide en *supervisado* (con etiquetas), *no supervisado* (sin etiquetas) y *por refuerzo* (aprendizaje por recompensa).

2. **Análisis Exploratorio de Datos (EDA)**
   Proceso de inspeccionar un dataset para entender su estructura, distribución, valores faltantes y relaciones entre variables *antes* de modelar. Es el primer paso obligatorio de todo proyecto.

3. **NumPy y pandas**
   `NumPy` aporta arreglos numéricos eficientes; `pandas` aporta el `DataFrame`, la estructura tabular central para cargar, limpiar y transformar datos (`read_csv`, `head`, `tail`, `describe`).

4. **Visualización con Matplotlib y Seaborn**
   `matplotlib.pyplot` dibuja gráficos de bajo nivel; `seaborn` (alias `sns`) añade gráficos estadísticos de alto nivel (distribuciones, heatmaps, boxplots) sobre matplotlib.

5. **Dataset Adult Income (Censo de 1994)**
   Dataset de clasificación binaria extraído del censo de EE. UU. (~32,561 registros y 14 atributos). Objetivo: predecir si una persona gana más o menos de **$50,000 USD** al año.

6. **Variables numéricas vs. categóricas**
   Las *numéricas* (`age`, `capital-gain`, `hours-per-week`) admiten operaciones aritméticas; las *categóricas* (`workclass`, `education`, `race`, `income`) representan grupos o etiquetas. Cada tipo requiere un tratamiento distinto.

7. **Valores nulos (NaN) y limpieza de datos**
   Los `NaN` son valores ausentes que distorsionan modelos y estadísticas. La práctica detecta nulos (`isna().sum()`) como paso previo a imputarlos o eliminarlos.

8. **División entrenamiento / prueba (`train_test_split`)**
   Separa los datos en conjunto de *entrenamiento* (aprende) y *prueba* (evalúa) para medir la generalización. En las prácticas se usa `test_size=0.2` y `random_state=42` para reproducibilidad.

9. **Random Forest (bosques aleatorios)**
   *Ensemble* que entrena muchos árboles de decisión sobre submuestras aleatorias de datos y características, combinando sus predicciones. Robusto, poco propenso a sobreajuste y útil tanto en clasificación como en regresión.

10. **XGBoost (gradient boosting)**
    Método de *boosting* que construye árboles secuencialmente, cada uno corrigiendo el error del anterior. Muy potente y habitual en competiciones; aquí se usa como `XGBRegressor` para regresión.

11. **IA Explicable (XAI)**
    Conjunto de técnicas para hacer *transparentes* las decisiones de los modelos, respondiendo a la pregunta: *¿por qué el modelo predijo esto?*. Fundamental en dominios regulados o críticos.

12. **Importancia de características (Feature Importance)**
    Ranking que indica cuánto aporta cada variable a la predicción. Permite priorizar las variables más influyentes y simplificar el modelo.

13. **Importancia por permutación (`permutation_importance`)**
    Técnica *agnóstica al modelo*: se permuta (baraja) una columna y se mide cuánto empeora el rendimiento. Si el error sube mucho, esa variable es importante; si no cambia, es prescindible.

14. **Valores SHAP**
    Basados en la teoría de juegos (valores de Shapley), cuantifican la *contribución individual* de cada característica a cada predicción concreta, permitiendo explicar casos puntuales (no solo el promedio global).

15. **Series de tiempo, tendencia y detección de cambio (ARIMA)**
    Una *serie de tiempo* es una secuencia ordenada en el tiempo. La práctica simula una pendiente (tasa de crecimiento) que cambia a mitad de serie para detectar el *cambio de tendencia*, usando `np.cumsum`, regresión lineal (`LinearRegression`), el error `mean_squared_error` y el modelo autorregresivo **ARIMA** de `statsmodels`.

---

## Resumen de la unidad

La Unidad 1 enseña el ciclo: **explorar → modelar → explicar**. Primero se domina el EDA sobre un dataset real de clasificación; luego se entrenan modelos de regresión (Random Forest, XGBoost) y se explica su comportamiento con dos técnicas XAI complementarias (importancia de características y SHAP); por último se introducen las series de tiempo para detectar cambios de tendencia.
