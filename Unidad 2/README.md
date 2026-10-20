# Unidad 2 — Aprendizaje supervisado y optimización de hiperparámetros

Esta unidad profundiza en el aprendizaje supervisado (clasificación y regresión), las métricas de evaluación y la búsqueda de los mejores hiperparámetros, desde la búsqueda exhaustiva hasta la optimización bayesiana.

## Prácticas de la unidad

- **Práctica 4 — Clasificación y curva ROC:** AdaBoost, probabilidades, ROC y AUC.
- **Práctica 5 — Regresión + búsqueda aleatoria:** Random Forest con `RandomizedSearchCV`.
- **Práctica 6A — Optimización Bayesiana (Optuna):** búsqueda guiada con `TPESampler`.
- **Práctica 6B — Búsqueda bayesiana (scikit-optimize):** `BayesSearchCV`.

---

## 15 temas y conceptos importantes

1. **Aprendizaje supervisado: clasificación vs. regresión**
   En *clasificación* la salida es una categoría o etiqueta (p. ej. ingreso ≤/> 50K, tumor benigno/maligno); en *regresión* la salida es un valor continuo (precio de una casa, nivel de glucosa). Ambos usan datos etiquetados.

2. **Clasificación binaria**
   Caso particular de clasificación con solo dos clases posibles. Las prácticas usan datasets binarios como *Breast Cancer* (maligno/benigno) y *Adult Income* (≤50K / >50K).

3. **AdaBoost (boosting adaptativo)**
   Algoritmo de *boosting* que combina varios clasificadores débiles, dando más peso a los ejemplos mal clasificados en cada iteración. En scikit-learn: `AdaBoostClassifier`.

4. **Matriz de confusión y métricas de clasificación**
   Tabla que cruza predicciones vs. valores reales (verdaderos/falsos positivos y negativos). De ella derivan exactitud, precisión, sensibilidad y especificidad.

5. **Exactitud (accuracy) y sus limitaciones**
   Proporción de aciertos. Es engañosa con clases desbalanceadas: un modelo que predice siempre la clase mayoritaria puede tener alta exactitud sin ser útil. Por eso se recurre a métricas adicionales.

6. **Probabilidades de predicción (`predict_proba`)**
   En lugar de solo la etiqueta, devuelve la *probabilidad* de pertenencia a cada clase. `predict_proba(X)[:, 1]` da la probabilidad de la clase positiva, base para la curva ROC y para ajustar umbrales.

7. **Curva ROC y AUC**
   La curva ROC grafica la *tasa de verdaderos positivos* (TPR) frente a la *tasa de falsos positivos* (FPR) a distintos umbrales. El **AUC** (área bajo la curva) resume la calidad: 1.0 = separación perfecta, 0.5 = azar.

8. **Umbral de decisión, FPR y TPR**
   El *umbral* define a partir de qué probabilidad se asigna la clase positiva; moverlo cambia el equilibrio entre FPR y TPR. `roc_curve` devuelve `fpr`, `tpr` y `thresholds` para cada posible umbral.

9. **Regresión y métricas de error (MSE y RMSE)**
   `mean_squared_error` (MSE) promedia los errores al cuadrado; **RMSE** es su raíz cuadrada, expresada en las mismas unidades que la variable objetivo y más interpretable.

10. **Random Forest para regresión**
    El mismo ensemble de árboles usado en clasificación, adaptado a salidas continuas (`RandomForestRegressor`). Promedia las predicciones de muchos árboles para reducir la varianza.

11. **Parámetros vs. hiperparámetros**
    Los *parámetros* los aprende el modelo con los datos (pesos, divisiones de los árboles); los *hiperparámetros* se fijan *antes* de entrenar (`n_estimators`, `max_depth`, `min_samples_split`, `min_child_weight`). Elegirlos bien es el objetivo de la optimización.

12. **Validación cruzada (cross-validation)**
    Divide los datos en `k` particiones y entrena/evalúa `k` veces, rotando la partición de prueba. El parámetro `cv=5` indica 5 particiones. Da una estimación más estable que una sola división.

13. **Búsqueda en cuadrícula (`GridSearchCV`)**
    Prueba *todas* las combinaciones de un `param_grid` definido. Es exhaustiva y garantiza encontrar la mejor combinación del espacio, pero es costosa cuando el espacio crece.

14. **Búsqueda aleatoria (`RandomizedSearchCV`)**
    Muestrea `n_iter` combinaciones al azar desde distribuciones (p. ej. `randint`). Más rápida que Grid y suele encontrar buenos resultados sin explorar todo el espacio.

15. **Optimización Bayesiana (Optuna / `TPESampler` / `BayesSearchCV`)**
    A diferencia de Grid y Random, *guía* la búsqueda usando un modelo probabilístico del rendimiento observado, concentrando las pruebas en las zonas prometedoras. Con **Optuna** se define una función objetivo (`create_study`, `optimize`, `n_trials`); con **scikit-optimize** se usa `BayesSearchCV` y espacios `Real`, `Integer` y `Categorical`.

---

## Resumen de la unidad

La Unidad 2 completa el ciclo del ML supervisado: clasificar evaluando con ROC/AUC y probabilidades (Práctica 4), regresar evaluando con RMSE (Práctica 5) y, en ambos casos, encontrar los mejores hiperparámetros comparando tres estrategias de búsqueda —Grid, Random y Bayesiana— con las librerías `scikit-learn`, `Optuna` y `scikit-optimize`.
