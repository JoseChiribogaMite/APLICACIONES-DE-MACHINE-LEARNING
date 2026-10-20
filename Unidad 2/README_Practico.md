# Unidad 2 — README Práctico

Explicación técnica, lógica y código comentado de cada actividad práctica de la Unidad 2.

## Índice

1. [Práctica 4 — Clasificación, curva ROC y AUC](#práctica-4--clasificación-curva-roc-y-auc)
2. [Práctica 5 — Regresión + búsqueda aleatoria](#práctica-5--regresión--búsqueda-aleatoria)
3. [Práctica 6A — Optimización Bayesiana con Optuna](#práctica-6a--optimización-bayesiana-con-optuna)
4. [Práctica 6B — Búsqueda bayesiana con scikit-optimize](#práctica-6b--búsqueda-bayesiana-con-scikit-optimize)

---

## Práctica 4 — Clasificación, curva ROC y AUC

### Objetivo
Entrenar un clasificador (AdaBoost) y evaluarlo con métricas que van **más allá de la exactitud**: probabilidades, curva ROC y AUC.

### Dataset y librerías
- **Dataset:** sintético 1D. `X_train = [[1],[2],[3],[8],[9],[10]]`, `y_train = [0,0,0,1,1,1]` (tres negativos a la izquierda, tres positivos a la derecha). `X_test = [[4],[5],[6],[7]]`, `y_test = [0,0,1,1]`.
- **Librerías:** `AdaBoostClassifier`, `roc_curve`, `roc_auc_score`, `numpy`, `matplotlib`.

### Lógica de la práctica
1. Crear un dataset sencillo de dos clases separables.
2. Entrenar `AdaBoostClassifier`.
3. Predecir etiquetas y **probabilidades** (`predict_proba`).
4. Obtener la curva ROC (`fpr`, `tpr`, `thresholds`).
5. Calcular el **AUC** y graficar la curva.

### Código técnico comentado

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, roc_auc_score

# 1. Dataset (una sola variable)
X_train = np.array([[1], [2], [3], [8], [9], [10]])
y_train = np.array([0, 0, 0, 1, 1, 1])

X_test = np.array([[4], [5], [6], [7]])
y_test = np.array([0, 0, 1, 1])

# 2. Entrenar AdaBoost
model = AdaBoostClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# 3. Predecir etiquetas
y_pred = model.predict(X_test)

# 4. Obtener PROBABILIDADES de la clase positiva
# [:, 1] -> columna 2 = probabilidad de la clase positiva
y_prob = model.predict_proba(X_test)[:, 1]
print("Probabilidades de la clase positiva:", y_prob)

# 5. Curva ROC
# roc_curve(etiquetas reales, probabilidades de la clase positiva)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
print("FPR:", fpr)
print("TPR:", tpr)
print("Thresholds:", thresholds)

# 6. AUC (área bajo la curva ROC)
auc = roc_auc_score(y_test, y_prob)
print("AUC:", auc)

# 7. Graficar ROC
plt.plot(fpr, tpr, marker="o", label=f"AdaBoost (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")   # línea de azar (AUC = 0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - AdaBoost")
plt.legend()
plt.show()
```

### Concepto clave
- **ROC** grafica TPR (tasa de verdaderos positivos) frente a FPR (tasa de falsos positivos) para distintos umbrales.
- **AUC = 1.0** → separación perfecta; **AUC = 0.5** → azar (línea diagonal).
- `predict_proba` permite ajustar el umbral de decisión en lugar de usar el 0.5 fijo.

---

## Práctica 5 — Regresión + búsqueda aleatoria

### Objetivo
Optimizar los hiperparámetros de un `RandomForestRegressor` usando **RandomizedSearchCV** y evaluar con **RMSE**.

### Dataset y librerías
- **Dataset:** Diabetes (incluido en scikit-learn). Objetivo: predecir la progresión de la enfermedad.
- **Librerías:** `RandomForestRegressor`, `RandomizedSearchCV`, `mean_squared_error`, `randint` (scipy.stats).

### Lógica de la práctica
1. Cargar datos y dividir train/test.
2. Definir el modelo base.
3. Definir un **espacio de búsqueda** con distribuciones (`randint`).
4. Ejecutar `RandomizedSearchCV` con `n_iter` combinaciones y validación cruzada.
5. Tomar el mejor modelo (`best_estimator_`) y calcular RMSE en test.

### Código técnico comentado

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import randint
import numpy as np

# 1. Cargar datos
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(X.shape)   # (442, 10)

# 2. Modelo base
model = RandomForestRegressor(random_state=42)

# 3. Espacio de búsqueda (distribuciones, no valores fijos)
# min_samples_split -> # mínimo de muestras en un nodo para poder dividirse
# randint -> genera enteros aleatorios dentro de un rango
param_distribuciones = {
    'n_estimators':      randint(50, 300),   # número de árboles
    'max_depth':         randint(2, 20),     # profundidad máxima
    'min_samples_split': randint(2, 10),     # mínimo para dividir un nodo
}

# 4. Búsqueda aleatoria
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distribuciones,
    n_iter=30,                        # 30 combinaciones al azar
    cv=5,                             # validación cruzada de 5 pliegues
    scoring='neg_mean_squared_error', # métrica (negativa para maximizar)
    random_state=42
)
random_search.fit(X_train, y_train)

print(random_search.best_params_)   # mejores hiperparámetros
print(random_search.best_score_)    # mejor score en CV

# 5. Evaluación final sobre el test
mejor_model = random_search.best_estimator_   # modelo reentrenado con lo mejor
y_pred = mejor_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')
```

### Concepto clave
`RandomizedSearchCV` muestrea combinaciones **al azar** desde distribuciones, en lugar de probarlas todas (como hace Grid). El `scoring='neg_mean_squared_error'` usa el MSE negativo porque scikit-learn *maximiza* la métrica. El RMSE final se calcula en el conjunto de prueba para medir la generalización real.

---

## Práctica 6A — Optimización Bayesiana con Optuna

### Objetivo
Encontrar los mejores hiperparámetros de un `XGBClassifier` mediante **optimización bayesiana** con Optuna, y compararla con Grid y Random.

### Dataset y librerías
- **Dataset:** Breast Cancer (scikit-learn). Clasificación binaria: tumor maligno o benigno.
- **Librerías:** `XGBClassifier`, `Optuna` (`TPESampler`, `create_study`, `optimize`), `GridSearchCV`, `RandomizedSearchCV`.

### Lógica de la práctica
1. Cargar datos y dividir con `stratify` (mantiene el balance de clases).
2. Definir el espacio de búsqueda y la **función objetivo** que Optuna maximizará.
3. Crear un *study* con sampler `TPESampler` y ejecutar `optimize` con `n_trials`.
4. Inspeccionar el historial y la importancia de los hiperparámetros.
5. Comparar contra `GridSearchCV` y `RandomizedSearchCV`.
6. Reentrenar con los mejores hiperparámetros y evaluar en test.

### Hiperparámetros explicados
- **`max_depth`** — profundidad máxima de cada árbol. Valores altos → modelo complejo y propenso a aprender ruido.
- **`n_estimators`** — cantidad total de árboles. Demasiados pueden sobreajustar si la tasa de aprendizaje es alta.
- **`min_child_weight`** — cuán exigente es el árbol para dividirse. Valor alto evita ramas con pocos datos (previene sobreajuste); valor bajo permite divisiones más complejas.

### Código técnico comentado

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from optuna.samplers import TPESampler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna
from scipy.stats import randint

# 1. Cargar datos
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Dividir con stratify para conservar el balance de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y)

# 3. Modelo base
# eval_metric="logloss" -> mide qué tan acertadas y seguras son las
# probabilidades predichas (penaliza fuertemente los errores confiados).
model = XGBClassifier(eval_metric="logloss", random_state=42)

# 4. Espacio de búsqueda
search_space = {
    "min_child_weight": (1, 10),
    "max_depth":        (2, 10),
    "n_estimators":     (50, 300),
}

# 5. Función OBJETIVO: Optuna la llama con distintas combinaciones
def objetivo(prueba):
    min_child_weight = prueba.suggest_int("min_child_weight", 1, 10)
    max_depth        = prueba.suggest_int("max_depth", 1, 10)
    n_estimators     = prueba.suggest_int("n_estimators", 50, 300)

    model = XGBClassifier(
        min_child_weight=min_child_weight,
        max_depth=max_depth,
        n_estimators=n_estimators,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)   # valor que Optuna maximiza
    return accuracy

# 6. Ejecutar la optimización bayesiana
semilla_optuna = TPESampler(seed=24)   # sampler que guía la búsqueda
contenedor = optuna.create_study(direction="maximize",
                                 sampler=semilla_optuna)
contenedor.optimize(objetivo, n_trials=20)   # 20 experimentos

# 7. Mejores resultados
print("Best parameters:", contenedor.best_params)   # mejores hiperparámetros
print("Best accuracy:", contenedor.best_value)      # mejor puntaje

# 8. Visualizaciones
optuna.visualization.plot_optimization_history(contenedor)  # evolución
optuna.visualization.plot_param_importances(contenedor)     # cuál hiperparámetro importa

# 9. Comparación con Grid y Random
param_grid = {
    "min_child_weight": [1, 3, 5, 7, 10],
    "max_depth":        [1, 2, 4, 6, 8, 10],
    "n_estimators":     [50, 100, 200, 300]
}
grid = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)

param_dist = {
    "min_child_weight": randint(1, 11),
    "max_depth":        randint(1, 11),
    "n_estimators":     randint(50, 301)
}
random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                   n_iter=30, cv=5, scoring="accuracy",
                                   random_state=42)
random_search.fit(X_train, y_train)
print(random_search.best_params_, random_search.best_score_)

# 10. Reentrenar y evaluar con los mejores hiperparámetros
best_params = contenedor.best_params
best_model = XGBClassifier(**best_params, eval_metric="logloss")  # ** desempaqueta el dict
best_model.fit(X_train, y_train)
test_score = best_model.score(X_test, y_test)   # accuracy en test
print(test_score)
```

### Concepto clave
A diferencia de Grid (prueba todo) y Random (prueba al azar), Optuna usa un **modelo probabilístico** (`TPESampler`) que *aprende* de los resultados previos para concentrar las pruebas en las zonas prometedoras. `create_study(direction="maximize")` define que se busca maximizar la métrica (accuracy).

---

## Práctica 6B — Búsqueda bayesiana con scikit-optimize

### Objetivo
Optimizar un `RandomForestClassifier` con `BayesSearchCV`, definiendo espacios de búsqueda **continuos, discretos y categóricos**.

### Dataset y librerías
- **Dataset:** Breast Cancer (scikit-learn).
- **Librerías:** `skopt.BayesSearchCV`, `skopt.space.Real/Integer/Categorical`, `RandomForestClassifier`.

### Lógica de la práctica
1. Cargar datos y dividir con `stratify`.
2. Definir el modelo base y el **espacio de búsqueda** con tipos de variable.
3. Configurar `BayesSearchCV` con `n_iter` evaluaciones y `cv=5`.
4. Ejecutar la optimización y leer `best_params_` / `best_score_`.
5. Evaluar el mejor modelo en test.

### Código técnico comentado

```python
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Cargar datos
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y)

# 2. Modelo base
modelo = RandomForestClassifier(random_state=42)

# 3. Espacio de búsqueda
# A diferencia de Random Search, aquí se definen rangos continuos/discretos
# que el algoritmo explorará de forma GUIADA, no puramente al azar.
espacio_busqueda = {
    "n_estimators":      Integer(50, 500),               # entero
    "max_depth":         Integer(3, 20),                 # entero
    "min_samples_split": Integer(2, 20),                 # entero
    "max_features":      Real(0.1, 1.0, prior="uniform"),# continuo
    "bootstrap":         Categorical([True, False]),     # categórico
}

# 4. Configurar la búsqueda bayesiana
bayes_search = BayesSearchCV(
    estimator=modelo,
    search_spaces=espacio_busqueda,
    n_iter=20,              # número de evaluaciones
    scoring="accuracy",
    cv=5,                   # validación cruzada de 5 pliegues
    random_state=42
)

# 5. Ejecutar la optimización
bayes_search.fit(X_train, y_train)

# 6. Resultados
print("Mejores hiperparámetros:", bayes_search.best_params_)
print(f"Mejor score (CV): {bayes_search.best_score_}")

# 7. Evaluar en test
mejor_modelo = bayes_search.best_estimator_
y_pred = mejor_modelo.predict(X_test)
print(mejor_modelo.score(X_test, y_test))
```

### Concepto clave
`BayesSearchCV` es el equivalente de `GridSearchCV`/`RandomizedSearchCV` pero con **optimización bayesiana**: en cada iteración decide la siguiente combinación a probar en función del rendimiento observado. Los espacios `Real`, `Integer` y `Categorical` permiten mezclar hiperparámetros continuos, enteros y categóricos en una misma búsqueda.

---

## Resumen comparativo de las tres estrategias de búsqueda

| Estrategia | Cómo explora | Costo | Caso de uso |
|---|---|---|---|
| `GridSearchCV` | Todas las combinaciones (exhaustivo) | Alto | Espacios pequeños |
| `RandomizedSearchCV` | Muestreo aleatorio (`n_iter`) | Medio | Espacios grandes, sin guía |
| Bayesiana (Optuna / `BayesSearchCV`) | Guiada por un modelo probabilístico | Bajo (mejor uso del presupuesto) | Espacios grandes y costosos de evaluar |
