# Unidad 1 — README Práctico

Explicación técnica, lógica y código comentado de cada actividad práctica de la Unidad 1.

## Índice

1. [Práctica 1 — Análisis Exploratorio de Datos (EDA)](#práctica-1--análisis-exploratorio-de-datos-eda)
2. [Práctica 2.1 — Importancia de Características (Feature Importance)](#práctica-21--importancia-de-características-feature-importance)
3. [Práctica 2.2 — Valores SHAP](#práctica-22--valores-shap)
4. [Práctica 3 — Series de tiempo, tendencia y detección de cambio](#práctica-3--series-de-tiempo-tendencia-y-detección-de-cambio)

---

## Práctica 1 — Análisis Exploratorio de Datos (EDA)

### Objetivo
Conocer la estructura y calidad del dataset *Adult Income* **antes** de construir cualquier modelo.

### Dataset y librerías
- **Dataset:** Adult Income (Censo de EE. UU., 1994). ~32,561 registros y 14 atributos. Objetivo: predecir si una persona gana más o menos de $50,000 USD al año.
- **Librerías:** `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`.

### Lógica de la práctica
El EDA sigue una secuencia de 10 etapas que van de lo general a lo específico:

1. **Importar librerías** — herramientas base del análisis.
2. **Conocer el dataset** — primeras/last filas, forma, muestra, columnas.
3. **Tipo de variable** — identificar numéricas vs. categóricas.
4. **Calidad de datos** — detectar duplicados y valores raros (`?`).
5. **Valores faltantes** — localizar y eliminar nulos.
6. **Outliers** — detectar valores atípicos con boxplot y cuartiles.
7. **Estadística descriptiva** — medidas de tendencia central y dispersión.
8. **Desbalance de clases** — distribución de la variable objetivo.
9. **Análisis univariado** — distribuciones de cada variable.
10. **Análisis bivariado/multivariado** — relaciones entre variables.

### Código técnico comentado

```python
# 0. Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv("adult.csv")

# 1. Conocer el dataset
df.head(4)      # primeras filas
df.tail()       # últimas filas
df.shape        # (filas, columnas) -> (32561, 15)
df.sample(6)    # muestra aleatoria
df.columns      # nombres de columnas

# 2. Tipo de variable
df.info()                            # tipos y nulos por columna
df.dtypes                            # dtype de cada columna
df.select_dtypes(include="object")   # solo categóricas
df.select_dtypes(include="number")   # solo numéricas

# 3. Calidad de datos: duplicados
df.duplicated().sum()                # cuántas filas duplicadas
df[df.duplicated(keep=False)]        # ver las filas duplicadas
df.drop_duplicates(inplace=True)     # eliminar duplicados

# 4. Valores raros representados con "?"
(df == "?").sum()                    # cuántos "?" hay por columna
df.replace("?", np.nan, inplace=True)   # convertir "?" en NaN

# 5. Valores faltantes
df.isnull().sum()            # nulos por columna
df.isnull().mean()*100       # porcentaje de nulos
df.dropna(inplace=True)      # eliminar filas con nulos
df.shape

# 6. Outliers (horas por semana)
sns.boxplot(x=df["hours-per-week"])
Q1 = df["hours-per-week"].quantile(0.25)   # primer cuartil
Q3 = df["hours-per-week"].quantile(0.75)   # tercer cuartil
# IQR = Q3 - Q1; outliers = fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR]

# 7. Estadística descriptiva
df.describe()               # numéricas
df.describe(include="all")  # incluye categóricas
df["age"].mean()            # media de una columna

# 8. Desbalance de clases
df["income"].value_counts()                          # conteo por clase
df["income"].value_counts(normalize=True)*100        # porcentajes
sns.countplot(data=df, x="income")                   # gráfico de barras

# 9. Análisis univariado / distribuciones
sns.histplot(df["education-num"])            # histograma
df["age"].skew()                             # asimetría
df["age"].kurt()                             # curtosis
sns.histplot(df["age"], kde=True)            # histograma + densidad
sns.boxplot(x=df["age"])                     # boxplot
```

### Concepto clave
El EDA revela **duplicados**, **valores nulos** (los `?` del dataset original) y **desbalance de clases** (la mayoría gana `<=50K`). Estos problemas deben resolverse antes de entrenar un modelo.

---

## Práctica 2.1 — Importancia de Características (Feature Importance)

### Objetivo
Responder: *¿qué variables influyen más en el precio de una casa?* usando dos técnicas de **IA Explicable (XAI)**.

### Dataset y librerías
- **Dataset:** California Housing (incluido en scikit-learn). Cada fila es un bloque censal; el objetivo es el valor medio de las casas (en cientos de miles de USD).
- **Librerías:** `fetch_california_housing`, `RandomForestRegressor`, `permutation_importance`.

### Lógica de la práctica
1. Cargar datos y separar en train/test.
2. Entrenar un `RandomForestRegressor`.
3. Calcular importancia con el **método nativo** del modelo (`feature_importances_`).
4. Calcular importancia con **permutación** (`permutation_importance`).
5. Comparar ambos resultados y graficar.

### Código técnico comentado

```python
# 1. Librerías
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt

# 2. Cargar dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)  # 8 variables
y = data.target

# Variables: MedInc, HouseAge, AveRooms, AveBedrms,
#            Population, AveOccup, Latitude, Longitude

# 3. Separar entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 4. Entrenar Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Importancia NATIVA del modelo (calculada durante el entrenamiento)
model.feature_importances_

# Convertir en Serie ordenada de mayor a menor
importance = pd.Series(model.feature_importances_,
                       index=X.columns).sort_values(ascending=False)
print(importance)

# 6. Gráfico de importancia nativa
plt.figure(figsize=(5, 3))
importance.plot(kind="bar")
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.show()

# 7. Importancia por PERMUTACIÓN (sobre el conjunto de prueba)
result = permutation_importance(model, X_test, y_test,
                                random_state=42, n_repeats=7)

# 8. Importancias medias de las permutaciones
importance = pd.Series(result.importances_mean,
                       index=X.columns).sort_values(ascending=False)
print(importance)

# 9. Gráfico de importancia por permutación
plt.figure(figsize=(5, 3))
importance.plot(kind="bar")
plt.title("Feature Importance (Permutation)")
plt.ylabel("Importance")
plt.show()
```

### Diferencia entre ambos métodos
- **`model.feature_importances_`** — se calcula *durante* el entrenamiento observando cómo cada variable ayuda a "dividir mejor" los datos dentro de los árboles. No cuesta tiempo extra.
- **`permutation_importance`** — se calcula *después*, barajando cada columna y midiendo cuánto empeora la predicción. Es más lento (por defecto repite 5 veces), pero **más confiable** porque mide el efecto real sobre las predicciones, no solo la construcción del árbol.

### Concepto clave
La variable **MedInc** (ingreso medio) suele dominar el precio de la vivienda. La permutación es "agnóstica al modelo" y sirve para cualquier estimador.

---

## Práctica 2.2 — Valores SHAP

### Objetivo
Explicar predicciones de forma **global** (todo el dataset) y **local** (una predicción individual) usando SHAP con XGBoost.

### Dataset y librerías
- **Dataset:** California Housing (igual que la práctica 2.1).
- **Librerías:** `XGBRegressor`, `shap`.

### Lógica de la práctica
1. Cargar datos y dividir train/test.
2. Entrenar `XGBRegressor`.
3. Crear un **explicador** (`shap.Explainer`).
4. Calcular **valores SHAP** sobre el test.
5. Visualizar **importancia global** (bar + beeswarm).
6. Construir una casa nueva y explicar **localmente** su predicción (waterfall).

### Código técnico comentado

```python
# 1. Librerías
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pandas as pd
import shap

# 2. Cargar dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 3. Separar train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 4. Entrenar XGBoost
model = XGBRegressor(n_estimators=100, random_state=42,
                     enable_categorical=False)
model.fit(X_train, y_train)

model.score(X_test, y_test)   # R² sobre el test

# 5. Crear el explicador SHAP
# model  -> modelo entrenado a evaluar
# X_train -> datos de referencia (background)
explainer = shap.Explainer(model, X_train)

# 6. Calcular valores SHAP (contribución de cada variable por fila)
shap_values = explainer(X_test)

# ---- Importancia GLOBAL ----
shap.plots.bar(shap_values)          # importancia media absoluta
shap.plots.beeswarm(shap_values)     # "enjambre": magnitud + dirección + valor

# ---- Importancia LOCAL ----
# 1. Crear una casa nueva (un solo registro)
new_house = pd.DataFrame({
    "MedInc": [5.0], "HouseAge": [20.0], "AveRooms": [6.0],
    "AveBedrms": [2.0], "Population": [1500.0], "AveOccup": [3.0],
    "Latitude": [34.0], "Longitude": [-118.0]
})

# 2. Predecir su precio
prediction = model.predict(new_house)
print("Predicción:", prediction)

# 3. Calcular SHAP para esa casa y explicar la predicción
new_house_shap = explainer(new_house)
shap.plots.waterfall(new_house_shap[0])
```

### Concepto clave
SHAP descompone una predicción en la suma de las contribuciones de cada variable (basado en la teoría de juegos). El gráfico **waterfall** muestra, para una predicción concreta, qué variables la empujan hacia arriba o hacia abajo respecto del valor base.

---

## Práctica 3 — Series de tiempo, tendencia y detección de cambio

### Objetivo
Generar una serie temporal sintética con un **cambio de tendencia** y comparar dos modelos de pronóstico: **regresión lineal** y **ARIMA**.

### Dataset y librerías
- **Dataset:** serie sintética construida con una pendiente (tasa de crecimiento) que pasa de `0.3` a `1.2` a mitad de la serie, más ruido aleatorio.
- **Librerías:** `numpy`, `pandas`, `matplotlib`, `ARIMA` (statsmodels), `LinearRegression`, `mean_squared_error`.

### Lógica de la práctica
1. Construir la serie: pendiente + ruido → acumulado (`cumsum`).
2. Separar en train (histórico) y test (últimos 10 valores).
3. Ajustar **ARIMA(1,1,1)** sobre el train y pronosticar 10 pasos.
4. Ajustar **regresión lineal** `Valor = b0 + b1·t` y pronosticar.
5. Comparar el **RMSE** de ambos y graficar.

### Código técnico comentado

```python
# 0. Librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Serie de ejemplo: la pendiente (tasa de crecimiento)
np.random.seed(0)                 # fija la semilla para reproducibilidad
n = 50                            # cantidad de puntos en el tiempo
# 30 puntos con pendiente 0.3 + 20 puntos con pendiente 1.2 (CAMBIO)
pendiente = np.concatenate([np.full(30, 0.3), np.full(20, 1.2)])

# Sumar ruido aleatorio (media 0, desviación 1)
incrementos = pendiente + np.random.normal(0, 1, n)

# La serie es el acumulado de los incrementos
serie = pd.Series(50 + np.cumsum(incrementos))

plt.plot(serie)
plt.title("Serie Temporal creciente")
plt.xlabel("Tiempo"); plt.ylabel("Valor")
plt.show()

# 2. Separar train (histórico) y test (últimos 10)
train = serie[:-10]   # todos excepto los últimos 10
test  = serie[-10:]   # últimos 10 valores

# 3. Modelo ARIMA(1,1,1)
modelo_arima = ARIMA(train, order=(1, 1, 1)).fit()
# AR=1 -> usa el valor del periodo anterior para predecir el actual
# I=1  -> aplica 1 diferencia (y_t - y_{t-1}) para volverla estacionaria
# MA=1 -> incorpora el error del periodo anterior para corregir

# Pronóstico de los siguientes 10 pasos
pred_arima = modelo_arima.forecast(steps=10)

# 4. Regresión lineal  (Valor = b0 + b1 * t)
# t_train: 0..39 ; t_test: 40..49  (la variable predictora es el tiempo)
t_train = np.arange(len(train)).reshape(-1, 1)
t_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

LR = LinearRegression().fit(t_train, train.values)
pred_LR = LR.predict(t_test)

# 5. Comparar error de pronóstico (RMSE)
rmse_arima = np.sqrt(mean_squared_error(test, pred_arima))
rmse_reg   = np.sqrt(mean_squared_error(test, pred_LR))
print(f"RMSE Regresión Lineal: {rmse_reg}")
print(f"RMSE ARIMA(1,1,1):     {rmse_arima}")

# 6. Graficar comparación
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Real")
plt.plot(test.index, pred_arima, label=f"ARIMA (RMSE={rmse_arima:.1f})")
plt.plot(test.index, pred_LR, label=f"Regresión lineal (RMSE={rmse_reg:.1f})")
plt.legend()
plt.title("ARIMA vs Regresión Lineal")
plt.show()
```

### Concepto clave
La regresión lineal asume una tendencia constante, por lo que **no detecta bien el cambio de pendiente**; ARIMA, al modelar la dependencia temporal, suele adaptarse mejor al cambio. El **RMSE** permite comparar numéricamente cuál pronostica mejor.
