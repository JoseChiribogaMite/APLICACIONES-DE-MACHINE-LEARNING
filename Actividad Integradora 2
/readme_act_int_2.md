# Actividad Integradora 2 — ML Regresión: IA Explicativa (XAI) y Búsqueda de Hiperparámetros

> Guía completa **punto a punto** para completar la práctica.
> Incluye: el código de cada parte listo para pegar en tu notebook (`.ipynb`),
> las explicaciones técnicas de cada concepto, cómo entender cada resultado y
> comentarios en (casi) cada línea de código.
>
> **Esta versión está alineada con las prácticas de clase del repositorio**
> (Unidad 1: Prácticas 1, 2.1, 2.2, 3 · Unidad 2: Prácticas 4, 5, 6A, 6B ·
> Actividad Integradora 1): usa la misma clase de EDA (`herramientas_eda.AutoEDA`),
> los mismos modelos (Random Forest y XGBoost), la misma API moderna de SHAP y las
> mismas librerías de optimización (Optuna `TPESampler` y `BayesSearchCV` de scikit-optimize).

---

## Cómo usar esta guía

1. Crea un notebook nuevo en Jupyter (o Google Colab) llamado `Actividad_Integradora_2.ipynb`, dentro de la carpeta `Actividad Integradora 2`.
2. Copia cada bloque de código en una **celda nueva**, en el orden en que aparecen.
3. Lee la sección *"¿Qué es y para qué sirve?"* antes de ejecutar el código de cada paso.
4. Ejecuta las celdas en orden (de arriba hacia abajo, sin saltarte ninguna).
5. Cuando termines, responde las preguntas de la sección **Conclusiones**: eso es lo que pide el enunciado cuando dice *"Explique..."*.

### Mapa: instrucción del enunciado → sección de esta guía

- **Descarga un dataset de regresión (diferente a los de las prácticas)** → Paso 1
- **Reutiliza el EDA de la Actividad Integradora 1** → Paso 2
- **Preprocesamiento y partición de datos** → Paso 3
- **Elige 2 modelos de ML** → Paso 4
- **GridSearch, Búsqueda Aleatoria y Optimización Bayesiana** → Pasos 5, 6 y 7
- **Comparar score y tiempos de ejecución** → Paso 8
- **Entrenar un nuevo modelo con los mejores hiperparámetros** → Paso 9
- **Feature Importance, Permutation Importance y SHAP (global)** → Pasos 10, 11 y 12
- **Predicción nueva + SHAP local** → Paso 13

### Alineación con las prácticas de clase

Cada parte de esta guía corresponde a una práctica que ya hiciste en el curso:

- **Paso 2 (EDA)** ← Práctica 1 (EDA) y, sobre todo, la **Actividad Integradora 1** (`herramientas_eda.py`, clase `AutoEDA`): el enunciado pide *reutilizar* ese código, no reescribirlo.
- **Pasos 10 y 11 (Feature Importance y Permutation Importance)** ← Práctica 2.1.
- **Pasos 12 y 13 (SHAP global y local)** ← Práctica 2.2 (misma API moderna: `shap.Explainer`, `shap.plots.bar/beeswarm/waterfall`).
- **Paso 5 (Grid Search)** ← Práctica 6A (comparación Grid/Random/Optuna).
- **Paso 6 (Búsqueda Aleatoria)** ← Práctica 5 (RandomizedSearchCV + regresión + RMSE).
- **Paso 7 (Optimización Bayesiana)** ← Práctica 6A (Optuna + `TPESampler`) y Práctica 6B (`BayesSearchCV` de scikit-optimize).

---

## Índice

1. Conceptos clave que debes dominar
2. Paso 0 — Preparar el entorno
3. Paso 1 — Descarga y carga del dataset (Diamonds)
4. Paso 2 — EDA automatizado (reutilizando la Actividad 1)
5. Paso 3 — Preprocesamiento y partición de datos
6. Paso 4 — Modelos base (línea base)
7. Paso 5 — Grid Search
8. Paso 6 — Búsqueda Aleatoria (Randomized Search)
9. Paso 7 — Optimización Bayesiana (Optuna y BayesSearchCV)
10. Paso 8 — Comparación de métodos: score y tiempo
11. Paso 9 — Entrenamiento del modelo final
12. Paso 10 — Feature Importance (importancia global)
13. Paso 11 — Permutation Importance (importancia global)
14. Paso 12 — SHAP Values (importancia global)
15. Paso 13 — Predicción nueva y SHAP local
16. Conclusiones (lo que debes redactar en tu informe)
17. Checklist final de entregables

---

## 1. Conceptos clave que debes dominar

Antes de programar, entiende estos términos: te pedirán que los expliques.

- **Regresión**: predecir un valor **numérico continuo** (precio, temperatura, demanda...), a diferencia de la clasificación que predice una categoría.
- **Parámetro vs hiperparámetro**:
  - *Parámetro*: lo que el modelo **aprende** de los datos (por ejemplo, los pesos de una regresión lineal).
  - *Hiperparámetro*: la "configuración" que decidimos **antes** de entrenar (número de árboles, profundidad máxima, learning rate...). La búsqueda de hiperparámetros es justamente encontrar la mejor configuración.
- **Espacio de búsqueda**: el conjunto de combinaciones posibles de hiperparámetros que vamos a probar.
- **Validación cruzada (K-Fold)**: dividir los datos de entrenamiento en K partes; se entrena K veces, cada vez usando K-1 partes para entrenar y 1 para validar. El score final es el promedio. Sirve para que el resultado no dependa de una sola partición afortunada.
- **Overfitting / underfitting**: el modelo se memoriza los datos (overfitting) o no aprende lo suficiente (underfitting). Los hiperparámetros controlan ese equilibrio.
- **Grid Search**: probar **todas** las combinaciones del espacio de búsqueda. Exhaustivo pero caro: el número de combinaciones crece multiplicativamente (explosión combinatoria).
- **Búsqueda Aleatoria (Random Search)**: muestrear combinaciones **al azar** del espacio. Según Bergstra & Bengio (2012), suele encontrar soluciones casi tan buenas como Grid en mucho menos tiempo.
- **Optimización Bayesiana**: construir un "modelo del modelo" (modelo sustituto) que aprende, según los resultados previos, **dónde** es más probable encontrar buenos hiperparámetros, y prueba ahí. Es "inteligente": no prueba a ciegas. En clase viste dos implementaciones:
  - **Optuna** con el sampler `TPESampler` (*Tree-structured Parzen Estimator*).
  - **scikit-optimize** con `BayesSearchCV`.
- **XAI (Inteligencia Artificial Explicativa)**: técnicas para entender por qué un modelo "caja negra" decide lo que decide.
  - *Importancia global*: qué features importan **en general** para el modelo.
  - *Importancia local*: qué features empujaron **una predicción concreta** hacia arriba o hacia abajo.
- **Feature Importance (MDI)**: importancia basada en cuánto reduce la impureza cada feature al dividir los nodos de los árboles. Viene "gratis" en los modelos de árboles (`feature_importances_`).
- **Permutation Importance (MDA)**: barajar (permutar) una columna y medir cuánto **empeora** el score. Si el modelo depende de esa feature, al romperla el error sube. Es agnóstica al modelo (sirve para cualquier algoritmo).
- **SHAP (SHapley Additive exPlanations)**: basado en valores de Shapley de la teoría de juegos cooperativos (Lundberg & Lee, 2017). Asigna a cada feature una "contribución" aditiva tal que:

  ```
  predicción = valor_base + contribución(feature1) + contribución(feature2) + ... + contribución(featureN)
  ```

  El **valor base** es la predicción esperada (promedio) del modelo. Es la técnica más completa: unifica las anteriores y además permite explicar predicciones individuales. Es exactamente lo que hiciste en la Práctica 2.2.

### Métricas que usaremos (regresión)

- **RMSE** (raíz del error cuadrático medio): `sqrt( media( (y_real - y_pred)² ) )`. Misma unidad que la variable objetivo. Penaliza fuerte los errores grandes. **Menor es mejor.**
- **MAE** (error absoluto medio): `media( |y_real - y_pred| )`. Interpretación directa: "en promedio me equivoco por X".
- **R²** (coeficiente de determinación): proporción de varianza explicada. 1.0 = perfecto, 0 = igual que predecir la media. **Mayor es mejor.** Es el valor que devuelve `model.score(X_test, y_test)` en regresión (lo viste en la Práctica 2.2).

---

## 2. Paso 0 — Preparar el entorno

**¿Para qué sirve?** Asegura que todas las librerías necesarias estén instaladas antes de empezar.

**Celda 0 — Instalación (ejecútala una sola vez)**

```python
# Instala (o actualiza) todas las librerías necesarias para la práctica.
# En un notebook usa %pip; si usas la terminal usa: pip install <paquete>
# %pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap optuna scikit-optimize scipy

# Opcionales:
#   statsmodels  → la clase AutoEDA de la Actividad 1 la usa para calcular el VIF (multicolinealidad)
#   plotly       → requerida por optuna.visualization (historia de optimización e importancia de hiperparámetros)
# %pip install statsmodels plotly

# Versiones recomendadas: Python 3.10+, scikit-learn >= 1.3, xgboost >= 2.0,
# shap >= 0.45, optuna >= 3.0, scikit-optimize >= 0.9
```

**Celda 1 — Imports generales**

```python
# Descomenta la siguiente línea si usas un notebook clásico (no es necesaria en Colab)
# %matplotlib inline

import numpy as np                          # Cálculo numérico (arreglos, medias, raíz cuadrada)
import pandas as pd                         # Manejo de datos tabulares (DataFrames)
import matplotlib.pyplot as plt             # Gráficas básicas
import seaborn as sns                       # Gráficas estadísticas (además provee datasets de ejemplo)
import time                                 # Medir tiempos de ejecución de cada búsqueda
from scipy.stats import randint, uniform    # Distribuciones para la búsqueda aleatoria

# Opcional: silencia warnings para que la salida del notebook quede más limpia
import warnings                             # Módulo para controlar mensajes de aviso
warnings.filterwarnings('ignore')           # Oculta los warnings que no son errores
```

---

## 3. Paso 1 — Descarga y carga del dataset (Diamonds)

**¿Qué pide el enunciado?** *"Descarga un dataset de libre elección enfocado en un problema de Regresión... Elige un dataset **diferente** a los de las prácticas de los repos."*

### ¿Qué es y para qué sirve?

El dataset es la materia prima: un conjunto de ejemplos con **features** (variables de entrada, `X`) y una **variable objetivo continua** (lo que queremos predecir, `y`).

### ⚠️ Regla del enunciado: NO usar los datasets de las prácticas

En el repositorio del curso ya se usaron estos datasets; **evítalos** para cumplir el enunciado:

- **Unidad 1**: Adult Income (Práctica 1), **California Housing** (Prácticas 2.1 y 2.2), serie sintética (Práctica 3).
- **Unidad 2**: datos sintéticos 1D (Práctica 4), **Diabetes** (Práctica 5), **Breast Cancer** (Prácticas 6A y 6B).
- **Actividad Integradora 1**: **insurance.csv** (gastos médicos).

### Dataset elegido para esta guía: Diamonds

El dataset **Diamonds** (un clásico de Kaggle: https://www.kaggle.com/datasets/shivam2503/diamonds) es perfecto para esta actividad:

- **Objetivo de regresión**: `price` (precio del diamante en USD).
- **53.940 filas** y **10 columnas**: 6 numéricas (`carat`, `depth`, `table`, `x`, `y`, `z`) y **3 categóricas** (`cut`, `color`, `clarity`) → demuestra el preprocesamiento completo (incluido OneHotEncoder).
- NO se usó en ninguna práctica del curso (solo se usaron Adult, California Housing, Diabetes, Breast Cancer, insurance y series sintéticas).
- Es el dataset con el que se comparan los modelos de árboles en la literatura, ideal para XAI.

Otras opciones válidas (tampoco usadas en clase):

- **Kaggle**: House Prices (https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques), Student Performance (https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression).
- **UCI**: Wine Quality (https://archive.ics.uci.edu/dataset/186/wine+quality), Bike Sharing (https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), Concrete Compressive Strength (https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), Auto MPG (https://archive.ics.uci.edu/dataset/9/auto+mpg), Combined Cycle Power Plant (https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant).

**Celda 2 — Carga del dataset Diamonds**

```python
# Opción A (recomendada): seaborn incluye el dataset Diamonds y lo cachea localmente.
# La primera ejecución requiere internet; después queda guardado en disco.
df = sns.load_dataset('diamonds')

# Opción B (equivalente, si prefieres cumplir la parte de "descarga" literalmente):
#   1. Descarga diamonds.csv desde https://www.kaggle.com/datasets/shivam2503/diamonds
#   2. Súbelo a tu notebook y carga:
# df = pd.read_csv('diamonds.csv')   # lee el CSV local y construye el DataFrame
```

**Celda 3 — Primer vistazo a los datos**

```python
# Muestra las primeras 5 filas para inspeccionar las columnas y sus valores
df.head()

# Dimensiones del dataset: (número de filas, número de columnas) → esperado (53940, 10)
df.shape

# Tipos de dato de cada columna y cuántos valores no nulos tiene cada una
df.info()
```

### ¿Cómo interpretar la salida?

- **Features numéricas**: `carat` (peso en quilates), `depth` (%) y `table` (%) son proporciones del tallado; `x`, `y`, `z` son las dimensiones en mm (largo, ancho, alto).
- **Features categóricas**: `cut` (calidad del corte: Fair, Good, Very Good, Premium, Ideal), `color` (escala D a J, de incoloro a amarillento), `clarity` (pureza: I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF).
- **Objetivo**: `price` en dólares (326 a 18.823, media ≈ 3.933, mediana ≈ 2.401 → distribución sesgada a la derecha; el EDA lo confirmará).
- Verifica que no haya columnas inútiles y que los tipos de dato sean los esperados (`cut`, `color`, `clarity` deben ser `object` o `category`).

---

## 4. Paso 2 — EDA automatizado (reutilizando la Actividad 1)

**¿Qué pide el enunciado?** *"Importa y reutiliza la clase/módulo/código de EDA automatizado desarrollado en la Actividad Integradora 1 para generar el diagnóstico inicial de calidad de datos, distribuciones, etc."*

### ¿Qué es y para qué sirve?

EDA = **Análisis Exploratorio de Datos**. Es el "chequeo médico" del dataset antes de entrenar nada: ¿hay valores faltantes?, ¿duplicados?, ¿outliers?, ¿cómo se distribuye cada variable?, ¿qué features se relacionan con el objetivo? Sin este diagnóstico, los modelos pueden fallar por razones que no entenderías.

### Reutilizar tu herramienta de la Actividad 1

En la Actividad 1 construiste la clase **`AutoEDA`** del módulo **`herramientas_eda.py`**, que ejecuta los **10 puntos de EDA de clase** con una sola llamada:

- `AutoEDA(df, target_col=None, figsize=(10, 6))` → constructor (recibe el DataFrame y la variable objetivo).
- `run_analysis()` → ejecuta los 10 puntos: conocer el dataset, tipo de variable, calidad de datos, valores faltantes, outliers (IQR de Tukey + boxplots), estadística descriptiva, análisis del target, análisis univariado y bivariado/multivariado (correlación de Pearson + VIF).
- `obtener_hallazgos()` → devuelve un DataFrame con la **bitácora de observaciones** 🔎 (`punto`, `tipo`, `detalle`): el insumo directo para tu interpretación.

> Principio metodológico de tu herramienta (no lo olvides al defender la práctica): **solo diagnostica y reporta**; nunca elimina filas, imputa nulos ni transforma variables automáticamente. Cada acción de limpieza es una decisión humana documentada.

**Celda 4 — Importar la clase AutoEDA de la Actividad 1**

```python
# Agregamos al PATH la carpeta hermana donde vive el módulo de la Actividad 1
import sys                                              # módulo para manipular rutas de búsqueda de imports
sys.path.append('../Actividad Integradora 1')           # ruta relativa a la carpeta de la Actividad 1 (el notebook vive en 'Actividad Integradora 2')

# Importamos la MISMA clase que construiste en la Actividad 1 (reutilización, no reescritura)
from herramientas_eda import AutoEDA                    # módulo herramientas_eda.py -> clase AutoEDA

# OJO si trabajas en Colab: sube 'herramientas_eda.py' a la sesión y usa simplemente:
# from herramientas_eda import AutoEDA
```

**Celda 5 — Ejecutar el EDA automatizado sobre Diamonds**

```python
# Instanciamos la herramienta declarando el target continuo del problema de regresión
eda = AutoEDA(df, target_col='price', figsize=(12, 6))   # target_col='price' habilita los análisis del target

# Ejecutamos los 10 puntos de EDA de clase con UNA sola llamada (automatización 100 %)
eda.run_analysis()

# Recuperamos la bitácora de observaciones: insumo directo de la sección de interpretación
hallazgos = eda.obtener_hallazgos()                      # DataFrame con columnas: punto, tipo, detalle
display(hallazgos)                                       # tabla interactiva de hallazgos 🔎 (en script: print(hallazgos))
```

### ¿Cómo interpretar la salida?

- **Faltantes y duplicados**: Diamonds no tiene valores nulos ni duplicados; tu herramienta lo confirmará con ✓.
- **Outliers**: `price`, `carat` y las dimensiones presentan valores atípicos altos (diamantes caros/grandes). NO los elimines: son datos reales y los árboles los manejan bien. Si los eliminaras, dilo explícitamente y justifícalo.
- **Sesgo**: `price` está sesgada a la derecha (muchos diamantes baratos, pocos carísimos). Opción avanzada (no obligatoria): modelar `log(price)`. Con árboles no es necesario.
- **Multicolinealidad (VIF)**: tu `AutoEDA` **va a marcar** 🔎 a `carat`, `x`, `y` y `z` (correlaciones > 0.95 entre sí: el peso y las dimensiones miden casi lo mismo). Decisión documentada: **conservamos todas** porque los modelos de árboles son robustos a la multicolinealidad y porque para XAI es más interesante ver cómo se reparten la importancia. Si tu modelo final fuera una regresión lineal, aquí sí convendría eliminar `x`, `y`, `z`.
- **Relación con el objetivo**: `carat` es la feature más correlacionada con `price` (~0.92): anticipa que dominará las importancias en los Pasos 10-12.

---

## 5. Paso 3 — Preprocesamiento y partición de datos

**¿Qué pide el enunciado?** *"Preprocesamiento y Partición de Datos."*

### ¿Qué es y para qué sirve?

- **Preprocesamiento**: dejar los datos en un formato que los algoritmos puedan usar: rellenar faltantes, escalar, codificar categóricas.
- **Partición**: separar los datos en **entrenamiento** (para aprender) y **prueba** (para evaluar como si fueran datos nuevos). La regla de oro: el conjunto de prueba **no se toca** hasta la evaluación final.

### Concepto crítico: fuga de datos (data leakage)

Si imputas/escalas/codificas usando estadísticas de TODO el dataset (incluido el test) antes de partir, estás "filtrando" información del test al entrenamiento. La solución es meter el preprocesamiento **dentro de un Pipeline** y solo hacer `.fit()` sobre el conjunto de entrenamiento. Así, dentro de cada fold de validación cruzada, el preprocesamiento se calcula únicamente con el fold de entrenamiento.

**Celda 6 — Separar features y objetivo, y partir los datos**

```python
# Herramientas de sklearn para partición, validación cruzada y pipelines
from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Separamos features (X) de la variable objetivo (y)
X = df.drop(columns=['price'])         # X = todas las columnas MENOS la objetivo
y = df['price']                        # y = solo la columna objetivo (precio en USD)

# Partición 80% entrenamiento / 20% prueba
# test_size=0.2 -> 20% para test; random_state=42 -> partición reproducible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Objeto de validación cruzada ÚNICO para todas las búsquedas
# Usar los MISMOS folds en Grid/Random/Bayesiana hace la comparación justa
cv = KFold(n_splits=5, shuffle=True, random_state=42)   # 5 folds, barajados, reproducibles
```

**Celda 7 — Pipelines de preprocesamiento**

```python
# Pipeline para columnas NUMÉRICAS: rellenar faltantes y estandarizar
pipeline_numerico = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),   # rellena NaN con la MEDIANA de cada columna (robusta a outliers)
    ('scaler', StandardScaler())                     # estandariza: resta la media y divide por la desviación (media=0, std=1)
])

# Pipeline para columnas CATEGÓRICAS: rellenar faltantes y codificar
pipeline_categorico = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),   # rellena NaN con la MODA (el valor más frecuente)
    ('encoder', OneHotEncoder(handle_unknown='ignore'))     # convierte categorías en columnas 0/1 (sin orden artificial)
])

# ColumnTransformer: aplica cada pipeline al tipo de columna que le corresponde
preprocesador = ColumnTransformer(transformers=[
    ('num', pipeline_numerico, make_column_selector(dtype_include=np.number)),   # numéricas -> pipeline_numérico
    ('cat', pipeline_categorico, make_column_selector(dtype_exclude=np.number)), # categóricas (cut, color, clarity) -> one-hot
])
```

### ¿Cómo interpretar esta parte?

- `SimpleImputer(median)`: la mediana es robusta a outliers; la media se distorsiona con valores extremos.
- `StandardScaler`: imprescindible para modelos sensibles a la escala (regresión lineal, SVM, redes). Para árboles no es necesario, pero **no les hace daño** y nos permite mantener un pipeline único.
- `OneHotEncoder`: convierte `cut` (5 categorías), `color` (7) y `clarity` (8) en 20 columnas binarias. Evita el error de asignarles un "orden" artificial (problema de la codificación ordinal/LabelEncoder).
- `make_column_selector`: selecciona columnas automáticamente por su tipo, así el mismo código funciona con cualquier dataset.

---

## 6. Paso 4 — Modelos base (línea base)

**¿Qué pide el enunciado?** *"Elige 2 modelos de ML..."*

### ¿Qué es y para qué sirve?

Antes de optimizar hiperparámetros, entrenamos los 2 modelos **con sus valores por defecto** para tener una referencia ("línea base"): si después de horas de búsqueda no mejoras la línea base, algo anda mal. También comparamos contra un modelo "tonto" que predice siempre la media: si tu modelo no supera eso, no aprendió nada.

**Modelos elegidos (los mismos del curso):**

- **Random Forest** (Práctica 2.1 y Práctica 5): muchos árboles de decisión entrenados sobre muestras aleatorias (bagging). Robusto, poco sensible a hiperparámetros, difícil de sobreajustar.
- **XGBoost** (Práctica 2.2 y Práctica 6A): boosting extremo por gradiente; árboles en secuencia que corrigen los errores de los anteriores, con regularización incorporada. Suele lograr el mejor score, pero es más sensible a hiperparámetros.

Ambos ofrecen `feature_importances_` nativo, lo que facilita la sección de XAI.

**Celda 8 — Línea base con hiperparámetros por defecto**

```python
# Modelos y métricas
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyRegressor

# Referencia "tonta": siempre predice la media del objetivo
dummy = DummyRegressor(strategy='mean')
scores_dummy = cross_val_score(dummy, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')  # MSE negativo en CV
print(f"Dummy (predice la media): RMSE CV = {np.sqrt(-scores_dummy.mean()):.0f} USD")

# Nuestros 2 modelos base, con hiperparámetros por defecto
# enable_categorical=False: como ya codificamos las categóricas con OneHot, XGBoost recibe solo números
# (si tu versión de xgboost es < 2.0, quita ese argumento)
modelos_base = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, enable_categorical=False),
}

# Evaluamos cada modelo base envuelto en el pipeline de preprocesamiento
for nombre, modelo in modelos_base.items():                        # iteramos sobre los 2 modelos
    pipe = Pipeline(steps=[
        ('preprocesador', preprocesador),                          # mismo preprocesamiento para ambos
        ('modelo', modelo)                                         # el modelo base del turno
    ])
    # Validación cruzada con MSE negativo; sqrt(-score) lo convierte en RMSE
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    print(f"{nombre}: RMSE CV = {np.sqrt(-scores.mean()):.0f} USD")
```

### ¿Cómo interpretar la salida?

- El RMSE del Dummy (~3.990 USD ≈ desviación estándar de `price`) es la "barrera mínima": si el modelo no baja de ahí, no aprende nada.
- Ambos modelos base deberían bajar claramente del Dummy (típicamente RMSE ≈ 540-570 USD en Diamonds).
- Anota estos números: los usarás en las conclusiones para mostrar cuánto mejoró la búsqueda de hiperparámetros.

### Glosario de hiperparámetros que optimizaremos

- **n_estimators**: número de árboles. Más árboles = más precisión, pero más tiempo.
- **max_depth**: profundidad máxima de cada árbol. Muy profundo = memoriza (overfitting); muy bajo = no aprende (underfitting).
- **min_samples_split** / **min_samples_leaf**: mínimo de muestras para dividir un nodo / quedar en una hoja. Valores altos regularizan.
- **max_features**: cuántas features considera cada árbol en cada división (`sqrt`, `log2` o todas). Añade diversidad entre árboles.
- **learning_rate** (boosting): cuánto aporta cada árbol nuevo. Bajo = aprende lento pero generaliza mejor (requiere más árboles).
- **subsample** (boosting): fracción de filas usada por cada árbol. Menor a 1 reduce overfitting.
- **min_child_weight** (XGBoost): suma mínima de pesos de instancias en una hoja; valores altos regularizan (equivale a `min_samples_leaf`).
- **colsample_bytree** (XGBoost): fracción de columnas que usa cada árbol (equivalente a `max_features`).

---

## 7. Paso 5 — Grid Search

**¿Qué pide el enunciado?** *"...encuentra los mejores hiperparámetros usando GridSearch..."*

### ¿Qué es y para qué sirve?

Grid Search **prueba todas las combinaciones** de los valores que le des. Es la forma más sistemática, pero sufre la **explosión combinatoria**: 3 valores × 3 valores × 4 valores = 36 combinaciones × 5 folds = 180 entrenamientos. Por eso los grids de demostración deben ser pequeños.

> Importante: como el modelo vive dentro de un Pipeline, los nombres de los parámetros llevan el prefijo `modelo__` (así sklearn sabe que `n_estimators` es del paso llamado `modelo`).

**Celda 9 — Grid Search para Random Forest**

```python
# Importamos el buscador por rejilla
from sklearn.model_selection import GridSearchCV

# Inicializamos la lista que acumulará los resultados de TODAS las búsquedas
# (la usaremos en el Paso 8 para comparar métodos y tiempos)
resumen = []

# Pipeline del modelo (el preprocesador es el mismo objeto de siempre: sklearn lo clona)
pipe_rf = Pipeline(steps=[
    ('preprocesador', preprocesador),
    ('modelo', RandomForestRegressor(random_state=42))
])

# Espacio de búsqueda: 2 x 3 x 2 = 12 combinaciones (chico para que corra rápido)
param_grid_rf = {
    'modelo__n_estimators': [100, 300],        # cuántos árboles probar
    'modelo__max_depth': [10, 20, None],       # None = sin límite de profundidad
    'modelo__min_samples_split': [2, 10],      # mínimo de muestras para dividir un nodo
}

# Configuramos el GridSearch: evalúa cada combinación con validación cruzada de 5 folds
grid_rf = GridSearchCV(
    estimator=pipe_rf,                          # pipeline que contiene el modelo
    param_grid=param_grid_rf,                   # las 12 combinaciones a probar
    scoring='neg_mean_squared_error',           # métrica interna (MSE negativo; menor es mejor)
    cv=cv,                                      # mismos folds que usaremos en todos los métodos
    n_jobs=-1                                   # usa todos los núcleos del CPU (paraleliza)
)

# Medimos el tiempo de ejecución de la búsqueda
inicio = time.perf_counter()                    # marca de tiempo inicial
grid_rf.fit(X_train, y_train)                   # ejecuta las 12 x 5 = 60 combinaciones de entrenamiento
tiempo_grid_rf = time.perf_counter() - inicio   # tiempo total transcurrido

# Evaluamos el mejor modelo encontrado sobre el conjunto de PRUEBA (que nunca vio)
y_pred_test = grid_rf.predict(X_test)                          # predicciones del mejor estimador
rmse_test_grid_rf = np.sqrt(mean_squared_error(y_test, y_pred_test))  # RMSE real en test

# Reportamos resultados del experimento
print('Mejores hiperparámetros:', grid_rf.best_params_)         # combinación ganadora
print(f"RMSE CV: {np.sqrt(-grid_rf.best_score_):.0f} | RMSE Test: {rmse_test_grid_rf:.0f} | Tiempo: {tiempo_grid_rf:.2f} s")

# Registramos el experimento en la tabla comparativa (Paso 8)
resumen.append({
    'Metodo': 'Grid Search', 'Modelo': 'Random Forest',
    'RMSE CV': np.sqrt(-grid_rf.best_score_),                   # score de validación cruzada
    'RMSE Test': rmse_test_grid_rf,                             # score real sobre test
    'Tiempo (s)': tiempo_grid_rf                                # tiempo de ejecución
})
```

**Celda 10 — Grid Search para XGBoost**

```python
# Pipeline del segundo modelo (XGBoost, el modelo de las prácticas 2.2 y 6A)
pipe_xgb = Pipeline(steps=[
    ('preprocesador', preprocesador),
    ('modelo', XGBRegressor(random_state=42, enable_categorical=False))
])

# Espacio de búsqueda: 2 x 2 x 2 = 8 combinaciones
param_grid_xgb = {
    'modelo__n_estimators': [100, 300],       # cuántos árboles en la secuencia
    'modelo__learning_rate': [0.05, 0.1],     # cuánto aporta cada árbol (paso de aprendizaje)
    'modelo__max_depth': [3, 5],              # profundidad de cada árbol
}

# Configuramos la búsqueda por rejilla (misma métrica, mismos folds)
grid_xgb = GridSearchCV(estimator=pipe_xgb, param_grid=param_grid_xgb,
                        scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

# Medimos tiempo y ejecutamos
inicio = time.perf_counter()
grid_xgb.fit(X_train, y_train)
tiempo_grid_xgb = time.perf_counter() - inicio

# Evaluación en test del mejor modelo encontrado
y_pred_test = grid_xgb.predict(X_test)
rmse_test_grid_xgb = np.sqrt(mean_squared_error(y_test, y_pred_test))

print('Mejores hiperparámetros:', grid_xgb.best_params_)
print(f"RMSE CV: {np.sqrt(-grid_xgb.best_score_):.0f} | RMSE Test: {rmse_test_grid_xgb:.0f} | Tiempo: {tiempo_grid_xgb:.2f} s")

# Registramos el experimento
resumen.append({
    'Metodo': 'Grid Search', 'Modelo': 'XGBoost',
    'RMSE CV': np.sqrt(-grid_xgb.best_score_),
    'RMSE Test': rmse_test_grid_xgb,
    'Tiempo (s)': tiempo_grid_xgb
})
```

### ¿Cómo interpretar la salida?

- `best_params_` te dice la combinación ganadora. Si el mejor valor quedó en el **borde** del rango (por ejemplo `n_estimators=300` siendo el máximo probado), sospecha que el óptimo podría estar fuera del rango: amplía el grid.
- `best_score_` es el RMSE de validación cruzada (promedio de los 5 folds del mejor modelo).
- El RMSE en test es la evaluación honesta: el modelo nunca vio esos datos.
- **Limitación de Grid**: solo explora los puntos exactos que le diste. Si el óptimo está "entre" dos valores, Grid no lo encuentra. Por eso existen los siguientes métodos.

---

## 8. Paso 6 — Búsqueda Aleatoria (Randomized Search)

**¿Qué pide el enunciado?** *"...Búsqueda Aleatoria..."*

### ¿Qué es y para qué sirve?

En vez de probar todas las combinaciones, **muestrea N combinaciones al azar** del espacio. Ventajas:

- Puedes definir **distribuciones** (rangos continuos), no solo listas de valores.
- Con el mismo presupuesto de tiempo explora zonas más variadas que Grid.
- Bergstra & Bengio (2012) demostraron que suele alcanzar resultados casi óptimos con muchos menos intentos, porque normalmente solo unos pocos hiperparámetros realmente importan.
- Es exactamente el método de la **Práctica 5** (RandomizedSearchCV + RandomForestRegressor + `neg_mean_squared_error`).

**Celda 11 — Búsqueda aleatoria para Random Forest**

```python
# Importamos el buscador aleatorio
from sklearn.model_selection import RandomizedSearchCV

# Espacio de búsqueda con DISTRIBUCIONES (rangos), no listas fijas
param_dist_rf = {
    'modelo__n_estimators': randint(100, 501),        # entero uniforme entre 100 y 500
    'modelo__max_depth': randint(5, 31),              # entero uniforme entre 5 y 30
    'modelo__min_samples_split': randint(2, 21),      # entero uniforme entre 2 y 20
    'modelo__min_samples_leaf': randint(1, 11),       # entero uniforme entre 1 y 10
    'modelo__max_features': ['sqrt', 'log2', 1.0],    # lista: se elige un valor al azar (1.0 = todas)
}

# Buscador aleatorio: 30 combinaciones al azar, 5 folds cada una
random_rf = RandomizedSearchCV(
    estimator=pipe_rf,                        # reutilizamos el mismo pipeline del Grid
    param_distributions=param_dist_rf,        # espacio definido con distribuciones
    n_iter=30,                                # número de combinaciones a muestrear
    scoring='neg_mean_squared_error',         # misma métrica que Grid (comparación justa)
    cv=cv,                                    # mismos folds
    n_jobs=-1,                                # paralelizar
    random_state=42                           # muestreo reproducible
)

# Medimos tiempo y ejecutamos
inicio = time.perf_counter()
random_rf.fit(X_train, y_train)
tiempo_random_rf = time.perf_counter() - inicio

# Evaluación en test
y_pred_test = random_rf.predict(X_test)
rmse_test_random_rf = np.sqrt(mean_squared_error(y_test, y_pred_test))

print('Mejores hiperparámetros:', random_rf.best_params_)
print(f"RMSE CV: {np.sqrt(-random_rf.best_score_):.0f} | RMSE Test: {rmse_test_random_rf:.0f} | Tiempo: {tiempo_random_rf:.2f} s")

# Registramos el experimento
resumen.append({
    'Metodo': 'Aleatoria', 'Modelo': 'Random Forest',
    'RMSE CV': np.sqrt(-random_rf.best_score_),
    'RMSE Test': rmse_test_random_rf,
    'Tiempo (s)': tiempo_random_rf
})
```

**Celda 12 — Búsqueda aleatoria para XGBoost**

```python
# Espacio con distribuciones para XGBoost
param_dist_xgb = {
    'modelo__n_estimators': randint(100, 401),            # entero uniforme entre 100 y 400
    'modelo__learning_rate': uniform(0.01, 0.2),          # uniforme CONTINUA entre 0.01 y 0.21
    'modelo__max_depth': randint(3, 8),                   # entero uniforme entre 3 y 7
    'modelo__subsample': uniform(0.6, 0.4),               # uniforme continua entre 0.6 y 1.0
    'modelo__min_child_weight': randint(1, 11),           # entero uniforme entre 1 y 10 (regularización)
}

# Buscador aleatorio: 30 combinaciones al azar
random_xgb = RandomizedSearchCV(
    estimator=pipe_xgb,
    param_distributions=param_dist_xgb,
    n_iter=30,
    scoring='neg_mean_squared_error',
    cv=cv,
    n_jobs=-1,
    random_state=42
)

# Medimos tiempo y ejecutamos
inicio = time.perf_counter()
random_xgb.fit(X_train, y_train)
tiempo_random_xgb = time.perf_counter() - inicio

# Evaluación en test
y_pred_test = random_xgb.predict(X_test)
rmse_test_random_xgb = np.sqrt(mean_squared_error(y_test, y_pred_test))

print('Mejores hiperparámetros:', random_xgb.best_params_)
print(f"RMSE CV: {np.sqrt(-random_xgb.best_score_):.0f} | RMSE Test: {rmse_test_random_xgb:.0f} | Tiempo: {tiempo_random_xgb:.2f} s")

# Registramos el experimento
resumen.append({
    'Metodo': 'Aleatoria', 'Modelo': 'XGBoost',
    'RMSE CV': np.sqrt(-random_xgb.best_score_),
    'RMSE Test': rmse_test_random_xgb,
    'Tiempo (s)': tiempo_random_xgb
})
```

### ¿Cómo interpretar la salida?

- `uniform(loc, scale)` de scipy muestrea en el intervalo `[loc, loc + scale)`. Por eso `uniform(0.01, 0.2)` va de 0.01 a 0.21.
- Compara el RMSE obtenido con el de Grid: con una cantidad similar de entrenamientos (o menos), la búsqueda aleatoria suele empatar o superar a Grid, porque explora el espacio de forma más eficiente.
- El resultado es aleatorio (controlado por `random_state`): con más `n_iter` tiende a mejores resultados, pero con rendimientos decrecientes.

---

## 9. Paso 7 — Optimización Bayesiana (Optuna y BayesSearchCV)

**¿Qué pide el enunciado?** *"...y Optimización Bayesiana."*

### ¿Qué es y para qué sirve?

A diferencia de Grid y Random (que prueban "a ciegas"), la optimización bayesiana **aprende de cada intento**:

1. Evalúa una combinación.
2. Un **modelo sustituto** (en Optuna, `TPESampler`; en scikit-optimize, procesos gaussianos) estima qué zonas del espacio son prometedoras según los resultados anteriores.
3. Una **función de adquisición** decide el siguiente punto a probar (equilibrando explotar buenas zonas y explorar zonas desconocidas).
4. Repite.

Resultado: converge más rápido a buenos hiperparámetros, especialmente con espacios grandes, y prueba valores **continuos** con total libertad (p. ej. `learning_rate=0.073`, imposible en un Grid discreto).

> En clase viste las **dos implementaciones**: Práctica 6A con **Optuna** (`TPESampler`) y Práctica 6B con **`BayesSearchCV`** (scikit-optimize). Abajo están ambas, aplicadas a regresión.
>
> **Diferencia metodológica con la Práctica 6A**: allí la función objetivo evaluaba con `.score(X_test, y_test)`; aquí evaluamos con **validación cruzada sobre el entrenamiento** para que el conjunto de test NO participe en la selección de hiperparámetros (evita fuga de datos). Es la misma idea, con una práctica más rigurosa.

### 9.1 Optuna (TPESampler) — como la Práctica 6A

**Celda 13 — Función objetivo para Random Forest (Optuna)**

```python
# Importamos Optuna (librería líder de optimización bayesiana de hiperparámetros)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)   # silenciamos los mensajes de progreso de cada trial

# Función objetivo: Optuna la llamará una vez por "trial" (intento)
# Cada trial SUGIERE una combinación de hiperparámetros; nosotros devolvemos el score a minimizar
def objetivo_rf(trial):
    # El trial sugiere valores dentro de los rangos definidos
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),            # entero en [100, 500]
        'max_depth': trial.suggest_int('max_depth', 5, 30),                      # entero en [5, 30]
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),      # entero en [2, 20]
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),        # entero en [1, 10]
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 1.0]),  # entre opciones
    }
    modelo = RandomForestRegressor(**params, random_state=42, n_jobs=-1)  # modelo con la combinación sugerida
    pipe = Pipeline(steps=[('preprocesador', preprocesador), ('modelo', modelo)])  # pipeline completo
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')  # CV con los MISMOS folds
    return float(np.sqrt(-scores.mean()))   # devolvemos el RMSE: Optuna lo MINIMIZARÁ

# Creamos el estudio: direction='minimize' porque RMSE menor es mejor
# TPESampler es el modelo sustituto (Tree-structured Parzen Estimator), el mismo de la Práctica 6A
estudio_rf = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))

# Ejecutamos 30 trials (intentos inteligentes)
inicio = time.perf_counter()
estudio_rf.optimize(objetivo_rf, n_trials=30)
tiempo_bayes_rf = time.perf_counter() - inicio

print('Mejores hiperparámetros:', estudio_rf.best_params)     # mejor combinación encontrada
print(f"RMSE CV: {estudio_rf.best_value:.0f} | Tiempo: {tiempo_bayes_rf:.2f} s")

# Optuna no devuelve el modelo ya entrenado: lo reconstruimos y entrenamos a mano
# (igual que la Práctica 6A: best_model = Model(**best_params))
mejor_rf_bayes = RandomForestRegressor(**estudio_rf.best_params, random_state=42, n_jobs=-1)
pipe_mejor_rf_bayes = Pipeline(steps=[('preprocesador', preprocesador), ('modelo', mejor_rf_bayes)])
pipe_mejor_rf_bayes.fit(X_train, y_train)                     # entrenamos con la combinación ganadora

# Evaluación en test
y_pred_test = pipe_mejor_rf_bayes.predict(X_test)
rmse_test_bayes_rf = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"RMSE Test: {rmse_test_bayes_rf:.0f}")

# Registramos el experimento
resumen.append({
    'Metodo': 'Bayesiana', 'Modelo': 'Random Forest',
    'RMSE CV': estudio_rf.best_value,
    'RMSE Test': rmse_test_bayes_rf,
    'Tiempo (s)': tiempo_bayes_rf
})
```

**Celda 14 — Función objetivo para XGBoost (Optuna)**

```python
# Función objetivo para XGBoost (misma lógica que la anterior)
def objetivo_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),                 # entero en [100, 400]
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),  # CONTINUO log-uniforme en [0.01, 0.2]
        'max_depth': trial.suggest_int('max_depth', 3, 8),                           # entero en [3, 8]
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),                     # continuo en [0.6, 1.0]
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),            # entero en [1, 10] (regularización)
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),       # fracción de columnas por árbol
    }
    modelo = XGBRegressor(**params, random_state=42, enable_categorical=False)       # modelo con la combinación sugerida
    pipe = Pipeline(steps=[('preprocesador', preprocesador), ('modelo', modelo)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    return float(np.sqrt(-scores.mean()))                                            # RMSE a minimizar

# Estudio para XGBoost
estudio_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))

# Ejecutamos 30 trials
inicio = time.perf_counter()
estudio_xgb.optimize(objetivo_xgb, n_trials=30)
tiempo_bayes_xgb = time.perf_counter() - inicio

print('Mejores hiperparámetros:', estudio_xgb.best_params)
print(f"RMSE CV: {estudio_xgb.best_value:.0f} | Tiempo: {tiempo_bayes_xgb:.2f} s")

# Reconstruimos y entrenamos el mejor modelo encontrado
mejor_xgb_bayes = XGBRegressor(**estudio_xgb.best_params, random_state=42, enable_categorical=False)
pipe_mejor_xgb_bayes = Pipeline(steps=[('preprocesador', preprocesador), ('modelo', mejor_xgb_bayes)])
pipe_mejor_xgb_bayes.fit(X_train, y_train)

# Evaluación en test
y_pred_test = pipe_mejor_xgb_bayes.predict(X_test)
rmse_test_bayes_xgb = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"RMSE Test: {rmse_test_bayes_xgb:.0f}")

# Registramos el experimento
resumen.append({
    'Metodo': 'Bayesiana', 'Modelo': 'XGBoost',
    'RMSE CV': estudio_xgb.best_value,
    'RMSE Test': rmse_test_bayes_xgb,
    'Tiempo (s)': tiempo_bayes_xgb
})
```

**Celda 15 — Visualizaciones de Optuna (como la Práctica 6A)**

```python
# Evolución del mejor RMSE trial a trial (muestra que la búsqueda APRENDE y mejora)
# (requiere plotly: %pip install plotly)
optuna.visualization.plot_optimization_history(estudio_xgb)

# Importancia de cada hiperparámetro en el resultado:
# útil para saber en qué hiperparámetro vale la pena invertir más tiempo refinando
optuna.visualization.plot_param_importances(estudio_xgb)
```

### 9.2 Alternativa de clase: BayesSearchCV (scikit-optimize) — Práctica 6B

La Práctica 6B usó `BayesSearchCV` de **scikit-optimize**, con espacios `Integer`, `Real` y `Categorical`. Inclúyela como segunda evidencia de optimización bayesiana (o úsala en lugar de Optuna si lo prefieres):

**Celda 16 — Búsqueda bayesiana con BayesSearchCV (opcional, Práctica 6B)**

```python
# Implementación de optimización bayesiana de la Práctica 6B (scikit-optimize)
# %pip install scikit-optimize
from skopt import BayesSearchCV                 # buscador bayesiano con API estilo sklearn
from skopt.space import Real, Integer, Categorical   # tipos de espacio: continuo, entero y categórico

# Espacio de búsqueda con rangos guiados (no puramente al azar, como en Randomized)
espacio_xgb = {
    'modelo__n_estimators': Integer(100, 400),          # entero entre 100 y 400
    'modelo__learning_rate': Real(0.01, 0.2, prior='log-uniform'),  # continuo en escala logarítmica
    'modelo__max_depth': Integer(3, 8),                 # entero entre 3 y 8
    'modelo__subsample': Real(0.6, 1.0, prior='uniform'),         # continuo uniforme
    'modelo__min_child_weight': Integer(1, 10),         # entero entre 1 y 10
}

# Configuramos la búsqueda bayesiana (API idéntica a Grid/RandomizedSearchCV)
bayes_xgb = BayesSearchCV(
    estimator=pipe_xgb,                # pipeline del modelo
    search_spaces=espacio_xgb,         # espacio de búsqueda bayesiano
    n_iter=30,                         # número de evaluaciones guiadas
    scoring='neg_mean_squared_error',  # misma métrica de comparación
    cv=cv,                             # mismos folds
    n_jobs=-1,
    random_state=42
)

# Medimos tiempo y ejecutamos
inicio = time.perf_counter()
bayes_xgb.fit(X_train, y_train)
tiempo_bayes_skopt = time.perf_counter() - inicio

# Evaluación en test
y_pred_test = bayes_xgb.predict(X_test)
rmse_test_bayes_skopt = np.sqrt(mean_squared_error(y_test, y_pred_test))
print('Mejores hiperparámetros:', bayes_xgb.best_params_)
print(f"RMSE CV: {np.sqrt(-bayes_xgb.best_score_):.0f} | RMSE Test: {rmse_test_bayes_skopt:.0f} | Tiempo: {tiempo_bayes_skopt:.2f} s")

# Registramos el experimento (puedes comparar Optuna vs BayesSearchCV, las dos variantes bayesianas)
resumen.append({
    'Metodo': 'Bayesiana (skopt)', 'Modelo': 'XGBoost',
    'RMSE CV': np.sqrt(-bayes_xgb.best_score_),
    'RMSE Test': rmse_test_bayes_skopt,
    'Tiempo (s)': tiempo_bayes_skopt
})
```

### ¿Cómo interpretar la salida?

- `trial.suggest_float(..., log=True)` muestrea en escala logarítmica: ideal para parámetros como `learning_rate` que funcionan en órdenes de magnitud (0.001, 0.01, 0.1...).
- Observa que Optuna prueba valores **continuos** (p. ej. `learning_rate=0.073`) imposibles de alcanzar con un Grid discreto.
- `plot_optimization_history` suele mostrar mejora rápida al inicio y estabilización después: evidencia de que el modelo sustituto está aprendiendo dónde buscar.
- `plot_param_importances` te dice qué hiperparámetros más influyen en el score (en Diamonds, típicamente `max_depth` y `learning_rate`).
- Si tu máquina es lenta, baja `n_trials`/`n_iter` a 15-20: la práctica sigue demostrando el concepto.

---

## 10. Paso 8 — Comparación de métodos: score y tiempo

**¿Qué pide el enunciado?** *"Explique qué método logró el mejor poder predictivo (score) y compare los tiempos de ejecución."*

### ¿Qué es y para qué sirve?

Reunimos los experimentos (3 métodos × 2 modelos, más la variante skopt si la ejecutaste) en una tabla y dos gráficas para responder con evidencia: ¿qué método encontró el mejor modelo? ¿cuánto costó en tiempo cada uno?

**Celda 17 — Tabla comparativa de resultados**

```python
# Convertimos la lista de experimentos en un DataFrame ordenado por el mejor RMSE en test
tabla = pd.DataFrame(resumen)                              # DataFrame a partir de los diccionarios acumulados
tabla = tabla.sort_values('RMSE Test').reset_index(drop=True)  # ordenamos de mejor a peor RMSE en test

# Mostramos la tabla (display es función de notebook; también puedes escribir solo: tabla)
display(tabla.round(2))   # round(2) limita los decimales para que se lea mejor
```

**Celda 18 — Gráfica 1: poder predictivo (RMSE en test)**

```python
# Gráfica de barras: RMSE en test por método de búsqueda, separado por modelo
plt.figure(figsize=(10, 5))
sns.barplot(data=tabla, x='Metodo', y='RMSE Test', hue='Modelo')   # barras agrupadas por modelo
plt.title('Poder predictivo (RMSE en test) según método de búsqueda')  # MENOS es MEJOR
plt.ylabel('RMSE en test (USD)')
plt.xlabel('Método de búsqueda')
plt.legend(title='Modelo')
plt.show()
```

**Celda 19 — Gráfica 2: tiempos de ejecución**

```python
# Gráfica de barras: tiempo de ejecución por método de búsqueda
plt.figure(figsize=(10, 5))
sns.barplot(data=tabla, x='Metodo', y='Tiempo (s)', hue='Modelo')   # barras agrupadas por modelo
plt.title('Tiempo de ejecución según método de búsqueda')
plt.ylabel('Tiempo (segundos)')
plt.xlabel('Método de búsqueda')
plt.legend(title='Modelo')
plt.show()
```

### ¿Cómo interpretar y redactar esta parte?

Busca responder con **tus números** (no copies un texto genérico):

1. ¿Qué método logró el **menor RMSE en test**? Ese es el ganador en poder predictivo.
2. ¿Cuánto tiempo tardó cada método? Calcula proporciones (p. ej., "Grid tardó X segundos, la bayesiana Y segundos, es decir, Z veces menos").
3. ¿Cuál es la **mejor relación calidad/tiempo**? El patrón típico es: Grid encuentra buenos resultados pero es el más lento (y con grids reales, impracticable); Random consigue casi lo mismo con menos tiempo; la Bayesiana iguala o mejora el score explorando de forma inteligente. Si tus resultados difieren, explícalos tal cual (eso demuestra análisis honesto).
4. **Ojo con los tiempos**: dependen de tu hardware y de `n_jobs`; compáralos siempre en términos **relativos**, nunca absolutos.

---

## 11. Paso 9 — Entrenamiento del modelo final

**¿Qué pide el enunciado?** *"Entrene un nuevo modelo con los mejores hiperparámetros."*

### ¿Qué es y para qué sirve?

Tomamos el **ganador** del paso anterior (mejor combinación de hiperparámetros) y entrenamos un **modelo nuevo y definitivo** con todos los datos de entrenamiento. Después, su única evaluación sobre el test nos da el veredicto final del proyecto.

### Decisión técnica: quitar el escalador

Nuestro ganador es un modelo de **árboles**, que es **invariante a transformaciones monótonas** de cada feature: si multiplicas/divides una columna por una constante, los árboles toman exactamente las mismas decisiones. Por eso, para el modelo final quitamos el `StandardScaler`:

- El rendimiento (RMSE/R²) es prácticamente idéntico.
- Las gráficas de XAI quedan en las **unidades originales** del dataset, mucho más fáciles de interpretar ("carat = 1.0" en vez de "carat = 0.42 estandarizado").

> Si tu modelo ganador fuera una **regresión lineal, SVR o red neuronal**, el escalador SÍ es obligatorio: déjalo. Para árboles, esta simplificación es válida y debes justificarla así en tu informe.

**Celda 20 — Entrenar el modelo final**

```python
# Preprocesador del modelo final: SOLO imputación y codificación (sin escalador), porque el ganador es un modelo de árboles
preprocesador_final = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='median'), make_column_selector(dtype_include=np.number)),  # imputa numéricas
    ('cat', pipeline_categorico, make_column_selector(dtype_exclude=np.number)),                # codifica categóricas
])

# Modelo final: reutiliza la mejor combinación encontrada en el Paso 8
# AQUÍ reemplaza por TU ganador. Ejemplo de otras opciones válidas:
#   Ganador Random Forest + Bayesiana:
#       modelo_final_estimador = RandomForestRegressor(**estudio_rf.best_params, random_state=42)
#   Ganador Random Forest + Grid/Random (los parámetros traen prefijo 'modelo__', hay que limpiarlos):
#       def limpiar(params): return {k.replace('modelo__', ''): v for k, v in params.items()}
#       modelo_final_estimador = RandomForestRegressor(**limpiar(grid_rf.best_params_), random_state=42)
# En esta guía asumimos que ganó XGBoost + Optimización Bayesiana (Optuna):
modelo_final = Pipeline(steps=[
    ('preprocesador', preprocesador_final),                            # imputación + one-hot (sin escala)
    ('modelo', XGBRegressor(**estudio_xgb.best_params, random_state=42, enable_categorical=False))  # mejores hiperparámetros
])

# Entrenamos el modelo final con TODOS los datos de entrenamiento
modelo_final.fit(X_train, y_train)
print('Modelo final entrenado.')
```

**Celda 21 — Evaluación final sobre el conjunto de prueba**

```python
# Importamos las métricas de regresión que faltan
from sklearn.metrics import mean_absolute_error, r2_score

# Predicciones del modelo final sobre datos que NUNCA vio (test)
y_pred_final = modelo_final.predict(X_test)

# Calculamos las tres métricas de evaluación
rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))   # raíz del error cuadrático medio (USD)
mae = mean_absolute_error(y_test, y_pred_final)            # error absoluto medio (interpretación directa)
r2 = r2_score(y_test, y_pred_final)                        # coeficiente de determinación (varianza explicada)

print(f"RMSE: {rmse:.0f} USD")
print(f"MAE:  {mae:.0f} USD")
print(f"R²:   {r2:.4f}")

# Gráfica de dispersión: valores reales vs predichos
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_final, alpha=0.3, s=10)         # cada punto = un diamante del test
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # línea ideal (predicción perfecta)
plt.xlabel('Precio real (USD)')
plt.ylabel('Precio predicho (USD)')
plt.title('Predicciones vs valores reales (conjunto de prueba)')
plt.show()
```

### ¿Cómo interpretar la salida?

- **RMSE**: en Diamonds un RMSE ≈ 500-560 USD significa que el error típico ronda los $500 en un rango de precios de $326 a $18.823. Es un resultado excelente.
- **MAE**: "en promedio, el modelo se equivoca por ~$X" (suele ser menor que el RMSE, ~380-450 USD).
- **R²**: si es ~0.97-0.98, el modelo explica el 97-98% de la varianza del precio. Arriba de 0.7 en datos reales ya es bueno; si es menor a ~0.3, revisa features y modelos.
- **Gráfica de dispersión**: si los puntos se pegan a la línea roja diagonal, el modelo predice bien. En Diamonds verás un abanico que se abre en los precios altos (pocos diamantes caros → más error ahí).
- Compara estas métricas con la línea base del Paso 4: deberías ver una mejora (aunque modesta: con árboles ya bien configurados, la optimización suele aportar 1-3%; díselo honestamente en el informe).

---

## 12. Paso 10 — Feature Importance (importancia global)

**¿Qué pide el enunciado?** *"Use Feature Importance... para evaluar la importancia global de las features."*

### ¿Qué es y para qué sirve?

Los modelos de árboles calculan, durante el entrenamiento, cuánto reduce la **impureza** (heterogeneidad de los valores objetivo en cada nodo) cada feature cuando se usa para dividir. Esa reducción acumulada, ponderada por el número de muestras que pasan por cada nodo, es la **importancia por disminución de impureza (MDI)** y viene gratis en `feature_importances_` (lo mismo que en la Práctica 2.1).

Sirve para responder: *"¿qué variables usa más el modelo para decidir?"*

**Celda 22 — Extraer y graficar las importancias**

```python
# Extraemos el estimador ya entrenado que vive dentro del pipeline final
modelo_entrenado = modelo_final.named_steps['modelo']

# Las importancias calculadas por el propio modelo durante el entrenamiento
importancias = modelo_entrenado.feature_importances_

# Nombres de las features tal como salen del preprocesador (mismo orden que las columnas de entrenamiento)
# OJO: las categóricas one-hot aparecen desglosadas (cat__cut_Ideal, cat__color_E, ...)
nombres = modelo_final.named_steps['preprocesador'].get_feature_names_out()

# Armamos un DataFrame ordenado de mayor a menor importancia
importancia_df = pd.DataFrame({
    'Feature': nombres,            # nombre de cada feature (pos-transformación)
    'Importancia': importancias    # su importancia MDI (suman 1 entre todas)
}).sort_values('Importancia', ascending=False)   # ordenamos de la más a la menos importante

print(importancia_df.head(12))     # top 12 features por importancia

# Gráfica de barras horizontales (top 15)
plt.figure(figsize=(10, 8))
sns.barplot(data=importancia_df.head(15), x='Importancia', y='Feature', color='steelblue')
plt.title('Feature Importance (disminución de impureza) — modelo final')
plt.xlabel('Importancia (fracción del total)')
plt.show()
```

### ¿Cómo interpretar la salida?

- En Diamonds, `num__carat` suele acaparar ~60-70% de la importancia: el peso del diamante es, por mucho, el mayor determinante del precio. Tiene total sentido de negocio.
- Después aparecen las categóricas one-hot (p. ej. `cat__clarity_I1`, `cat__cut_Ideal`, `cat__color_D`) y las dimensiones `num__x/y/z`. Interpreta cada columna binaria como "qué tanto pesa esa categoría en las decisiones".
- **Limitaciones del MDI que debes mencionar en tu informe**:
  1. Se calcula sobre los datos de **entrenamiento**: mide cuánto *usa* el modelo cada feature, no cuánto *ayuda* a generalizar.
  2. Favorece features con **muchos valores posibles** (alta cardinalidad) o continuas.
  3. No dice la **dirección** del efecto (si sube o baja la predicción).
  Por eso lo complementamos con las dos técnicas siguientes.

---

## 13. Paso 11 — Permutation Importance (importancia global)

**¿Qué pide el enunciado?** *"Use... Permutation Importance... para evaluar la importancia global de las features."*

### ¿Qué es y para qué sirve?

Idea central (MDA, *Mean Decrease in Accuracy*) — es la técnica de la Práctica 2.1: si una feature es importante, **romperla** debe hacer que el modelo empeore. El procedimiento:

1. Evalúa el modelo sobre el test (score base).
2. Toma una columna y **baraja sus valores al azar** (se destruye su relación con el objetivo).
3. Vuelve a evaluar: la **caída del score** es la importancia de esa feature.
4. Repite varias veces (`n_repeats`) para tener estabilidad (promedio ± desviación).

Ventajas: es **agnóstica al modelo** (sirve para cualquier algoritmo), se mide sobre datos de **prueba** (mide valor predictivo real) y no sufre el sesgo de cardinalidad del MDI.

**Celda 23 — Calcular y graficar la importancia por permutación**

```python
# Función de sklearn para importancia por permutación
from sklearn.inspection import permutation_importance

# Calculamos la importancia permutando cada columna 15 veces
resultado_pi = permutation_importance(
    modelo_final,                            # el pipeline completo (se encarga de transformar X_test)
    X_test,                                  # datos de PRUEBA sin transformar
    y_test,                                  # valores reales del test
    n_repeats=15,                            # repeticiones por columna (más = más estable)
    random_state=42,                         # barajado reproducible
    scoring='neg_mean_squared_error',        # métrica: cuánto EMPEORA el MSE al barajar
    n_jobs=-1                                # paraleliza las repeticiones
)

# Armamos DataFrame con promedio y desviación estándar de la caída de score
pi_df = pd.DataFrame({
    'Feature': nombres,                              # nombres en el orden de las columnas
    'Importancia media': resultado_pi.importances_mean,   # caída PROMEDIO del score
    'Desviación': resultado_pi.importances_std            # variabilidad entre repeticiones
}).sort_values('Importancia media', ascending=False)      # ordenamos de mayor a menor

print(pi_df.head(12))     # top 12

# Gráfica de barras horizontales (top 15) con barras de error (una desviación estándar)
plt.figure(figsize=(10, 8))
top_pi = pi_df.head(15)
plt.barh(top_pi['Feature'], top_pi['Importancia media'],
         xerr=top_pi['Desviación'],               # barras de error = incertidumbre de la estimación
         color='darkorange', capsize=3)           # capsize = topes en los extremos de las barras de error
plt.gca().invert_yaxis()                         # invertimos el eje para ver la más importante arriba
plt.title('Permutation Importance (caída de score al barajar cada feature)')
plt.xlabel('Aumento del error (mayor = feature más importante)')
plt.show()
```

### ¿Cómo interpretar la salida?

- Cuanto mayor la caída del score, más depende el modelo de esa feature. En Diamonds, `num__carat` vuelve a encabezar.
- **Fenómeno interesante con features correlacionadas** (¡menciónalo en tu informe!): como `carat`, `x`, `y` y `z` están fuertemente correlacionadas, al barajar solo `carat` el modelo puede "recuperarse" usando las dimensiones. Por eso la importancia de permutación de `carat` se ve **menor** que su importancia MDI. La lección: con multicolinealidad, la importancia individual se reparte entre las features redundantes; es una limitación conocida de la permutación (y de casi todas las técnicas de importancia).
- **Desviación grande** = la importancia varía mucho entre repeticiones (menos confiable). Regla práctica: si `importancia media < 2 × desviación`, duda de su relevancia.
- Si una feature sale con importancia ~0 (o negativa), el modelo no la necesita para predecir.
- Compara el **ranking** con el MDI del paso anterior: si coinciden en el top, tienes evidencia sólida; si difieren, explica por qué (multicolinealidad o sesgo de cardinalidad del MDI).

---

## 14. Paso 12 — SHAP Values (importancia global)

**¿Qué pide el enunciado?** *"Use... SHAP Values para evaluar la importancia global de las features."*

### ¿Qué es y para qué sirve?

SHAP (Lundberg & Lee, 2017) calcula la contribución **aditiva y justa** de cada feature a cada predicción, usando valores de Shapley de la teoría de juegos cooperativos: reparte la "ganancia" (la diferencia entre la predicción y el valor base) entre las features según su aporte marginal en todas las combinaciones posibles. Propiedades clave:

- **Aditividad**: `predicción = valor_base + Σ contribuciones` (exacta).
- **Consistencia**: si el modelo depende más de una feature, su contribución SHAP no puede disminuir.
- Para modelos de árboles, el explicador calcula los valores SHAP **exactos y rápido** (sin simulaciones).

Es **exactamente la técnica de la Práctica 2.2**, con la misma API moderna: `shap.Explainer(model, X_train)`, `explainer(X)`, `shap.plots.bar`, `shap.plots.beeswarm` y `shap.plots.waterfall`.

**Celda 24 — Calcular SHAP values globales**

```python
# Importamos SHAP (librería de IA explicativa, la misma de la Práctica 2.2)
import shap

# Preparamos los datos de test pasados por el MISMO preprocesador del modelo final
X_test_transformado = modelo_final.named_steps['preprocesador'].transform(X_test)
X_test_transformado = pd.DataFrame(X_test_transformado, columns=nombres)  # DataFrame con nombres

# Datos de ENTRENAMIENTO transformados: sirven como "fondo" (background) para calcular el valor base
X_train_transformado = pd.DataFrame(
    modelo_final.named_steps['preprocesador'].transform(X_train),
    columns=nombres
)

# Creamos el explicador pasando el modelo entrenado y el fondo de datos (API de la Práctica 2.2)
explainer = shap.Explainer(modelo_entrenado, X_train_transformado)

# Calculamos los valores SHAP para todo el conjunto de prueba
explicacion = explainer(X_test_transformado)

# verificamos la forma: (n_muestras_test, n_features)
print('Forma de los valores SHAP:', explicacion.values.shape)
```

**Celda 25 — Gráfica global 1: barras (importancia global promedio)**

```python
# Gráfica de barras: media del valor ABSOLUTO de los SHAP values por feature
# (es el análogo de SHAP a las importancias de los pasos anteriores; igual que la Práctica 2.2)
shap.plots.bar(explicacion, max_display=15)
```

**Celda 26 — Gráfica global 2: beeswarm (enjambre)**

```python
# Beeswarm: cada PUNTO es una muestra; su posición X es el impacto SHAP en la predicción
# y su COLOR es el valor de la feature (rojo = alto, azul = bajo)
shap.plots.beeswarm(explicacion, max_display=15)   # max_display limita las features mostradas
```

### ¿Cómo interpretar la salida?

**Barras**: promedio de |SHAP| por feature. Es la "tabla de posiciones" global: compárala con MDI y Permutation. Si las tres técnicas coinciden en el top (`num__carat` primero, seguido de las categorías de `clarity`, `cut` y `color`), tu conclusión es robusta.

**Beeswarm**:

- El eje Y ordena las features por importancia global (la más arriba = más importante).
- Cada punto es una **muestra del test**: si está a la derecha del cero, esa feature **empujó esa predicción hacia arriba**; a la izquierda, hacia abajo.
- El color indica el **valor de la feature**: en Diamonds, los puntos rojos de `num__carat` (diamantes pesados) están muy a la derecha → a mayor peso, mayor precio predicho. Confirma una relación **positiva y consistente**.
- Para las categóricas one-hot (0/1), el "rojo" significa presencia de la categoría: p. ej. `cat__clarity_I1` rojo a la izquierda → un diamante de pureza I1 (la peor) baja el precio; `cat__cut_Ideal` rojo a la derecha → el corte Ideal lo sube. Así SHAP te dice la **dirección** del efecto, cosa que MDI y Permutation no hacen.

---

## 15. Paso 13 — Predicción nueva y explicación LOCAL con SHAP

**¿Qué pide el enunciado?** *"Realice una predicción nueva y use SHAP VALUES para evaluar la importancia local de las features."*

### ¿Qué es y para qué sirve?

Hasta ahora explicamos el modelo **en general**. Ahora tomamos **un solo diamante nuevo** (que el modelo nunca vio), predecimos su precio y desglosamos **por qué** el modelo dijo ese número: qué features empujaron el precio hacia arriba y cuáles hacia abajo. Esto es la **explicación local** (el waterfall de la Práctica 2.2), y es lo que usarías en producción para justificar una predicción ante un usuario.

**Celda 27 — Crear y predecir un diamante nuevo**

```python
# Creamos UNA nueva observación: un diamante de 1 quilate, corte Ideal, color G, pureza SI1
# Valores en las unidades originales del dataset (mismo orden de columnas)
nuevo_diamante = pd.DataFrame({
    'carat':   [1.0],        # peso en quilates
    'cut':     ['Ideal'],    # calidad del corte (categórica)
    'color':   ['G'],        # grado de color (categórica)
    'clarity': ['SI1'],      # pureza (categórica)
    'depth':   [61.5],       # profundidad total (%)
    'table':   [57.0],       # ancho de la mesa (%)
    'x':       [6.4],        # largo (mm)
    'y':       [6.4],        # ancho (mm)
    'z':       [3.9],        # alto (mm)
})

# El modelo final predice el precio del diamante
prediccion_nueva = modelo_final.predict(nuevo_diamante)

print(f"Precio predicho: ${prediccion_nueva[0]:,.0f} USD")
```

**Celda 28 — Explicación local con SHAP (waterfall)**

```python
# Transformamos el diamante nuevo con el preprocesador del modelo final (imputación + one-hot, sin escala)
X_nueva_transformada = pd.DataFrame(
    modelo_final.named_steps['preprocesador'].transform(nuevo_diamante),
    columns=nombres
)

# Calculamos los valores SHAP de esa ÚNICA predicción
explicacion_local = explainer(X_nueva_transformada)

# Waterfall: muestra cómo cada feature empujó la predicción desde el valor base
# hasta el valor final predicho (rojo = empuja hacia arriba, azul = hacia abajo)
shap.plots.waterfall(explicacion_local[0], max_display=12)
```

**Celda 29 — Verificación numérica de la aditividad de SHAP**

```python
# Comprobamos la identidad fundamental de SHAP: predicción = base + suma de contribuciones
base = float(np.ravel(explicacion_local.base_values)[0])   # valor base = predicción esperada del modelo
contribuciones = explicacion_local.values[0]               # contribuciones de cada feature

print(f"Valor base (predicción esperada del modelo): ${base:,.0f}")
print(f"Suma de contribuciones SHAP:                 ${contribuciones.sum():,.0f}")
print(f"Predicción reconstruida (base + suma):       ${base + contribuciones.sum():,.0f}")
print(f"Predicción real del modelo:                  ${prediccion_nueva[0]:,.0f}")
```

### ¿Cómo interpretar la salida?

**Waterfall**: se lee de abajo hacia arriba:

- `E[f(x)]` (base): lo que el modelo predeciría "sin saber nada" (la media de sus predicciones de entrenamiento, ≈ $3.930 en Diamonds).
- Cada fila es una feature: la flecha **roja** (derecha) suma al precio, la **azul** (izquierda) resta. El número de la derecha es el valor de la feature (en unidades originales, porque quitamos el escalador).
- `f(x)` arriba: la predicción final = base + todas las flechas.

Para el diamante de ejemplo (1 ct, Ideal, G, SI1), espera una historia como esta:

- `num__carat = 1.0` (por encima de la media de 0.8): flecha roja grande → +$1.500 aprox.
- `cat__cut_Ideal = 1`: flecha roja pequeña → el corte Ideal suma (premium de marca).
- `cat__clarity_SI1 = 1`: flecha azul → la pureza SI1 (media-baja) resta un poco frente a claridades superiores.
- `cat__color_G = 1`: flecha azul pequeña → el color G resta frente a los D/E/F más raros.
- `num__x/y/z`: contribuciones pequeñas (la información de tamaño ya la aporta `carat`).

La **celda 29** debe dar dos números prácticamente idénticos en las dos últimas líneas: esa es la prueba matemática de que SHAP descompone la predicción de forma exacta (pequeñas diferencias de 1e-8 son solo error numérico de punto flotante).

> **Nota**: `force_plot` (la gráfica antigua de SHAP) está deprecada en versiones recientes; el waterfall es su reemplazo oficial (es el que usaste en la Práctica 2.2).

---

## 16. Conclusiones (lo que debes redactar en tu informe)

Tu informe debe responder, con tus propios números y gráficas:

1. **Dataset**: cuál elegiste, de dónde (cita la fuente) y por qué cumple "diferente a los de las prácticas" (las prácticas usaron Adult, California Housing, Diabetes, Breast Cancer, series sintéticas e insurance).
2. **EDA**: hallazgos principales de tu `AutoEDA` reutilizada (faltantes, outliers, sesgo, multicolinealidad VIF en `carat`/`x`/`y`/`z`, correlaciones con el objetivo) y las decisiones que tomaste a partir de ellos (p. ej. "no elimino outliers porque los árboles son robustos").
3. **Preprocesamiento**: qué hiciste (imputación, escalado, OneHot) y por qué, y cómo el Pipeline evita la fuga de datos.
4. **Métodos de búsqueda**: tabla comparativa de los experimentos. ¿Qué método logró el mejor poder predictivo (score)? ¿Cuánto tardó cada uno? ¿Cuál ofrece la mejor relación calidad/tiempo y por qué? (fundamenta con los conceptos: explosión combinatoria de Grid, eficiencia de muestreo de Random, modelo sustituto de Bayesiana).
5. **Modelo final**: qué modelo ganó, con qué hiperparámetros, y sus métricas finales (RMSE, MAE, R²) vs la línea base.
6. **Importancia global**: qué features son las más importantes según MDI, Permutation y SHAP. ¿Coinciden los rankings? ¿Tienen sentido de negocio (el peso del diamante manda en el precio)? Explica el efecto de la multicolinealidad en la Permutation Importance.
7. **Explicación local**: muestra el waterfall de tu predicción nueva y narra la historia ("el modelo predice $X porque carat=1.0 sumó +..., la pureza SI1 restó -...").
8. **Limitaciones**: sesgos del MDI, dependencia del hardware en los tiempos, error mayor en diamantes caros (pocos ejemplos), etc.

---

## 17. Checklist final de entregables

Marca cada punto antes de entregar:

- [ ] Dataset de regresión **diferente a los de las prácticas** (no usar Adult, California Housing, Diabetes, Breast Cancer, insurance ni series sintéticas) y **citado**.
- [ ] EDA automatizado **reutilizado de la Actividad 1**: `from herramientas_eda import AutoEDA` + `run_analysis()` + `obtener_hallazgos()`.
- [ ] Preprocesamiento con Pipeline (sin fuga de datos) y partición 80/20 con semilla fija.
- [ ] 2 modelos de ML con línea base por defecto + referencia Dummy (Random Forest y XGBoost, los del curso).
- [ ] Grid Search, Búsqueda Aleatoria y Optimización Bayesiana (Optuna `TPESampler` y/o `BayesSearchCV`) ejecutados para **ambos** modelos.
- [ ] Tabla comparativa y 2 gráficas: RMSE en test y tiempos de ejecución.
- [ ] Explicación escrita de qué método ganó en score y en tiempo.
- [ ] Modelo final entrenado con los mejores hiperparámetros + métricas RMSE/MAE/R².
- [ ] Feature Importance, Permutation Importance y SHAP global (con sus gráficas: bar, beeswarm).
- [ ] Predicción nueva + waterfall SHAP local + verificación de aditividad.
- [ ] Conclusiones respondiendo punto por punto lo que pide el enunciado.
- [ ] Notebook ejecutado completo, sin errores, de arriba hacia abajo.

---

## Referencias

- Pace, R. K., & Barry, R. (1997). Sparse spatial autoregressions. *Statistics & Probability Letters*, 33(3), 291-297. (Dataset California Housing, usado en las prácticas 2.1 y 2.2)
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *JMLR*, 13, 281-305.
- Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for hyper-parameter optimization. *NeurIPS 24*. (TPE)
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*. (SHAP)
- Documentación: scikit-learn (https://scikit-learn.org), XGBoost (https://xgboost.readthedocs.io), SHAP (https://shap.readthedocs.io), Optuna (https://optuna.org), scikit-optimize (https://scikit-optimize.github.io)
- Prácticas de clase del repositorio del curso: `Unidad 1` (Prácticas 1, 2.1, 2.2, 3), `Unidad 2` (Prácticas 4, 5, 6A, 6B) y `Actividad Integradora 1` (`herramientas_eda.py`).
