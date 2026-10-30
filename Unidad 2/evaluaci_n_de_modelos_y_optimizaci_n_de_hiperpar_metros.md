# Guía Avanzada de Evaluación de Modelos, Métricas y Optimización de Hiperparámetros

Este documento complementa el manual técnico de Ciencia de Datos y Machine Learning. Está centrado en los métodos cuantitativos para evaluar algoritmos de clasificación y regresión, la calibración de umbrales de decisión y el uso de técnicas modernas de optimización de hiperparámetros.

## Tabla de Contenidos

 1. [Aprendizaje Supervisado: Clasificación vs. Regresión](#1-aprendizaje-supervisado-clasificación-vs-regresión)

 2. [Clasificación Binaria y Datasets Benchmark](#2-clasificación-binaria-y-datasets-benchmark)

 3. [AdaBoost (Adaptive Boosting)](#3-adaboost-adaptive-boosting)

 4. [Matriz de Confusión y Métricas Derivadas](#4-matriz-de-confusión-y-métricas-derivadas)

 5. [Exactitud (Accuracy) y la Paradoja de la Exactitud](#5-exactitud-accuracy-y-la-paradoja-de-la-exactitud)

 6. [Probabilidades de Predicción (`predict_proba`)](#6-probabilidades-de-predicción-predict_proba)

 7. [Curva ROC y Área Bajo la Curva (AUC)](#7-curva-roc-y-área-bajo-la-curva-auc)

 8. [Umbral de Decisión, TPR y FPR](#8-umbral-de-decisión-tpr-y-fpr)

 9. [Regresión y Métricas de Error: MSE y RMSE](#9-regresión-y-métricas-de-error-mse-y-rmse)

10. [Random Forest para Regresión (`RandomForestRegressor`)](#10-random-forest-para-regresión-randomforestregressor)

11. [Parámetros vs. Hiperparámetros](#11-parámetros-vs-hiperparámetros)

12. [Validación Cruzada (Cross-Validation - CV)](#12-validación-cruzada-cross-validation---cv)

13. [Búsqueda en Cuadrícula (`GridSearchCV`)](#13-búsqueda-en-cuadrícula-gridsearchcv)

14. [Búsqueda Aleatoria (`RandomizedSearchCV`)](#14-búsqueda-aleatoria-randomizedsearchcv)

15. [Optimización Bayesiana (Optuna, TPESampler y `BayesSearchCV`)](#15-optimización-bayesiana-optuna-tpesampler-y-bayessearchcv)

### 1. Aprendizaje Supervisado: Clasificación vs. Regresión

El **Aprendizaje Supervisado** busca construir una función mapeadora $f: \mathcal{X} \rightarrow \mathcal{Y}$ que minimice el error de predicción sobre datos etiquetados $\{(x_i, y_i)\}_{i=1}^N$. La naturaleza del espacio de salida $\mathcal{Y}$ define las dos subdisciplinas principales:

* **Clasificación:** El dominio del objetivo $\mathcal{Y}$ es discreto o categórico, $\mathcal{Y} \in \{C_1, C_2, ..., C_K\}$.

  * *Ejemplos:* Detección de fraude financiero (Sí/No), diagnóstico oncológico (Benigno/Maligno), clasificación de documentos por temas.

* **Regresión:** El dominio del objetivo $\mathcal{Y}$ es continuo y cuantitativo, $\mathcal{Y} \in \mathbb{R}$.

  * *Ejemplos:* Pronóstico de la demanda eléctrica en megavatios, estimación del valor comercial de un inmueble, medición de niveles de concentración de glucosa en sangre.

Ambas tareas requieren estrategias de validación y funciones de pérdida (*loss functions*) diferenciadas.

### 2. Clasificación Binaria y Datasets Benchmark

La **Clasificación Binaria** representa la forma más fundamental de clasificación donde la variable objetivo posee exactamente dos estados mutuamente excluyentes: $\vert{} \mathcal{Y} \vert{} = 2$. Por convención cuantitativa, se codifican como $y \in \{0, 1\}$, representando la **clase negativa** ($0$) y la **clase positiva** ($1$).

#### Datasets Benchmark Clásicos:

* **Breast Cancer Wisconsin Dataset:**

  * *Registros:* $569$ pacientes.

  * *Atributos:* $30$ características continuas extraídas de imágenes digitalizadas de una aspiración con aguja fina de una masa mamaria (radio, textura, perímetro, área, suavidad).

  * *Objetivo:* Clasificar el tumor como Maligno ($1$) o Benigno ($0$).

* **Adult Income Dataset:**

  * *Registros:* $32,561$ filas en su partición estándar.

  * *Atributos:* $14$ variables socioeconómicas y demográficas.

  * *Objetivo:* Predecir si los ingresos anuales superan los $\$50,000$ USD (`>50K` $\rightarrow 1$, `<=50K` $\rightarrow 0$).

### 3. AdaBoost (Adaptive Boosting)

**AdaBoost**, abreviatura de **Adaptive Boosting**, es un algoritmo precursor de aprendizaje por ensamble propuesto por Yoav Freund y Robert Schapire.

#### Mecanismo Operativo:

A diferencia del embolsado (*Bagging*), que entrena estimadores en paralelo de forma independiente, AdaBoost entrena clasificadores débiles (*weak learners* —típicamente árboles de decisión de un solo nivel conocidos como *decision stumps*) de manera **secuencial**.

1. Inicialmente asigna un peso uniforme $w_i = \frac{1}{N}$ a cada observación $i$.

2. En cada iteración $m$:
   a. Ajusta el clasificador débil $h_m(x)$ maximizando el acierto según los pesos actuales.
   b. Evalúa la tasa de error ponderada $\epsilon_m$.
   c. Calcula la importancia del clasificador $\alpha_m = \frac{1}{2} \ln \left(\frac{1 - \epsilon_m}{\epsilon_m}\right)$.
   d. Actualiza e incrementa exponencialmente los pesos de las observaciones **incorrectamente clasificadas**, obligando al siguiente estimador a enfocarse en los casos difíciles.

3. La predicción final se calcula como una combinación lineal ponderada:

$$
H(x) = \text{sign}\left(\sum_{m=1}^M \alpha_m h_m(x)\right)
$$

En la librería `scikit-learn`, se instancia mediante la clase `AdaBoostClassifier`.

### 4. Matriz de Confusión y Métricas Derivadas

La **Matriz de Confusión** es una herramienta matricial $2 \times 2$ que contrapone las clases reales observadas frente a las predicciones emitidas por un modelo de clasificación binaria:

|  | **Predicción Negativa (**$0$**)** | **Predicción Positiva (**$1$**)** | 
| ----- | ----- | ----- | 
| **Clase Real Negativa (**$0$**)** | **VN** (Verdaderos Negativos) | **FP** (Falsos Positivos - Error Tipo I) | 
| **Clase Real Positiva (**$1$**)** | **FN** (Falsos Negativos - Error Tipo II) | **VP** (Verdaderos Positivos) | 

#### Métricas Estadísticas Derivadas:

* **Exactitud (*Accuracy*):**
  

  $$
  \text{Accuracy} = \frac{VP + VN}{VP + VN + FP + FN}
  $$

* **Precisión (*Precision* / Valor Predictivo Positivo):** Mide la confiabilidad de las alarmas positivas emitidas por el modelo.
  

  $$
  \text{Precisión} = \frac{VP}{VP + FP}
  $$

* **Sensibilidad (*Recall* / *Sensitivity* / Tasa de Verdaderos Positivos - TPR):** Mide la capacidad del modelo para identificar todos los casos positivos reales.
  

  $$
  \text{Sensibilidad (TPR)} = \frac{VP}{VP + FN}
  $$

* **Especificidad (*Specificity* / Tasa de Verdaderos Negativos - TNR):** Mide la tasa de acierto sobre la clase negativa real.
  

  $$
  \text{Especificidad} = \frac{VN}{VN + FP}
  $$

### 5. Exactitud (Accuracy) y la Paradoja de la Exactitud

La métrica de **Exactitud (*Accuracy*)** evalúa la fracción global de predicciones correctas. A pesar de su simplicidad conceptual, presenta limitaciones graves en conjuntos de datos con **desbalance severo de clases**.

#### La Paradoja de la Exactitud (*Accuracy Paradox*):

Considere un dataset de detección de fraudes donde el $99\%$ de las transacciones son legítimas ($0$) y solo el $1\%$ son fraudulentas ($1$).

Un modelo ingenuo o colapsado que simplemente prediga siempre la clase mayoritaria ($0$) alcanzará un $99\%$ **de Exactitud**. Sin embargo:

* Su **Sensibilidad (*Recall*)** será de $0\%$, siendo incapaz de detectar un solo fraude.

* El modelo resulta completamente inútil para el negocio.

Por este motivo, en contextos reales es indispensable complementar o sustituir la exactitud por métricas como la puntuación **F1-Score** (media armónica entre Precisión y Recall), el área bajo la curva **ROC-AUC** o el **Coeficiente de Correlación de Matthews (MCC)**.

### 6. Probabilidades de Predicción (`predict_proba`)

La mayoría de los algoritmos de aprendizaje supervisado no emiten directamente una etiqueta rígida $\{0, 1\}$, sino que estiman internamente la distribución de probabilidad a posteriori de la clase positiva dada la entrada de características $x$: $P(Y = 1 \mid X = x)$.

En `scikit-learn`, esto se calcula mediante el método `.predict_proba(X)`:

```
# Retorna una matriz de dimensiones (N_muestras, 2)
probabilidades = modelo.predict_proba(X_test)

# La columna 0 representa P(Y=0|X) y la columna 1 representa P(Y=1|X)
prob_clase_positiva = probabilidades[:, 1]

```

#### Aplicaciones de `predict_proba`:

* Permite ordenar observaciones según su nivel de riesgo relativo.

* Es el insumo cuantitativo indispensable para trazar la curva ROC y calcular el área AUC.

* Permite modificar dinámicamente el **umbral de decisión (*decision threshold*)** sin necesidad de reentrenar el modelo.

### 7. Curva ROC y AUC

* **ROC:** *Receiver Operating Characteristic* (Característica Operativa del Receptor).

* **AUC:** *Area Under the Curve* (Área Bajo la Curva).

La **Curva ROC** es una representación gráfica continua de la capacidad diagnóstica de un clasificador binario. Trama la **Tasa de Verdaderos Positivos (TPR / Sensibilidad)** en el eje $Y$ contra la **Tasa de Falsos Positivos (FPR)** en el eje $X$, conforme se varía el umbral de decisión desde $1.0$ hasta $0.0$.

$$
\text{FPR} = \frac{FP}{FP + VN} = 1 - \text{Especificidad}
$$

```
   1.0 |                   /--- (Modelo Excelente, AUC ~ 0.95)
       |                 /
  T    |               /
  P    |             /
  R    |           /
       |         /  (Modelo Aleatorio, AUC = 0.50)
   0.0 +-----------------------
      0.0                     1.0
                 FPR

```

#### Interpretación del Valor AUC:

El **AUC** sintetiza el rendimiento global de la curva ROC en un único valor escalar acotado entre $0.0$ y $1.0$:

* $\text{AUC} = 1.0$: Separador perfecto. El modelo distingue todas las instancias positivas de las negativas sin solapamiento.

* $\text{AUC} = 0.5$: Desempeño equivalente a lanzar una moneda al aire (aleatoriedad pura).

* $\text{AUC} < 0.5$: El modelo invierte sistemáticamente las predicciones (indica que invertir las etiquetas produciría un buen modelo).

### 8. Umbral de Decision, FPR y TPR

Por defecto, la gran mayoría de los clasificadores aplican un **umbral de decisión de** $0.50$: si $P(Y=1 \mid X) \ge 0.50$, asignan la clase $1$; de lo contrario, asignan la clase $0$.

Sin embargo, este umbral por defecto rara vez es óptimo en escenarios no simétricos donde los costos de los errores difieren (ej. un Falso Negativo en cáncer es inmensamente más grave que un Falso Positivo).

#### Curva de Umbrales con `roc_curve`:

La función `sklearn.metrics.roc_curve` evalúa exhaustivamente todos los umbrales posibles y retorna los arreglos de $FPR$, $TPR$ y los correspondientes umbrales:

```
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_true, prob_clase_positiva)

```

Al reducir el umbral de decisión (ej. de $0.5$ a $0.2$):

* Se facilita la clasificación como positiva $\rightarrow$ **Aumenta la Sensibilidad (TPR)** (se capturan más casos reales positivos).

* **Aumenta también la Tasa de Falsos Positivos (FPR)** (se generan más falsas alarmas).

El analista debe seleccionar el umbral que maximice la métrica del negocio o que optimice el **Índice J de Youden**: $J = \text{TPR} - \text{FPR}$.

### 9. Regresión y Métricas de Error: MSE y RMSE

En problemas de regresión, la evaluación mide la discrepancia continua entre el valor real $y_i$ y la estimación calculada $\hat{y}_i$.

#### Mean Squared Error (MSE - Error Cuadrático Medio):

Calcula el promedio aritmético de las desviaciones al cuadrado entre la variable real y la predicción:

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

* *Propiedad:* Penaliza de forma severa y cuadrática las desviaciones grandes (*outliers*).

* *Desventaja:* Sus unidades corresponden al cuadrado de las unidades originales del objetivo (ej. $\text{USD}^2$), lo que dificulta su interpretación intuitiva.

#### Root Mean Squared Error (RMSE - Raíz del Error Cuadrático Medio):

Se obtiene calculando la raíz cuadrada del MSE:

$$
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

* *Ventaja:* Conserva las **mismas unidades de medida que la variable objetivo** $y$ (ej. dólares USD, grados Celsius), ofreciendo una interpretación clara de la magnitud promedio del margen de error.

### 10. Random Forest para Regresión (`RandomForestRegressor`)

El algoritmo **Random Forest** puede adaptarse nativamente de problemas de clasificación a tareas de regresión mediante la clase `RandomForestRegressor`.

#### Diferencias Clave con la Versión de Clasificación:

1. **Criterio de Divisibilidad en los Nodos:** En lugar del Índice Gini o la Entropía, las divisiones (*splits*) en los árboles se determinan minimizando la **Varianza** o el **Error Cuadrático Medio (MSE)** del subconjunto de datos resultante.

2. **Predicción en las Hojas:** La predicción individual de cada árbol individual para un nuevo registro no se realiza por votación, sino asignando el **promedio numérico** de las etiquetas reales de las observaciones de entrenamiento que cayeron en ese nodo hoja.

3. **Predicción del Ensamble:** La predicción final del bosque es la media de las predicciones de todos los árboles individuales:

$$
\hat{y}_{RF} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)
$$

Esta promediación reduce drásticamente la varianza global del modelo en comparación con un solo árbol de decisión.

### 11. Parámetros vs. Hiperparámetros

En Machine Learning es crítico distinguir los componentes que el modelo aprende internamente de aquellos que deben configurarse externamente:

```
+-------------------------------------------------------------------+
|                        MODELO DE ML                               |
|                                                                   |
|   HIPERPARÁMETROS (Definidos previo al entrenamiento)             |
|   ├── n_estimators, max_depth, learning_rate                      |
|                                                                   |
|   [ Proceso de Entrenamiento / Fit ]                              |
|                                                                   |
|   PARÁMETROS (Aprendidos automáticamente de los datos)            |
|   └── Pesos (w), Sesgo (b), Umbrales de corte en los nodos        |
+-------------------------------------------------------------------+

```

#### Comparación Detallada:

| Atributo | Parámetros | Hiperparámetros | 
| ----- | ----- | ----- | 
| **Definición** | Valores internos que el algoritmo ajusta durante el entrenamiento. | Configuraciones estructurales fijadas antes de iniciar el entrenamiento. | 
| **Origen** | Aprendidos automáticamente a partir de los datos. | Definidos manualmente o ajustados por algoritmos de optimización. | 
| **Ejemplos** | Coeficientes $\beta$ en regresión, pesos $w$ en redes neuronales, divisiones en árboles. | `n_estimators`, `max_depth`, `min_samples_split`, `learning_rate`. | 
| **Propósito** | Representar la relación subyacente de los datos. | Controlar la capacidad, complejidad y grado de regularización del modelo. | 

### 12. Validación Cruzada (Cross-Validation - CV)

Dividir los datos simplemente en entrenamiento ($80\%$) y prueba ($20\%$) puede generar estimaciones inestables de rendimiento si la partición de prueba resulta ser atípicamente "fácil" o "difícil". La **Validación Cruzada de K-Fold (K-Fold Cross-Validation)** resuelve esta fragilidad.

```
Datos Totales
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│  Fold 1  │  Fold 2  │  Fold 3  │  Fold 4  │  Fold 5  │
└──────────┴──────────┴──────────┴──────────┴──────────┘
 Iteración 1:  [ TEST ]   [ TRAIN]   [ TRAIN]   [ TRAIN]   [ TRAIN]  --> Score 1
 Iteración 2:  [ TRAIN]   [ TEST ]   [ TRAIN]   [ TRAIN]   [ TRAIN]  --> Score 2
 Iteración 3:  [ TRAIN]   [ TRAIN]   [ TEST ]   [ TRAIN]   [ TRAIN]  --> Score 3
 Iteración 4:  [ TRAIN]   [ TRAIN]   [ TRAIN]   [ TEST ]   [ TRAIN]  --> Score 4
 Iteración 5:  [ TRAIN]   [ TRAIN]   [ TRAIN]   [ TRAIN]   [ TEST ]  --> Score 5

 Rendimiento Final = Promedio(Score 1, Score 2, Score 3, Score 4, Score 5)

```

#### Pasos del Algoritmo con `cv=5`:

1. El dataset original se particiona en $K = 5$ bloques (*folds*) mutuamente excluyentes de igual tamaño.

2. Se ejecutan $5$ iteraciones. En cada una, $4$ bloques se combinan para entrenar el modelo y el bloque restante se reserva como validación.

3. Se calcula el promedio y la desviación estándar de las $5$ métricas obtenidas, obteniendo una evaluación de la capacidad de generalización del modelo.

### 13. Búsqueda en Cuadrícula (`GridSearchCV`)

`GridSearchCV` es una estrategia exhaustiva y determinista para la optimización de hiperparámetros.

#### Funcionamiento:

El usuario define un espacio discreto de búsqueda mediante un diccionario de hiperparámetros (`param_grid`). El algoritmo genera el **producto cartesiano** de todas las combinaciones posibles y evalúa cada una aplicando Validación Cruzada.

```
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5]
}
# Total de combinaciones = 2 * 3 * 2 = 12 combinaciones.
# Con cv=5, se entrenan e inspeccionan 12 * 5 = 60 modelos.

```

* **Ventaja:** Garantiza la evaluación del punto óptimo dentro de la malla definida.

* **Desventaja:** Sufre severamente de la **maldición de la dimensionalidad**. El costo computacional crece exponencialmente al añadir nuevos hiperparámetros o valores a la rejilla.

### 14. Búsqueda Aleatoria (`RandomizedSearchCV`)

Para sobrellevar la ineficiencia computacional de `GridSearchCV`, Bergstra y Bengio demostraron que muestrear puntos aleatorios en el espacio de hiperparámetros es significativamente más eficiente.

`RandomizedSearchCV` implementa esta metodología muestreando un número fijo de combinaciones (`n_iter`) a partir de distribuciones probabilísticas continuas o discretas asignadas a cada hiperparámetro:

```
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': randint(2, 11)
}

random_search = RandomizedSearchCV(
    estimator=modelo,
    param_distributions=param_distributions,
    n_iter=50,  # Evalúa exactamente 50 combinaciones al azar
    cv=5,
    random_state=42
)

```

* **Ventaja:** Explora un rango de valores más amplio y diverso asignando el presupuesto computacional deseado a través de `n_iter`.

### 15. Optimización Bayesiana (Optuna, TPESampler y `BayesSearchCV`)

Tanto Grid como Random Search sufren de una ineficiencia fundamental: **tratan cada evaluación de forma independiente e ignoran los resultados pasados**.

La **Optimización Bayesiana** aborda la búsqueda como un problema de aprendizaje secuencial. Construye un modelo probabilístico sustituto (*surrogate model*) que aprende la relación entre los hiperparámetros y la métrica de rendimiento, orientando las siguientes evaluaciones hacia las regiones más prometedoras.

```
       Evaluaciones Pasadas (Puntos de prueba)
                         │
                         ▼
        ┌───────────────────────────────────┐
        │  Modelo Sustituto Probabilístico  │
        └───────────────────────────────────┘
                         │
                         ▼
         Función de Adquisición (EI / UCB)
                         │
                         ▼
     Selecciona el PRÓXIMO candidato a evaluar

```

#### Marcos de Trabajo Modernos:

1. **Optuna:**
   Un marco moderno e intuitivo de optimización de hiperparámetros basado en funciones objetivo.

   * Utiliza por defecto el algoritmo **TPESampler** (**Tree-structured Parzen Estimator**), el cual modela de forma independiente las distribuciones de hiperparámetros que produjeron buenos resultados frente a las que produjeron malos resultados.

   * Incorpora mecánicas de poda automática (*pruning*) para detener tempranamente los experimentos que muestran un rendimiento inicial deficiente.

   ```
   import optuna
   
   def objective(trial):
       max_depth = trial.suggest_int('max_depth', 3, 20)
       n_estimators = trial.suggest_int('n_estimators', 50, 400)
       lr = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
   
       # Entrenar y evaluar modelo con Validación Cruzada
       score = evaluar_modelo(max_depth, n_estimators, lr)
       return score
   
   study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
   study.optimize(objective, n_trials=100)
   
   ```

2. **Scikit-Optimize (`BayesSearchCV`):**
   Provee una interfaz compatible con `scikit-learn` para ejecutar optimización bayesiana definiendo espacios de búsqueda explícitos:

   * **`Real(low, high, prior='log-uniform')`:** Para hiperparámetros continuos como la tasa de aprendizaje (*learning rate*) o términos de regularización.

   * **`Integer(low, high)`:** Para recuentos discretos como `n_estimators` o profundidad de árbol.

   * **`Categorical(['gini', 'entropy'])`:** Para atributos cualitativos.