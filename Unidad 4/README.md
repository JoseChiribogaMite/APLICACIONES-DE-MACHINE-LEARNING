# Unidad 4 — Redes neuronales: fundamentos, implementación y aplicaciones

Guía de aprendizaje para comprender qué calcula una red neuronal, cómo aprende mediante gradientes y cómo se implementa una clasificación multiclase. La **Práctica 8** utiliza TensorFlow/Keras para predecir tipos de cobertura forestal.

El objetivo no es solo ejecutar `fit`: es relacionar datos, arquitectura, pérdida, optimización y evaluación para explicar cada decisión técnica.

## Contenido y ruta de aprendizaje

| Recurso | Contenido | Cómo utilizarlo |
|---|---|---|
| [Presentación de la unidad](Unid4_%20Redes%20Neuronales%20.pptx) | Neurona artificial, activaciones, perceptrón multicapa y propagación hacia atrás | Estudiar los fundamentos antes de entrenar |
| [Práctica 8](Practica8.ipynb) | Clasificación de Forest CoverType con una red densa | Ejecutar las celdas en orden, considerando la aclaración sobre `.score` |

Ruta sugerida: neurona → capas y activaciones → pérdida y gradientes → preparación de datos → arquitectura → entrenamiento → evaluación → diagnóstico y mejoras.

La unidad anterior trata [agrupación sin etiquetas](../Unida%203/README.md). Aquí el aprendizaje es **supervisado**: se dispone del tipo de cobertura verdadero para calcular el error durante el entrenamiento.

## 1. La neurona artificial

La analogía con una neurona biológica sirve como motivación; una neurona artificial es una operación matemática, no una reproducción detallada del cerebro.

Recibe entradas, calcula una combinación ponderada y aplica una activación:

$$
z=\sum_{j=1}^{d}w_jx_j+b,\qquad a=f(z)
$$

| Elemento | Función |
|---|---|
| `x_j` | Variable de entrada |
| `w_j` | Peso aprendido que modifica su contribución |
| `b` | Sesgo aprendido que desplaza la respuesta |
| `z` | Valor antes de la activación |
| `f` | Función de activación |
| `a` | Salida de la neurona |

Para ilustrarlo, considera dos entradas: 5 horas de estudio y 90 % de asistencia, con pesos `0.5`, `0.03` y sesgo `-4`:

$$
z=0.5(5)+0.03(90)-4=1.2
$$

La sigmoide transforma ese valor en aproximadamente `0.768`. En un clasificador binario podría interpretarse como una probabilidad **estimada**, siempre que el modelo se haya entrenado y evaluado adecuadamente. Elegir pesos a mano no demuestra una probabilidad real de aprobar.

La unidad de medida importa: representar asistencia como `90` o como `0.90` cambia el cálculo si no se adapta el peso o el preprocesamiento.

## 2. Activaciones y capacidad de representación

| Activación | Expresión | Rango | Uso habitual |
|---|---|---|---|
| Sigmoide | `1 / (1 + exp(-z))` | `(0, 1)` | Salida binaria |
| ReLU | `max(0, z)` | `[0, infinito)` | Capas ocultas |
| `tanh` | Tangente hiperbólica de `z` | `(-1, 1)` | Capas ocultas en ciertas arquitecturas |
| Softmax | Exponencial normalizada por clase | Cada salida en `(0, 1)`; suman uno | Salida multiclase excluyente |
| Lineal | `z` | Reales | Salida de regresión |

Para `z=1.2`, la sigmoide vale aproximadamente `0.768`, ReLU vale `1.2` y `tanh` aproximadamente `0.834`.

**¿Por qué se necesita no linealidad?** La composición de varias transformaciones afines sin activaciones no lineales equivale a otra transformación afín. Añadir capas lineales no permite por sí solo representar relaciones más complejas.

ReLU es sencilla y su gradiente no se satura para valores positivos, aunque una neurona puede dejar de activarse si permanece en la región negativa. Sigmoide y `tanh` pueden presentar gradientes muy pequeños en sus regiones saturadas.

Softmax convierte los valores de salida —*logits*— en una distribución:

$$
p_k=\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}
$$

Es apropiada cuando cada observación pertenece a **una sola clase**. Un problema multietiqueta, donde varias clases pueden ser verdaderas simultáneamente, requiere otra configuración.

## 3. Del perceptrón al perceptrón multicapa

Un perceptrón clásico es un clasificador con una frontera lineal. Un **MLP** (*Multilayer Perceptron*) conecta capas de neuronas e introduce activaciones no lineales para aprender relaciones más complejas.

- **Entrada:** recibe las características; no constituye por sí sola una capa de pesos entrenables.
- **Capas ocultas:** transforman las representaciones.
- **Salida:** produce valores compatibles con la tarea y la pérdida elegida.

En una capa densa cada neurona recibe todas las salidas de la capa anterior. Para un lote representado por filas:

$$
A^{(l)}=f\left(A^{(l-1)}W^{(l)}+b^{(l)}\right)
$$

Si entran `m` características y salen `n` neuronas, la capa tiene `m × n` pesos y `n` sesgos: **`(m + 1) × n` parámetros**.

Una red más grande tiene mayor capacidad, pero también puede sobreajustar, consumir más recursos y ser más difícil de optimizar. “Más capas” no significa automáticamente “mejor modelo”.

## 4. Cómo aprende una red neuronal

### 4.1 Propagación hacia adelante

Los datos pasan por las capas hasta producir una predicción. A continuación se compara esa predicción con la etiqueta verdadera mediante una función de pérdida.

### 4.2 Función de pérdida

La pérdida cuantifica el error que el optimizador intentará reducir.

| Tarea | Salida típica | Pérdida habitual | Representación de `y` |
|---|---|---|---|
| Regresión | Lineal | Error cuadrático medio (MSE) | Valores continuos |
| Clasificación binaria | Una sigmoide | Entropía cruzada binaria | `0` o `1` |
| Clasificación multiclase | Softmax con `K` salidas | Entropía cruzada categórica | Vectores one-hot |
| Clasificación multiclase | Softmax con `K` salidas | Entropía cruzada categórica dispersa | Enteros de `0` a `K - 1` |

Para regresión:

$$
\mathrm{MSE}=\frac{1}{n}\sum_i(y_i-\hat y_i)^2
$$

Para clasificación multiclase con etiquetas enteras:

$$
\mathcal{L}=-\frac{1}{n}\sum_i\log p_{i,y_i}
$$

Predecir la clase verdadera con baja probabilidad genera una pérdida alta. **La Práctica 8 utiliza entropía cruzada dispersa, no MSE.**

### 4.3 Backpropagation y actualización

La propagación hacia atrás aplica la regla de la cadena para calcular cuánto cambia la pérdida respecto de cada peso y sesgo. El **optimizador** utiliza esos gradientes para actualizar los parámetros.

Una actualización básica de descenso de gradiente es:

$$
w\leftarrow w-\eta\frac{\partial\mathcal{L}}{\partial w}
$$

`eta` es la tasa de aprendizaje. Un valor muy grande puede producir inestabilidad; uno muy pequeño puede hacer el entrenamiento lento.

**Distinción técnica:** backpropagation calcula gradientes; no es por sí sola la regla de actualización. Adam, utilizado en la práctica, mantiene estimaciones de primer y segundo momento de los gradientes para adaptar las actualizaciones.

### 4.4 Ejemplo manual de un gradiente

En una red simplificada sin activaciones, con `x=1`, `h=w1*x`, `y_pred=w2*h`, `w1=0.8`, `w2=0.4` y `y=0.5`:

$$
\hat y=0.32,\qquad \mathcal{L}=(0.32-0.5)^2=0.0324
$$

Para esta pérdida de una observación, sin factor `1/2`:

$$
\frac{\partial\mathcal{L}}{\partial w_2}=2(\hat y-y)h=-0.288
$$

$$
\frac{\partial\mathcal{L}}{\partial w_1}=2(\hat y-y)w_2x=-0.144
$$

Con tasa `0.1`, una actualización simultánea produce `w1=0.8144` y `w2=0.4288`. La nueva predicción es aproximadamente `0.3492`, más cercana a `0.5`.

Este ejemplo ilustra la regla de la cadena; no es la arquitectura ni el optimizador del notebook.

### 4.5 Épocas, lotes y validación

| Término | Significado |
|---|---|
| Época (*epoch*) | Un recorrido por los datos usados para actualizar pesos |
| Lote (*batch*) | Subconjunto procesado antes de una actualización |
| Iteración de entrenamiento | Un paso sobre un lote |
| Validación | Datos usados para observar generalización y decidir ajustes |
| Prueba (*test*) | Datos reservados para la evaluación final |

Validación y prueba no actualizan los pesos. Sin embargo, si se toman decisiones mirando repetidamente prueba, ese conjunto deja de ser una evaluación independiente.

## 5. El problema práctico: Forest CoverType

`fetch_covtype()` obtiene un conjunto con **581 012 observaciones, 54 características y 7 clases**. Cada observación representa una parcela de terreno de 30 × 30 metros.

Las características comprenden 10 variables cuantitativas, 4 indicadores de área silvestre y 40 indicadores de tipo de suelo. Los indicadores ya vienen codificados como columnas binarias.

| Etiqueta original | Tipo de cobertura |
|---|---|
| 1 | Spruce/Fir |
| 2 | Lodgepole Pine |
| 3 | Ponderosa Pine |
| 4 | Cottonwood/Willow |
| 5 | Aspen |
| 6 | Douglas-fir |
| 7 | Krummholz |

La práctica muestra un histograma y reconoce **desbalance de clases**. Una accuracy global alta puede ocultar bajo rendimiento en tipos poco frecuentes.

`data.target_names` contiene el nombre del objetivo, `Cover_Type`; no contiene los siete nombres de clases. Para revisar las etiquetas observadas se utiliza `np.unique(y)`.

## 6. Preparar el entorno

Desde la raíz del repositorio, una opción de entorno local es:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyter numpy matplotlib scikit-learn tensorflow
python -m jupyter notebook "Unidad 4/Practica8.ipynb"
```

Selecciona una versión de Python compatible con TensorFlow y tu plataforma. En Windows cambia la activación del entorno. También puedes utilizar Colab; revisa las versiones y el entorno de ejecución.

`fetch_covtype()` necesita acceso a Internet la primera vez, salvo que los datos ya estén en la caché de scikit-learn. El notebook no utiliza un CSV local para este conjunto.

La ejecución completa consume memoria y tiempo por el tamaño de los datos. GPU puede acelerar el entrenamiento, pero no es obligatoria. Los tiempos guardados en el notebook no garantizan los de otro equipo.

## 7. Práctica 8, paso a paso

### Paso 1 — Importar, cargar e inspeccionar

```python
data = fetch_covtype()
X = data.data
y = data.target
```

`X.shape` es `(581012, 54)` y `y.shape` es `(581012,)`. Revisa nombres, dimensiones y etiquetas antes de definir la salida de la red.

**Aprendizaje:** la cantidad de entradas depende de las características; la cantidad de salidas depende de las clases, no del número de observaciones.

### Paso 2 — Separar entrenamiento y prueba

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Quedan 464 809 observaciones en entrenamiento y 116 203 en prueba. `stratify=y` conserva aproximadamente las proporciones de clases en ambos conjuntos; **no balancea** las clases.

La semilla fija esta división, no toda la aleatoriedad de TensorFlow. Además, una división aleatoria no demuestra que el modelo generalice a otras regiones geográficas.

### Paso 3 — Estandarizar sin utilizar prueba

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

El escalador aprende estadísticas solo de entrenamiento y las aplica a prueba. Ajustarlo con todo `X` antes de dividir introduciría información de prueba en el preprocesamiento.

La práctica estandariza las 54 columnas, incluidos indicadores binarios. Es una decisión del notebook; no convierte esas variables en continuas de origen ni implica que sea la única estrategia válida.

Los mínimos y máximos impresos pueden ser negativos o superar uno: estandarizar no es normalizar al intervalo `[0,1]`.

### Paso 4 — Construir la arquitectura

```python
mimodelo = Sequential([
    Input(shape=(54,)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(7, activation="softmax"),
])
```

| Etapa | Salida por observación | Parámetros entrenables |
|---|---|---|
| Entrada | 54 valores | 0 |
| Densa 1 | 64 activaciones | `(54 + 1) × 64 = 3520` |
| Densa 2 | 32 activaciones | `(64 + 1) × 32 = 2080` |
| Densa 3 | 16 activaciones | `(32 + 1) × 16 = 528` |
| Salida | 7 probabilidades | `(16 + 1) × 7 = 119` |
| **Total** | — | **6247** |

La forma de entrada `(54,)` excluye la dimensión de lote, que Keras gestiona. `Sequential` expresa una secuencia de capas; `Dense` significa totalmente conectada. Como comprobación adicional puedes consultar `mimodelo.summary()`.

### Paso 5 — Compilar: definir cómo aprender

```python
mimodelo.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

Compilar configura optimización, pérdida y métricas; todavía no entrena. `accuracy` sirve para observar aciertos, pero el objetivo de optimización es la pérdida.

Se usa `sparse_categorical_crossentropy` porque las etiquetas son enteros. No se requiere convertirlas a one-hot.

### Paso 6 — Entrenar y adaptar etiquetas

```python
historia = mimodelo.fit(
    X_train,
    y_train - 1,
    epochs=20,
    batch_size=512,
    validation_split=0.1,
    verbose=1,
)
```

Las etiquetas originales son `1…7`, pero los índices de las salidas softmax son `0…6`. Restar uno al entrenar y evaluar mantiene la correspondencia correcta.

`validation_split=0.1` reserva el último 10 % de los arreglos recibidos, antes del barajado de entrenamiento; no realiza una nueva división estratificada. En este caso se reservan 46 481 observaciones y se actualizan pesos con las otras 418 328.

Con lotes de 512 hay 818 pasos por época, incluido el último lote parcial. Las 20 épocas corresponden a 20 recorridos por los datos efectivos de entrenamiento.

**Matiz del procedimiento original:** el escalador se ajustó antes de reservar esa validación, por lo que utilizó sus características para estimar medias y desviaciones. Prueba sigue separada, pero para una validación más estricta conviene crearla primero y ajustar el escalador solo con el subconjunto que actualiza pesos.

### Paso 7 — Interpretar las curvas

`historia.history["loss"]` y `historia.history["val_loss"]` guardan la pérdida por época.

| Patrón | Interpretación posible | Qué revisar |
|---|---|---|
| Ambas pérdidas bajan | El modelo está aprendiendo y puede generalizar mejor | Confirmar con métricas por clase |
| Entrenamiento baja y validación sube | Posible sobreajuste | Parada temprana, capacidad y regularización |
| Ambas quedan altas | Posible subajuste o mala configuración | Variables, etiquetas, arquitectura y optimización |
| Curvas inestables o `NaN` | Posible problema numérico o de datos | Valores no finitos, escalas y tasa de aprendizaje |

Estos patrones orientan el diagnóstico; no sustituyen una evaluación cuantitativa. No elijas más épocas únicamente porque mejora la pérdida de entrenamiento.

### Paso 8 — Aclaración sobre la celda `.score`

El notebook incluye una celda con `mimodelo.score`. Un modelo `Sequential` de Keras **no tiene el método `.score` propio de muchos estimadores de scikit-learn**. Ejecutar esa celda puede producir `AttributeError`.

Omite esa celda para continuar o sustitúyela, en tu copia de trabajo, por la evaluación correcta mostrada al final de la práctica:

```python
resultados = mimodelo.evaluate(X_test, y_test - 1)
```

Consultar un atributo tampoco equivale a ejecutar una evaluación. Este README documenta la limitación existente; no modifica el notebook.

### Paso 9 — Obtener probabilidades y clases

El notebook obtiene probabilidades:

```python
y_prob = mimodelo.predict(X_test)
```

Su forma es `(116203, 7)`. Cada fila contiene una probabilidad estimada por clase y suma aproximadamente uno.

**Ampliación para obtener etiquetas originales:**

```python
y_pred = np.argmax(y_prob, axis=1) + 1
```

`argmax` selecciona el índice de mayor probabilidad; sumar uno recupera los códigos `1…7`. No compares directamente índices `0…6` con `y_test`, que conserva `1…7`.

Una probabilidad alta no garantiza calibración ni corrección. Las redes también pueden equivocarse con mucha confianza.

### Paso 10 — Evaluar sin entrenar

El final de la práctica utiliza `evaluate` con `y_test - 1`. Para la configuración de pérdida y métrica del notebook, el resultado contiene primero pérdida y después accuracy.

Las **salidas guardadas** muestran aproximadamente:

| Medida | Valor |
|---|---|
| Entropía cruzada en prueba | `0.3725` |
| Accuracy en prueba | `0.8453`, equivalente a `84.53 %` |

Son resultados de una ejecución registrada, no una garantía de repetición ni una nueva ejecución realizada para esta guía. No representan precisión por clase y no demuestran rendimiento en otros bosques.

## 8. Evaluación más completa — ampliación propuesta

El notebook importa herramientas de matriz de confusión, pero no las utiliza en sus celdas finales. Tras obtener `y_pred` con las etiquetas originales, se puede añadir:

```python
from sklearn.metrics import classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, labels=np.arange(1, 8)))

matriz = confusion_matrix(y_test, y_pred, labels=np.arange(1, 8))
ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=np.arange(1, 8),
).plot()
plt.show()
```

En la matriz de scikit-learn, las filas son clases reales y las columnas, clases predichas. La diagonal indica aciertos.

| Métrica | Qué responde |
|---|---|
| Accuracy | ¿Qué proporción total se clasificó correctamente? |
| Precisión por clase | De los casos predichos como esa clase, ¿cuántos lo eran? |
| Recall por clase | De los casos reales de esa clase, ¿cuántos se reconocieron? |
| F1 por clase | ¿Cómo se equilibran precisión y recall? |
| F1 macro | ¿Cómo rinde el modelo dando igual importancia a cada clase? |
| F1 ponderado | ¿Cómo rinde ponderando por la frecuencia de cada clase? |

Si una clase no tiene predicciones, pueden aparecer advertencias por métricas indefinidas. Deben interpretarse como una señal de rendimiento insuficiente, no ocultarse.

Por el desbalance, conviene revisar recall y F1 macro además de accuracy. Comparar con una predicción constante de la clase mayoritaria ayuda a establecer una referencia mínima.

## 9. Mejoras con una justificación técnica

Estas propuestas **no forman parte de la ejecución original**:

| Mejora | Problema que atiende | Precaución |
|---|---|---|
| Crear validación explícita y estratificada antes del escalado | Separación más rigurosa y representación de clases | Ajustar el escalador solo con entrenamiento |
| `EarlyStopping` con `restore_best_weights=True` | Evitar continuar cuando validación deja de mejorar | Monitorizar una métrica de validación |
| Regularización L2 o Dropout | Reducir sobreajuste | Una intensidad excesiva puede causar subajuste |
| Ajustar neuronas, tasa de aprendizaje y lote | Buscar una configuración adecuada | Comparar con el mismo protocolo y presupuesto |
| Ponderación de clases | Dar más relevancia a clases minoritarias | Puede cambiar el equilibrio de errores y la calibración |
| Comparar con modelos de árboles | Establecer una referencia para datos tabulares | Usar la misma separación y métricas |
| Evaluación espacial o por región, si hay información suficiente | Medir transferencia a ubicaciones diferentes | No inferir ubicaciones independientes solo de una división aleatoria |

Para mejorar reproducibilidad, una ampliación es fijar `tf.keras.utils.set_random_seed(42)` antes de crear el modelo y registrar versiones y hardware. Según las operaciones y el dispositivo puede ser necesaria configuración adicional para determinismo.

Si el modelo va a reutilizarse, conserva **también** el escalador y el mapeo de etiquetas: una red guardada sin su preprocesamiento no representa todo el sistema de predicción.

## 10. Errores frecuentes y cómo resolverlos

| Situación | Causa o razonamiento | Acción |
|---|---|---|
| `ModuleNotFoundError: tensorflow` | El kernel no dispone de TensorFlow | Instalarlo en el entorno del kernel y comprobar compatibilidad |
| Falla la descarga de CoverType | No hay conexión o la descarga no está disponible | Revisar el error y la caché; no sustituir silenciosamente el conjunto |
| Error por etiquetas fuera del rango de la pérdida | Se pasaron valores `1…7` a una salida con índices `0…6` | Usar `y_train - 1` y `y_test - 1` |
| `AttributeError` al consultar `.score` | API de Keras distinta de scikit-learn | Utilizar `evaluate` |
| Rendimiento artificialmente alto | Posible fuga de información o evaluación no independiente | Revisar separaciones y dónde se ajusta el escalador |
| Accuracy alta pero bajo recall minoritario | Clases desbalanceadas | Revisar matriz y métricas por clase |
| Resultados distintos entre ejecuciones | Inicialización, orden de lotes o cómputo no determinista | Registrar semillas, entorno y variación |
| Memoria insuficiente | Arreglos grandes y copias de preprocesamiento | Revisar recursos; usar una muestra estratificada solo como experimento separado |

Reducir el conjunto o cambiar la arquitectura modifica el experimento. Si lo haces, documenta la diferencia y no presentes las métricas como equivalentes a las de la práctica completa.

## 11. Actividades para consolidar el aprendizaje

| Actividad | Aprendizaje esperado | Evidencia |
|---|---|---|
| Calcular una neurona a mano | Relacionar entradas, pesos, sesgo y activación | Operaciones y resultado |
| Explicar un paso de gradiente | Distinguir forward, pérdida, backpropagation y actualización | Derivadas del ejemplo simplificado |
| Inspeccionar la arquitectura | Entender formas y número de parámetros | `summary()` y cálculo de 6247 parámetros |
| Ejecutar la práctica | Conectar teoría y código | Curvas y evaluación de prueba |
| Construir matriz e informe | Identificar errores por clase | Recall, F1 y confusiones relevantes |
| Cambiar un hiperparámetro | Evaluar un cambio controlado | Comparación con el mismo conjunto de validación |
| Redactar limitaciones | Evitar conclusiones exageradas | Desbalance, reproducibilidad y alcance geográfico |

Preguntas de autoevaluación:

- ¿Por qué hay 54 entradas y 7 salidas?
- ¿Por qué ReLU se usa en capas ocultas y softmax en la salida?
- ¿Qué diferencia hay entre pérdida y accuracy?
- ¿Por qué la pérdida dispersa requiere etiquetas `0…6`?
- ¿Qué calcula backpropagation y qué hace Adam con ese resultado?
- ¿Por qué `stratify` no elimina el desbalance?
- ¿Cómo puede mejorar entrenamiento y empeorar validación?
- ¿Qué información debes guardar además de la red para predecir datos nuevos?

Un informe completo debe justificar la representación de datos, separación, escalado, arquitectura, pérdida, optimizador e hiperparámetros; presentar métricas y explicar qué clases fallan y en qué condiciones podría no generalizar.

## 12. Referencias de consulta

- [Guía de Sequential en TensorFlow/Keras](https://www.tensorflow.org/guide/keras/sequential_model)
- [Entrenamiento y evaluación en Keras](https://www.tensorflow.org/guide/keras/training_and_evaluation)
- [Capa Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
- [Entropía cruzada categórica dispersa](https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy)
- [Optimizador Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)
- [Datos CoverType en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html)
- [Métricas de clasificación](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
