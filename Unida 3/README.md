# Unidad 3 — Aplicaciones de Machine Learning en agrupación

Guía conceptual, técnica y práctica para comprender cómo se descubren grupos en datos sin etiquetas, cómo se selecciona un algoritmo de agrupación y cómo se interpretan sus resultados. El eje práctico es el **Modelo de Mezclas Gaussianas (GMM)** aplicado a Iris.

> La carpeta se llama `Unida 3` en el repositorio. Se conserva ese nombre para no alterar las rutas existentes.

## Contenido y ruta de aprendizaje

| Recurso | Contenido | Cómo utilizarlo |
|---|---|---|
| [Material de la unidad](Unid3_%20Aplicaciones%20de%20ML%20en%20Agrupaci%C3%B3n.pptx.pdf) | K-means, DBSCAN, HDBSCAN, gaussianas y algoritmo EM | Revisar la teoría y contrastarla con esta guía |
| [Práctica 7](Practica7.ipynb) | Agrupación probabilística de Iris con GMM | Ejecutar las celdas en orden y justificar cada paso |
| [Datos GPS](gps_real.csv) | Columnas `trip_id`, `step`, `lat` y `lon` | Recurso para una ampliación geoespacial; no lo utiliza la Práctica 7 |

Ruta sugerida: comprender qué significa similitud → revisar escalas y geometría → comparar algoritmos → entender GMM y EM → ejecutar Iris → analizar incertidumbre → experimentar y documentar conclusiones.

Al finalizar deberías poder explicar **por qué dos observaciones se agrupan**, qué supuestos impone el modelo y qué evidencia permite confiar —o no— en una segmentación.

## 1. Fundamentos: aprender sin etiquetas

En aprendizaje supervisado se dispone de una salida conocida `y` para enseñar al modelo. En agrupación o *clustering*, el entrenamiento busca estructura en las variables `X` sin utilizar una etiqueta objetivo.

Un **clúster** es un conjunto de observaciones similares según las variables, la métrica y el algoritmo elegidos. No es necesariamente una categoría verdadera ni una explicación causal.

En Iris existen etiquetas de especies, pero la práctica no las usa para ajustar el GMM. Consultar `iris.target_names` permite conocer el contexto; no convierte el entrenamiento en supervisado.

### Conceptos que debes distinguir

| Concepto | Significado | Consecuencia práctica |
|---|---|---|
| Observación | Una fila del conjunto de datos | En Iris, una flor |
| Característica | Una variable que describe la observación | Longitud y ancho del pétalo |
| Similitud | Cercanía definida mediante una representación y una métrica | Cambiar variables o escala puede cambiar los grupos |
| Centroide | Promedio de un grupo en K-means | Resume su ubicación, no es necesariamente una observación real |
| Densidad | Concentración de puntos en una región | Permite reconocer grupos y ruido |
| Asignación dura | Una etiqueta por observación | Es la salida habitual de K-means y de `GMM.predict` |
| Asignación suave | Probabilidades por componente | Es la salida de `GMM.predict_proba` |
| Ruido | Punto no incorporado a un grupo por un algoritmo de densidad | En DBSCAN suele representarse con `-1` |
| Hiperparámetro | Decisión configurada antes del ajuste | Número de grupos, radio o tipo de covarianza |
| Parámetro aprendido | Cantidad estimada a partir de los datos | Medias, covarianzas y pesos de GMM |

**Ejemplo lógico:** si agrupas clientes solo por gasto, descubres segmentos de gasto. Si añades frecuencia y antigüedad, cambia la definición de similitud. El algoritmo no decide por sí solo qué variables tienen sentido para el negocio.

## 2. Preparación de datos y distancia

Antes de agrupar, revisa tipos, valores faltantes, duplicados, distribuciones y posibles valores atípicos. Un identificador numérico no debe usarse automáticamente como característica: su magnitud puede no tener significado.

La distancia euclidiana entre dos observaciones es:

$$
d(x,z)=\sqrt{\sum_{j=1}^{d}(x_j-z_j)^2}
$$

Si una variable se mide en miles y otra en unidades, la primera puede dominar esa distancia. `StandardScaler` transforma cada columna mediante:

$$
x'_{ij}=\frac{x_{ij}-\mu_j}{s_j}
$$

Aquí `mu` y `s` son la media y la desviación estándar estimadas en los datos de ajuste. Estandarizar **no** limita los valores a `[0, 1]` y **no** transforma automáticamente una distribución en normal.

En la exploración de Iris se ajusta el escalador al conjunto completo porque se está describiendo ese conjunto. Si se quiere medir generalización en observaciones nuevas, hay que separar los datos y ajustar tanto el escalador como el modelo únicamente con entrenamiento.

## 3. Revisión de modelos

### 3.1 K-means: compactar grupos alrededor de centroides

K-means busca minimizar la suma de distancias **al cuadrado** dentro de los grupos:

$$
J=\sum_{k=1}^{K}\sum_{x_i\in C_k}\|x_i-\mu_k\|^2
$$

1. Elegir `K` e inicializar centroides.
2. Asignar cada punto al centroide más cercano.
3. Recalcular cada centroide como el promedio de sus puntos.
4. Repetir asignación y actualización hasta cumplir el criterio de parada.

Por ejemplo, para `(1,1)`, `(1,2)`, `(5,4)` y `(6,5)`, una separación natural en dos grupos produce centroides `(1,1.5)` y `(5.5,4.5)`.

La **inercia** corresponde al objetivo anterior. Al aumentar `K` normalmente disminuye, por lo que elegir el menor valor sin penalizar complejidad no es una buena estrategia. El método del codo busca una reducción marginal menos pronunciada, pero puede no mostrar un codo claro.

K-means es útil para perfiles compactos e interpretables. Es sensible a escalas, valores atípicos e inicialización, y suele representar mal grupos curvos o con dispersión muy diferente.

### 3.2 DBSCAN: encontrar regiones densas

DBSCAN utiliza `eps`, el radio de vecindad, y `min_samples`, la cantidad mínima de observaciones para considerar un punto núcleo. En scikit-learn este conteo **incluye al propio punto**.

| Tipo de punto | Condición |
|---|---|
| Núcleo (*core*) | Tiene al menos `min_samples` puntos en su vecindad |
| Borde (*border*) | No es núcleo, pero está dentro de la vecindad de un núcleo |
| Ruido (*noise*) | No es núcleo ni pertenece a un grupo por conexión con un núcleo |

El algoritmo expande grupos conectando puntos núcleo y añade los bordes alcanzables. No requiere especificar `K`.

Un `eps` muy pequeño puede generar mucho ruido; uno muy grande puede unir regiones distintas. Su valor depende de las **unidades y de la métrica**. No basta con copiar el mismo radio entre conjuntos de datos.

DBSCAN detecta formas arbitrarias y separa ruido, pero un único umbral de densidad puede resultar insuficiente cuando los grupos tienen densidades muy distintas.

### 3.3 HDBSCAN: jerarquía y estabilidad

HDBSCAN explora niveles de densidad, construye una jerarquía y selecciona grupos persistentes. `min_cluster_size` controla el tamaño mínimo de los grupos; `min_samples` influye en cuán conservadora es la selección.

Es útil cuando no hay un único radio satisfactorio para DBSCAN, pero no elimina la necesidad de revisar variables, métricas y parámetros.

**Precisión sobre la escala:** multiplicar todas las distancias por una constante puede conservar la estructura en configuraciones sin umbrales absolutos; escalar cada característica de manera diferente cambia la geometría. HDBSCAN no vuelve innecesario el preprocesamiento.

### 3.4 Comparación para elegir con criterio

| Algoritmo | Criterio principal | ¿Fijar número de grupos? | Tratamiento de incertidumbre o ruido | Uso típico |
|---|---|---|---|---|
| K-means | Cercanía a centroides | Sí | Asigna todos los puntos | Segmentos compactos |
| DBSCAN | Conectividad por densidad | No | Identifica ruido | Grupos irregulares con densidad compatible |
| HDBSCAN | Persistencia en una jerarquía de densidad | No | Identifica ruido; algunas implementaciones exponen pertenencia | Densidades variables |
| GMM | Densidad de una mezcla probabilística | Sí | Probabilidades por componente; no etiqueta ruido explícitamente | Grupos solapados o elípticos |

No hay un algoritmo universalmente mejor. La elección debe corresponder a la estructura buscada y a la utilidad de los grupos.

## 4. GMM: Modelo de Mezclas Gaussianas

### 4.1 Qué representa

Un GMM supone que los datos provienen de una combinación de `K` distribuciones normales:

$$
p(x)=\sum_{k=1}^{K}\pi_k\,\mathcal{N}(x\mid\mu_k,\Sigma_k)
$$

Cada componente tiene una media `mu_k`, una matriz de covarianza `Sigma_k` y un peso `pi_k`, con pesos no negativos que suman uno.

- **Media:** ubicación del componente.
- **Covarianza:** dispersión y relación lineal entre variables dentro del componente.
- **Peso:** proporción global estimada del componente en la mezcla.

Para `d` variables, la densidad normal multivariada es:

$$
\mathcal{N}(x\mid\mu,\Sigma)=
\frac{\exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)}
{(2\pi)^{d/2}|\Sigma|^{1/2}}
$$

El término cuadrático es la **distancia de Mahalanobis al cuadrado**. A diferencia de la euclidiana, considera dispersión y correlación. El determinante interviene en la normalización de la densidad.

La densidad `p(x)` no es la probabilidad de observar exactamente un punto continuo: las probabilidades de regiones se obtienen integrando la densidad. En cambio, la pertenencia a cada componente sí es una probabilidad discreta.

### 4.2 Responsabilidad local frente a peso global

La responsabilidad del componente `k` sobre el punto `i` es:

$$
r_{ik}=P(z_i=k\mid x_i)=
\frac{\pi_k\,\mathcal{N}(x_i\mid\mu_k,\Sigma_k)}
{\sum_{j=1}^{K}\pi_j\,\mathcal{N}(x_i\mid\mu_j,\Sigma_j)}
$$

Las responsabilidades de una observación suman uno. Un resultado `[0.31, 0.00, 0.69]` significa que el modelo asigna mayor pertenencia al componente `2`, aunque existe solapamiento con el componente `0`.

**No confundas** `weights_[k]`, que describe toda la mezcla, con `predict_proba(X)[i, k]`, que describe una observación concreta.

### 4.3 Cómo aprende: Expectation-Maximization

EM alterna dos operaciones:

1. **Inicialización:** establecer una estimación inicial de los componentes. `GaussianMixture` utiliza `init_params="kmeans"` por defecto.
2. **Paso E:** calcular responsabilidades utilizando los parámetros actuales.
3. **Paso M:** actualizar parámetros con promedios ponderados por esas responsabilidades.
4. **Convergencia:** repetir hasta satisfacer la tolerancia o alcanzar el máximo de iteraciones.

Para covarianzas completas, las actualizaciones conceptuales son:

$$
N_k=\sum_i r_{ik},\qquad
\pi_k=\frac{N_k}{n},\qquad
\mu_k=\frac{1}{N_k}\sum_i r_{ik}x_i
$$

$$
\Sigma_k=\frac{1}{N_k}\sum_i r_{ik}(x_i-\mu_k)(x_i-\mu_k)^T
$$

Así, un punto ambiguo contribuye a más de un componente, con distinta intensidad. La implementación añade regularización numérica a las covarianzas.

EM busca mejorar la verosimilitud, pero puede llegar a una solución local. Converger no garantiza recuperar las categorías verdaderas ni encontrar la mejor solución global.

### 4.4 Tipos de covarianza

| `covariance_type` | Supuesto | Forma de `covariances_` |
|---|---|---|
| `"full"` | Cada componente tiene su propia matriz completa | `(K, d, d)` |
| `"tied"` | Todos comparten una matriz completa | `(d, d)` |
| `"diag"` | Cada componente tiene varianzas propias, sin covarianzas cruzadas | `(K, d)` |
| `"spherical"` | Una varianza por componente, igual en todas sus dimensiones | `(K,)` |

`full` permite elipses orientadas y ofrece mayor flexibilidad, pero requiere más parámetros. `diag` y `spherical` simplifican el modelo; no deben elegirse únicamente por rapidez si contradicen la estructura de los datos.

## 5. Preparar y ejecutar la Práctica 7

Se necesita Python, Jupyter y las bibliotecas importadas por el notebook. Desde la raíz del repositorio, una opción de entorno local es:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyter numpy matplotlib scikit-learn
python -m jupyter notebook "Unida 3/Practica7.ipynb"
```

En Windows la activación del entorno cambia. También se puede abrir el archivo en Google Colab, revisando las dependencias disponibles. Iris se incluye en scikit-learn y no requiere el CSV GPS.

### Paso 1 — Cargar y seleccionar variables

```python
iris = load_iris()
X = iris.data[:, [2, 3]]
```

`X` tiene forma `(150, 2)`: 150 flores, longitud del pétalo y ancho del pétalo. La selección de dos columnas facilita la visualización, pero deja fuera la información del sépalo.

**Aprendizaje:** identifica qué significa cada eje antes de interpretar un gráfico.

### Paso 2 — Explorar la nube de puntos

El primer `scatter` usa medidas originales en centímetros. Permite reconocer separación, solapamiento y posibles observaciones alejadas; todavía no hay asignaciones del modelo.

**Aprendizaje:** una separación visual en dos dimensiones no demuestra que exista la misma estructura en todas las variables.

### Paso 3 — Estandarizar

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

`fit` estima media y desviación; `transform` aplica la transformación. Conserva el escalador para transformar cualquier dato nuevo del mismo modo.

### Paso 4 — Crear y ajustar el modelo

```python
miGMM = GaussianMixture(
    n_components=3,
    random_state=42,
    covariance_type="full",
    n_init=1,
)
miGMM.fit(X_scaled)
```

Se fijan tres componentes como ejercicio. Saber que Iris tiene tres especies no demuestra que tres sea el mejor número de componentes para cualquier representación.

`random_state=42` controla la aleatoriedad del ajuste. `n_init=1` realiza una inicialización; aumentar este valor permite comparar varios ajustes y conservar el de mejor criterio de verosimilitud, a costa de más cómputo.

### Paso 5 — Revisar convergencia

El notebook consulta `miGMM.converged_` y conserva un resultado `True`.

**Aclaración:** aunque el comentario de esa celda menciona el número de iteraciones, `converged_` es un booleano. Para ver cuántas iteraciones se ejecutaron, consulta `miGMM.n_iter_`.

Si aparece una advertencia de convergencia, revisa inicialización, escalas, `max_iter`, `tol` y posibles covarianzas casi singulares; no ignores la advertencia.

### Paso 6 — Asignar grupos y visualizar

```python
labels = miGMM.predict(X_scaled)
```

La salida tiene forma `(150,)`. Los valores `0`, `1` y `2` son identificadores arbitrarios de componentes, no códigos garantizados de especies.

El segundo gráfico usa variables estandarizadas. Sus ejes ya no están en centímetros, aunque los nombres indiquen longitud y ancho del pétalo.

### Paso 7 — Interpretar probabilidades y ambigüedad

```python
probabilidades = miGMM.predict_proba(X_scaled)
confidence = probabilidades.max(axis=1)
```

`probabilidades` tiene forma `(150, 3)` y cada fila suma aproximadamente uno. `confidence` conserva la mayor pertenencia de cada flor; ayuda a localizar zonas ambiguas.

En las salidas guardadas, la observación de índice `83` muestra aproximadamente `[31, 0, 69]` por ciento y se asigna al componente `2`. Los índices de NumPy empiezan en cero.

El notebook también busca una pertenencia redondeada al 46 %. Eso ilustra cómo inspeccionar puntos, pero no es una regla general de incertidumbre. Para analizarla, utiliza las probabilidades sin redondear y un criterio explícito.

**Límite importante:** una pertenencia máxima alta no garantiza que el punto sea típico ni que la asignación sea correcta. Un punto muy alejado puede tener alta pertenencia relativa si todos los componentes le asignan densidades pequeñas.

### Paso 8 — Leer los parámetros aprendidos

| Atributo | Qué revisar en esta práctica |
|---|---|
| `means_` | Tres centros de dos dimensiones, en escala estandarizada |
| `covariances_` | Tres matrices `2 × 2`; diagonales para varianzas y elementos cruzados para covarianzas |
| `weights_` | Tres proporciones globales que suman uno |

Para interpretar los centros en centímetros, una ampliación útil es `scaler.inverse_transform(miGMM.means_)`. Las covarianzas no se convierten mediante esa misma llamada: requieren transformar ambas dimensiones de la matriz.

## 6. Evaluación y experimentación — ampliación propuesta

Esta sección propone actividades **no implementadas en el notebook original**.

### Comparar cantidad de componentes y complejidad

GMM permite calcular:

$$
\mathrm{AIC}=2p-2\log L,\qquad
\mathrm{BIC}=p\log n-2\log L
$$

`p` es el número de parámetros libres, `n` el número de observaciones y `L` la verosimilitud. **Menor AIC o BIC es preferible entre los candidatos comparados** sobre los mismos datos y la misma representación. No son porcentajes de acierto.

Una prueba guiada consiste en comparar `n_components` de 1 a 6 y los cuatro tipos de covarianza, registrando AIC, BIC, convergencia y estabilidad entre inicializaciones.

### Medir separación sin etiquetas

La silueta compara cohesión interna y separación respecto del grupo vecino; toma valores entre `-1` y `1`. Solo está definida cuando hay entre 2 y `n - 1` etiquetas distintas.

Para DBSCAN/HDBSCAN debes decidir y declarar si excluyes ruido antes de calcularla. La silueta no mide directamente el ajuste probabilístico de GMM y puede favorecer geometrías compactas.

### Contrastar con especies conocidas

Como análisis externo, se pueden comparar agrupaciones con `iris.target` mediante **Adjusted Rand Index (ARI)** o información mutua ajustada. ARI no necesita que los identificadores de grupos coincidan con los de las especies.

No calcules `accuracy_score(iris.target, labels)` directamente: una permutación de etiquetas puede dar un resultado engañoso. Tampoco selecciones el modelo con las especies si deseas conservar una evaluación realmente independiente.

### Extensión con GPS

El CSV incluye coordenadas y secuencias por viaje. Antes de experimentar:

1. Verifica rangos de latitud y longitud, faltantes y orden por `trip_id` y `step`.
2. Define si quieres agrupar **puntos**, zonas de permanencia o **trayectorias**; son problemas diferentes.
3. Para distancias físicas, usa una proyección métrica apropiada o una métrica geográfica. En DBSCAN con `metric="haversine"`, las coordenadas deben estar en radianes y `eps` expresa distancia angular, no metros.
4. Evita utilizar `trip_id` como distancia numérica. `step` describe orden, pero no garantiza tiempos ni velocidades.
5. Si trabajas con puntos individuales, no concluyas automáticamente que has agrupado rutas completas.

## 7. Errores frecuentes y razonamiento correcto

| Error | Cómo evitarlo |
|---|---|
| Interpretar clúster `0` como una especie fija | Tratar los identificadores como arbitrarios |
| Afirmar que convergencia implica modelo óptimo | Comparar criterios, estabilidad y utilidad |
| Creer que GMM detecta ruido como DBSCAN | Separar pertenencia relativa de densidad; GMM asigna todos los puntos |
| Confundir pesos con probabilidades individuales | Distinguir `weights_` de responsabilidades |
| Elegir `K` solo porque hay `K` etiquetas conocidas | Justificar la selección y declarar el uso de información externa |
| Aplicar `fit_transform` a cada conjunto nuevo | Reutilizar `transform` del escalador ajustado |
| Concluir que el redondeo al 100 % elimina incertidumbre | Examinar valores originales y supuestos del modelo |
| Afirmar que HDBSCAN es inmune a cualquier cambio de escala | Revisar cómo se modifican las distancias entre características |

## 8. Actividades de aprendizaje y evidencia

| Actividad | Pregunta que debes responder | Evidencia sugerida |
|---|---|---|
| Ejecutar la práctica original | ¿Qué hacen carga, escalado, ajuste y predicción? | Formas de arreglos y gráficos comentados |
| Comparar `full` y `spherical` | ¿Cómo cambia la geometría permitida? | Covarianzas y asignaciones |
| Variar `n_init` | ¿La solución es estable? | Criterios y convergencia de los ajustes |
| Revisar puntos ambiguos | ¿Qué implica una pertenencia repartida? | Probabilidades y ubicación en el gráfico |
| Usar las cuatro variables de Iris | ¿Qué información se perdía al usar dos? | Comparación de criterios sobre cada representación, sin mezclar sus escalas |
| Proponer un uso real | ¿Qué decisión permite tomar cada grupo? | Interpretación, limitaciones y riesgos |

Preguntas de autoevaluación:

- ¿Por qué `predict` devuelve una etiqueta si GMM es probabilístico?
- ¿Cómo participa un punto ambiguo en el paso M de EM?
- ¿Qué diferencia hay entre covarianza diagonal y esférica?
- ¿Por qué una alta pertenencia no demuestra ausencia de anomalías?
- ¿Cuándo elegirías DBSCAN en lugar de GMM?

Una conclusión técnica completa debe indicar variables, preprocesamiento, algoritmo, hiperparámetros, convergencia, evidencia de calidad e interpretación. Evita limitar el informe a “se obtuvieron tres grupos”.

## 9. Referencias de consulta

- [Guía de agrupación de scikit-learn](https://scikit-learn.org/stable/modules/clustering.html)
- [Modelos de mezclas gaussianas](https://scikit-learn.org/stable/modules/mixture.html)
- [API de GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
- [API de StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [API de DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [API de HDBSCAN en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)

**Conexión con la siguiente unidad:** aquí se estiman grupos y densidades sin utilizar etiquetas; en la [Unidad 4](../Unidad%204/README.md) se entrena una red neuronal para predecir una clase conocida.
