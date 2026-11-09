# Actividad Integradora 3: construcción educativa desde cero

Esta guía explica **cómo, por qué y para qué se construye cada celda** de [integradora_3.ipynb](integradora_3.ipynb). No se limita a describir funciones: relaciona los datos, las decisiones metodológicas, las dependencias entre celdas y la interpretación de los resultados.

El proyecto resuelve una **clasificación binaria de ingresos** mediante un único algoritmo, Random Forest. Compara tres búsquedas de hiperparámetros, entrena un modelo nuevo y explica sus decisiones con técnicas de inteligencia artificial explicativa (XAI).

**Archivos de la actividad:**

- [Notebook ejecutado](integradora_3.ipynb): código, tablas, gráficas e interpretaciones.
- [Copia corregida y ejecutada](integradora_3%20%28copia%29.ipynb): instalación restaurada y ejecución completa validada en Python 3.12 local. La interrupción guardada en Colab ocurrió durante la importación de seaborn, antes del modelado; no se verificó nuevamente ese entorno Python 3.13.
- [Reporte técnico](Reporte_Actividad_Integradora_3.docx): código explicado línea por línea, resultados e interpretación, con presentación adaptada a la guía APA 7.
- [Instrucciones](intrucciones.txt): requisitos y entregables.
- Este README: recorrido de aprendizaje para reconstruir el notebook.

## Verificación del reporte

Se contrastó el reporte con las salidas guardadas del notebook, no con cifras estimadas ni con otra versión del dataset:

| Evidencia | Verificación |
|---|---|
| Código | Las 196 instrucciones no vacías aparecen en el reporte junto con sus comentarios explicativos. |
| Resultados | Coinciden 406 líneas y valores revisados de consola, tablas e interpretaciones numéricas. |
| Figuras | Las ocho imágenes del reporte son idénticas a las del notebook. |
| Modelo seleccionado | Randomized Search, por empate de score y menor tiempo observado. |
| Evaluación final | Accuracy 0.8045; recall positivo 0.8376; ROC-AUC 0.9044. |
| Caso nuevo | Perfil hipotético, sin etiqueta real, con P(>50K) = 0.8083. |

**El reporte sí muestra tanto el código como los resultados obtenidos.** Las instrucciones y sus comentarios se presentan en columnas separadas; las salidas incluyen tablas, texto y figuras. La tabla comparativa del notebook abrevia algunos diccionarios de parámetros: el reporte muestra completa la configuración ganadora y no inventa los valores abreviados de las demás búsquedas.

Los números de esta guía corresponden a la ejecución guardada del **7 de noviembre de 2026**. Son una referencia para comprenderla, no valores que deban escribirse manualmente en el programa. Los resultados se calculan al ejecutar; los tiempos dependen del equipo y su carga.

## 1. Qué se debe comprender antes de programar

### El problema

Cada fila describe un registro del censo estadounidense de 1994. Se intenta predecir si su ingreso anual pertenece a una de estas clases:

| Etiqueta original | Valor utilizado por el modelo | Significado |
|---|---|---|
| `<=50K` | `0` | Ingreso anual menor o igual a USD 50 000. |
| `>50K` | `1` | Ingreso anual mayor a USD 50 000; clase positiva de interés. |

**Clasificar** significa asignar una clase. **Estimar una probabilidad** significa obtener un valor entre 0 y 1 que el modelo utiliza para decidir. Esa probabilidad no se considera calibrada ni garantiza cuál es el ingreso real.

### Conceptos que se usarán

| Concepto | Uso en este trabajo |
|---|---|
| Dataset | Tabla original que se descarga de OpenML. |
| Predictor o feature | Columna utilizada para explicar o predecir el ingreso. |
| Target | Columna `income`, que contiene la clase a predecir. |
| EDA | Análisis exploratorio: conocer calidad, distribuciones, extremos y asociaciones. |
| Entrenamiento, `train` | Datos con los que se aprenden transformaciones y árboles. |
| Prueba, `test` | Datos reservados para estimar desempeño final. |
| Validación cruzada, CV | División de train en folds para comparar configuraciones sin consultar test. |
| Pipeline | Cadena que aprende preprocesamiento y modelo dentro de cada ajuste. |
| Parámetro aprendido | Estado que el algoritmo obtiene al entrenar, como los cortes de sus árboles. |
| Hiperparámetro | Configuración elegida antes de entrenar, como profundidad o cantidad de árboles. |
| Fuga de información | Usar información de validación o test para aprender algo que debería ajustarse solo con train. |
| XAI global | Explicación de la dependencia del modelo respecto de variables en un conjunto de casos. |
| XAI local | Explicación de una predicción concreta. |

### Qué dataset se utiliza realmente

La descarga es **Adult, versión 2 de OpenML, `data_id=1590`**. La tabla original contiene **48 842 filas y 15 columnas**, contando el target: 14 atributos más la etiqueta.

No todos esos registros se usan para entrenar:

| Etapa | Filas | Propósito |
|---|---:|---|
| Descarga y EDA | 48 842 | Diagnosticar el dataset original completo. |
| Selección de atributos y deduplicación | 40 382 | Conservar una copia de cada combinación idéntica de predictores y etiqueta. |
| Muestra para modelado | 8 000 | Limitar el costo del experimento y conservar aproximadamente la proporción de clases. |
| Entrenamiento | 6 000 | Buscar hiperparámetros y ajustar el modelo final. |
| Prueba | 2 000 | Medir desempeño final y realizar explicaciones diagnósticas. |
| Muestra SHAP global | 200 | Explicar una parte reproducible de test sin recalcular el entrenamiento. |
| Predicción nueva | 1 perfil hipotético | Mostrar inferencia y explicación local fuera del dataset. |

Los **8 460 duplicados** se detectan después de excluir atributos, considerando juntos predictores y etiqueta. No son el mismo conteo que los **52 duplicados completos** encontrados en el EDA, que utiliza todas las columnas originales.

Adult ya se utilizó en las prácticas. Su elección permite reutilizar y comprender el EDA previo, pero no satisface la recomendación de usar un dataset inédito. Las instrucciones permiten libre elección y no lo prohíben.

## 2. Cómo empezar en Google Colab

1. Crear un notebook nuevo en Google Colab y nombrarlo `integradora_3.ipynb`.
2. Seleccionar un entorno de ejecución de **CPU**. Este trabajo no necesita GPU.
3. Crear las celdas de texto y código en el orden indicado a continuación.
4. Copiar cada bloque de código a una celda de **código**. Las explicaciones de esta guía sirven para redactar las celdas de **texto**.
5. Ejecutar de arriba hacia abajo. No es necesario montar Drive, subir un CSV ni crear `herramientas_eda.py`.
6. Esperar a que la instalación termine sin errores, reiniciar la sesión o el kernel y continuar desde la celda de importaciones. No reinstalar paquetes mientras se ejecutan otras celdas. Si se interrumpe una importación, reiniciar antes de reintentar.
7. Guardar el notebook después de la ejecución completa, incluyendo sus salidas.

Una celda de **texto o Markdown** documenta decisiones; no entrena modelos. Una celda de **código** ejecuta Python y crea objetos que otras celdas utilizarán. `%pip` es una instrucción del entorno Jupyter/Colab: no se ejecuta directamente en un archivo `.py` común.

**Regla de reconstrucción:** no reemplazar los cálculos por los resultados impresos en esta guía. Las cifras explican una ejecución; el código debe volver a producir sus propios resultados.

## 3. Mapa de las 28 celdas

La numeración siguiente empieza en **1** y cuenta también las celdas Markdown. Coincide con la numeración utilizada para describir celdas en el reporte.

| Celda | Tipo | Función |
|---:|---|---|
| 1 | Texto | Definir problema, objetivo, datos, referencias y límites. |
| 2 | Texto | Justificar dependencias y reproducibilidad. |
| 3 | Código | Instalar las dependencias. |
| 4 | Código | Importar herramientas y establecer constantes. |
| 5 | Texto | Explicar descarga y normalización. |
| 6 | Código | Descargar Adult y validar etiquetas. |
| 7 | Texto | Justificar el EDA automatizado reutilizado. |
| 8 | Código | Definir y ejecutar `AutoEDA`. |
| 9 | Código | Interpretar los hallazgos del EDA. |
| 10 | Texto | Explicar atributos, deduplicación, partición y pipeline. |
| 11 | Código | Preparar train/test y definir el preprocesador. |
| 12 | Texto | Justificar modelo, métrica y búsquedas. |
| 13 | Código | Ejecutar Grid, Randomized y BayesSearchCV. |
| 14 | Texto | Explicar cómo se elige el ganador. |
| 15 | Código | Comparar resultados y seleccionar configuración. |
| 16 | Texto | Justificar evaluación final, matriz y ROC. |
| 17 | Código | Entrenar un modelo nuevo y evaluar test. |
| 18 | Texto | Explicar Feature Importance. |
| 19 | Código | Obtener importancia interna del bosque. |
| 20 | Texto | Explicar Permutation Importance. |
| 21 | Código | Medir caída de score por predictor. |
| 22 | Texto | Explicar SHAP global y sus unidades. |
| 23 | Código | Calcular y contrastar SHAP global. |
| 24 | Texto | Justificar el caso nuevo y SHAP local. |
| 25 | Código | Predecir un perfil externo y explicar su salida. |
| 26 | Texto | Plantear conclusiones y alcance. |
| 27 | Código | Redactar una conclusión con métricas calculadas. |
| 28 | Texto | Revisar cumplimiento y entregables. |

## 4. Construcción paso a paso

### Celda 1 — Presentación del problema

**Tipo:** texto.

**Qué escribir:** el objetivo de predecir `income`, la clase positiva, el origen de Adult, el uso de un único Random Forest, las tres búsquedas y las tres técnicas XAI.

**Por qué se necesita:** antes de seleccionar herramientas debe quedar claro qué se predice y cómo se evaluará. Sin esa definición, una métrica alta puede no responder a la pregunta de interés.

**Para qué sirve:** establece el contrato del experimento y sus límites. Incluye que los datos son de 1994, que el uso es académico y que excluir raza/sexo no demuestra equidad. También conserva la palabra `elefante` exigida por la consigna.

**Relación con las unidades:** la Unidad 1 aporta EDA e interpretabilidad; la Unidad 2, clasificación y búsquedas. La práctica de GMM de Unida 3 y la red neuronal de Unidad 4 se revisan como contexto, pero no se ejecutan: añadirlas introduciría tareas o modelos no necesarios.

### Celda 2 — Dependencias y reproducibilidad

**Tipo:** texto.

**Qué explicar:** se fijan versiones para utilizar APIs compatibles y se limita el paralelismo. No se instalan paquetes que el trabajo no usa.

**Por qué:** cambios de versiones pueden alterar interfaces o resultados; instalar librerías innecesarias aumenta costo y riesgo de conflictos.

**Para qué:** preparar una ejecución portátil y comprensible. El trabajo no requiere TensorFlow, XGBoost, Optuna ni `statsmodels`.

### Celda 3 — Instalar las herramientas

**Tipo:** código. **Requisito:** disponer de un kernel de Colab/Jupyter con acceso a internet.

```python
%pip install numpy==2.1.3 pandas==2.2.3 matplotlib==3.10.0 seaborn==0.13.2 scikit-learn==1.6.1 scikit-optimize==0.10.2 shap==0.48.0 # Instala todas las dependencias y muestra el progreso y los errores.
```

**Cómo funciona:** `%pip` instala en el intérprete del notebook; `==` fija la versión. Se conserva la salida de instalación para ver progreso y errores. No sustituir esta celda por una reinstalación aislada de `python-dateutil`: eso no prepara todas las dependencias.

**Para qué sirve cada familia:** NumPy y pandas manejan datos; matplotlib y seaborn dibujan; scikit-learn prepara, entrena y evalúa; scikit-optimize realiza la búsqueda bayesiana; SHAP explica predicciones.

**Resultado esperado:** instalación sin errores. Un aviso de posible reinicio no equivale a un error del modelo.

### Celda 4 — Importaciones, semilla y presupuesto

**Tipo:** código. **Depende de:** celda 3.

```python
import time  # Permite medir el tiempo real transcurrido de cada búsqueda.
import numpy as np  # Proporciona operaciones numéricas y comprobaciones de consistencia.
import pandas as pd  # Maneja las tablas de datos y de resultados.
import matplotlib.pyplot as plt  # Dibuja y muestra las figuras.
print("Cargando seaborn; espera a que termine la importación...")  # Identifica la importación que fue interrumpida en la copia.
import seaborn as sns  # Proporciona el mapa de correlaciones y el estilo estadístico.
import sklearn  # Permite registrar la versión efectiva de scikit-learn.
print("Cargando SHAP; la primera importación puede tardar...")  # Hace visible el avance antes de importar la herramienta XAI.
import shap  # Calcula contribuciones globales y locales del modelo.
from IPython.display import display, Markdown  # Muestra tablas e interpretaciones calculadas en el notebook.
from sklearn.datasets import fetch_openml  # Descarga el dataset público sin archivos locales.
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV  # Separa datos y configura CV y dos búsquedas.
from sklearn.compose import ColumnTransformer  # Aplica transformaciones distintas a columnas numéricas y categóricas.
from sklearn.impute import SimpleImputer  # Aprende valores de sustitución para los faltantes.
from sklearn.preprocessing import OneHotEncoder  # Convierte categorías en indicadores numéricos.
from sklearn.pipeline import Pipeline  # Encadena preparación y modelo dentro de cada fold.
from sklearn.ensemble import RandomForestClassifier  # Implementa el único clasificador del trabajo.
from sklearn.metrics import accuracy_score, recall_score, make_scorer, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay  # Evalúa selección, clasificación y discriminación.
from sklearn.inspection import permutation_importance  # Mide la pérdida de score al alterar cada predictor.
from sklearn.base import clone  # Crea un modelo final nuevo sin conservar ajustes previos.
from skopt import BayesSearchCV  # Implementa la búsqueda guiada por un modelo sustituto.
from skopt.space import Categorical  # Define las mismas opciones discretas para la búsqueda bayesiana.
SEMILLA = 42  # Fija las decisiones aleatorias de muestreo, CV, modelo y búsquedas.
MAX_FILAS = 8000  # Limita el costo de modelado en una sesión de Colab de CPU.
N_ITER = 12  # Asigna el mismo presupuesto a Randomized Search y BayesSearchCV.
sns.set_theme(style="whitegrid")  # Usa un estilo común y legible para las figuras.
print(f"numpy={np.__version__} | pandas={pd.__version__} | sklearn={sklearn.__version__} | SHAP={shap.__version__}")  # Registra las versiones efectivas.
```

**Cómo funciona:** `import` incorpora módulos al entorno; `from ... import ...` trae herramientas concretas. Las constantes quedan disponibles para las celdas posteriores.

**Por qué se fijan estas constantes:**

| Constante | Decisión educativa |
|---|---|
| `SEMILLA = 42` | Repetir muestreo, partición, folds y elecciones aleatorias en condiciones equivalentes. |
| `MAX_FILAS = 8000` | Mantener razonable el costo de búsquedas y explicaciones en CPU. |
| `N_ITER = 12` | Dar igual cantidad de propuestas a Randomized Search y BayesSearchCV. |

**Resultado esperado:** se imprimen las versiones efectivas. La semilla no fija el tiempo de ejecución, la carga del equipo ni posibles diferencias entre versiones.

**Detalle importante:** el número de procesos del bosque se configura más adelante en `1`, mientras que cada buscador usa hasta `2`. Así se evita paralelismo anidado excesivo.

### Celda 5 — Plan de descarga y limpieza

**Tipo:** texto.

**Qué explicar:** descargar por identificador estable, unificar texto y comprobar que las clases son válidas.

**Por qué:** un espacio, un `?` o un punto final puede representar el mismo dato de forma diferente. Normalizar no significa imputar ni entrenar.

**Para qué:** obtener una tabla consistente sin usar medianas, modas ni información de test antes de la partición.

### Celda 6 — Descargar y normalizar Adult

**Tipo:** código. **Depende de:** celda 4.

```python
TARGET = "income"  # Define un nombre único para la variable objetivo.
datos_openml = fetch_openml(data_id=1590, as_frame=True)  # Obtiene Adult versión 2 como tabla, con sus tipos originales.
df = datos_openml.frame.rename(columns={"class": TARGET}).copy()  # Separa una copia de trabajo y normaliza el nombre del target.
for columna in df.select_dtypes(exclude=np.number).columns:  # Recorre únicamente variables de texto o categorías.
    df[columna] = df[columna].astype(object).str.strip().replace("?", np.nan)  # Quita espacios y representa faltantes con NaN compatible con el imputador.
df[TARGET] = df[TARGET].str.rstrip(".")  # Unifica las variantes del target que terminan en punto.
if df[TARGET].isna().any() or set(df[TARGET].unique()) != {"<=50K", ">50K"}:  # Comprueba que no haya etiquetas ausentes ni desconocidas.
    raise ValueError("Adult debe contener únicamente etiquetas válidas <=50K y >50K.")  # Expone un problema del origen de datos sin ocultarlo.
print(f"Dataset completo: {df.shape[0]:,} filas y {df.shape[1]} columnas.")  # Informa el tamaño real descargado.
display(df.head())  # Presenta ejemplos para verificar nombres, tipos y valores.
```

**Cómo funciona:**

1. `fetch_openml(data_id=1590, as_frame=True)` obtiene el dataset como DataFrame.
2. `rename` cambia `class` a `income`; `copy` crea una tabla de trabajo.
3. Las columnas no numéricas se convierten a objetos de texto, se limpian con `strip` y sus `?` se convierten en `NaN`.
4. `rstrip(".")` elimina un punto final de las etiquetas si aparece.
5. La comprobación exige ambas clases documentadas y ausencia de etiquetas nulas.

**Por qué se detiene ante etiquetas inesperadas:** eliminar filas silenciosamente podría ocultar una descarga equivocada o una transformación incorrecta. El error obliga a revisar el origen.

**Resultado esperado:** **48 842 filas y 15 columnas**, junto con cinco ejemplos. Esta celda no excluye atributos ni elimina filas por faltantes: el EDA debe poder observarlos.

**Objeto que deja preparado:** `df`, el dataset original normalizado.

### Celda 7 — Diseño del EDA reutilizado

**Tipo:** texto.

**Qué explicar:** `AutoEDA` procede de `Actividad Integradora 1/herramientas_EDA_(3).ipynb`. Se conserva su estructura de diagnóstico y se adapta al target categórico.

**Por qué:** la consigna exige reutilizar el EDA previo. Incluir la clase directamente permite ejecutar en Colab sin escribir ni importar un módulo local.

**Para qué:** conocer los datos antes de modelar. Un diagnóstico no debe eliminar, imputar ni transformar automáticamente observaciones.

### Celda 8 — Crear y ejecutar `AutoEDA`

**Tipo:** código. **Depende de:** `df` de la celda 6.

```python
class AutoEDA:  # Reutiliza la interfaz y los métodos de la clase de la Integradora 1, adaptados a clasificación.
    def __init__(self, df: pd.DataFrame, target_col: str):  # Recibe la tabla y el nombre del target categórico.
        self.df = df  # Conserva una referencia para diagnosticar sin modificar los datos.
        self.target_col = target_col  # Identifica la variable que se analizará como clase.
        self.num_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # Identifica columnas numéricas por su tipo.
        self.cat_cols = df.select_dtypes(exclude=np.number).columns.drop(target_col).tolist()  # Identifica categorías sin incluir el target.

    def _diagnose_missing_and_duplicates(self):  # Reutiliza el diagnóstico de nulos y duplicados de AutoEDA.
        self.calidad = pd.DataFrame({"Tipo": self.df.dtypes.astype(str), "Valores distintos": self.df.nunique(), "Nulos": self.df.isna().sum()})  # Consolida la calidad por columna sin repetir tablas.
        self.calidad["Nulos (%)"] = 100 * self.calidad["Nulos"] / len(self.df)  # Expresa los faltantes como proporción del dataset.
        display(self.calidad)  # Muestra tipos, cardinalidad y ausencia de datos.
        print(f"Filas completamente duplicadas: {self.df.duplicated().sum():,}")  # Cuenta duplicados conservando todos los atributos originales.

    def _summary_statistics(self):  # Reutiliza el resumen descriptivo de las variables numéricas.
        display(self.df[self.num_cols].describe().T)  # Muestra conteo, media, dispersión y cuantiles en una sola tabla.
        self.df[self.num_cols].hist(bins=25, figsize=(12, 8))  # Visualiza las distribuciones y concentraciones numéricas.
        plt.tight_layout()  # Evita que se superpongan los títulos de los histogramas.
        plt.show()  # Presenta la figura de distribuciones.

    def _detect_outliers(self):  # Reutiliza la regla IQR de la clase original como diagnóstico, no como filtro.
        numericas = self.df[self.num_cols]  # Selecciona los atributos numéricos para calcular sus cuantiles.
        q1 = numericas.quantile(0.25)  # Calcula el primer cuartil por variable.
        q3 = numericas.quantile(0.75)  # Calcula el tercer cuartil por variable.
        iqr = q3 - q1  # Obtiene el rango intercuartílico de cada distribución.
        extremos = ((numericas < q1 - 1.5 * iqr) | (numericas > q3 + 1.5 * iqr)).sum()  # Cuenta valores fuera de las cercas de Tukey.
        self.outliers = extremos.rename("Extremos IQR").to_frame()  # Guarda los conteos para su interpretación posterior.
        display(self.outliers)  # Muestra alertas sin eliminar ni transformar observaciones.

    def _correlation_analysis(self):  # Reutiliza Pearson para explorar relaciones entre atributos numéricos.
        self.correlacion = self.df[self.num_cols].corr()  # Calcula una sola matriz de correlación.
        plt.figure(figsize=(8, 6))  # Reserva una figura legible para el mapa.
        sns.heatmap(self.correlacion, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)  # Representa fuerza y dirección de las asociaciones lineales.
        plt.title("Correlación de Pearson entre variables numéricas")  # Aclara que el target categórico no participa en Pearson.
        plt.tight_layout()  # Ajusta etiquetas y márgenes.
        plt.show()  # Presenta el mapa de correlación.

    def _classification_analysis(self):  # Agrega el diagnóstico de clases y categorías necesario para este problema.
        self.balance = self.df[self.target_col].value_counts().rename("Casos").to_frame()  # Cuenta la frecuencia de cada clase.
        self.balance["Proporción"] = self.balance["Casos"] / len(self.df)  # Calcula el peso relativo de cada etiqueta.
        display(self.balance)  # Expone el desbalance que justifica estratificación y recall.
        for columna in self.cat_cols:  # Recorre los predictores categóricos del EDA de la Unidad 1.
            frecuencias = pd.crosstab(self.df[columna], self.df[self.target_col])  # Cuenta ambas clases para cada categoría, excluyendo nulos ya diagnosticados.
            frecuencias["Casos"] = frecuencias.sum(axis=1)  # Registra el soporte para evitar interpretar tasas sin contexto.
            frecuencias["Proporción >50K"] = frecuencias[">50K"] / frecuencias["Casos"]  # Mide la asociación descriptiva de cada categoría con la clase positiva.
            display(frecuencias[["Casos", "Proporción >50K"]].sort_values("Casos", ascending=False).head(8).rename_axis(columna))  # Muestra las ocho categorías más frecuentes para limitar salidas.

    def run_analysis(self):  # Conserva el punto de entrada automatizado de la clase reutilizada.
        self._diagnose_missing_and_duplicates()  # Ejecuta una vez el diagnóstico inicial de calidad.
        self._summary_statistics()  # Ejecuta una vez el resumen y las distribuciones.
        self._detect_outliers()  # Ejecuta una vez la alerta de extremos IQR.
        self._correlation_analysis()  # Ejecuta una vez el análisis multivariado numérico.
        self._classification_analysis()  # Ejecuta el balance de clases y las relaciones categóricas.

eda = AutoEDA(df, target_col=TARGET)  # Instancia la herramienta reutilizada sobre todos los registros originales.
eda.run_analysis()  # Produce el diagnóstico completo sin imputar ni limpiar automáticamente.
```

**Cómo comprender una clase:** `class AutoEDA` reúne estado y operaciones relacionadas. `__init__` recibe la tabla y registra sus columnas. `self` identifica la instancia; los métodos trabajan con sus atributos.

| Método | Qué hace | Por qué / para qué |
|---|---|---|
| `__init__` | Guarda tabla, target y listas de columnas. | Reutilizar la herramienta sin repetir selección de tipos. |
| `_diagnose_missing_and_duplicates` | Construye una tabla con tipo, cardinalidad y nulos; cuenta filas duplicadas completas. | Detectar ausencia de datos, categorías y problemas de calidad. |
| `_summary_statistics` | Calcula estadística descriptiva e histogramas numéricos. | Reconocer escalas, concentración, asimetría y dispersión. |
| `_detect_outliers` | Calcula Q1, Q3, IQR y conteos fuera de las cercas de Tukey. | Señalar valores extremos que necesitan interpretación, no borrarlos. |
| `_correlation_analysis` | Calcula y dibuja Pearson entre números. | Explorar asociaciones lineales entre predictores. |
| `_classification_analysis` | Cuenta clases y cruza categorías con ingreso. | Comprender desbalance y asociaciones categóricas. |
| `run_analysis` | Ejecuta los diagnósticos una vez y en orden. | Automatizar el flujo sin repetir bloques. |

**Regla IQR:** un dato se señala cuando queda por debajo de `Q1 - 1.5 × IQR` o por encima de `Q3 + 1.5 × IQR`. Si Q1 y Q3 coinciden, el IQR es cero: valores distintos de esa concentración pueden quedar señalados sin ser necesariamente errores.

**Cómo leer el análisis categórico:** `Casos` indica cuántos registros sustentan una tasa; `Proporción >50K` es la fracción positiva dentro de esa categoría. Se muestran las ocho categorías más frecuentes de cada atributo para limitar la salida. Los nulos se diagnostican en calidad; `crosstab` no los incluye como una categoría adicional.

**Resultados relevantes:** se encuentran 52 duplicados completos; faltantes en `workclass`, `occupation` y `native-country`; y una clase positiva cercana al 23.9 %. Se producen dos figuras de EDA: histogramas y correlación.

**Qué no concluir:** Pearson no mide directamente el target categórico ni causalidad. Una categoría con alta tasa positiva tampoco causa necesariamente el ingreso.

### Celda 9 — Convertir el EDA en decisiones justificadas

**Tipo:** código. **Depende de:** la instancia `eda` de la celda 8.

```python
columnas_con_nulos = eda.calidad.index[eda.calidad["Nulos"] > 0].tolist()  # Recupera las variables que requieren imputación dentro de CV.
proporcion_positiva = eda.balance.loc[">50K", "Proporción"]  # Obtiene la prevalencia de la clase de interés.
mayor_alerta_iqr = eda.outliers["Extremos IQR"].idxmax()  # Identifica dónde la regla IQR genera más alertas.
display(Markdown(f"**Interpretación del EDA:** la clase `>50K` representa {proporcion_positiva:.1%} de los registros; predecir siempre `<=50K` daría accuracy {1 - proporcion_positiva:.1%}, pero recall positivo cero. Los faltantes aparecen en {columnas_con_nulos}, por lo que se usará imputación por tipo de variable. La mayor alerta IQR corresponde a `{mayor_alerta_iqr}`; los extremos se conservan porque pueden ser valores censales legítimos, no errores. Pearson solo resume relaciones lineales entre números; las tasas por categoría son asociaciones descriptivas, no efectos causales."))  # Explica los hallazgos con cifras calculadas en esta ejecución.
```

**Cómo funciona:** recupera tablas que ya calculó `AutoEDA`, sin volver a ejecutar estadísticas ni gráficas. `Markdown` muestra una explicación con cifras actuales.

**Por qué:** producir tablas no basta; hay que explicar sus consecuencias. Esta celda conecta faltantes con imputación y desbalance con una métrica que incluya recall.

**Resultado esperado:** la clase positiva representa 23.9 % del dataset original. Predecir siempre `<=50K` daría aproximadamente 76.1 % de accuracy y **recall positivo cero**.

**Aprendizaje central:** una accuracy aparentemente alta puede ocultar que el modelo no detecta ningún caso de interés. La mayor alerta IQR corresponde a `hours-per-week`; los extremos se conservan porque una alerta no prueba un error censal.

### Celda 10 — Plan de atributos, partición y pipeline

**Tipo:** texto.

**Qué explicar:** los atributos se seleccionan con criterios explícitos; se deduplican ejemplos, se limita la muestra y se reserva test antes de buscar configuraciones.

| Atributo excluido | Motivo |
|---|---|
| `fnlwgt` | Es un peso censal. Este experimento no realiza estimaciones poblacionales ponderadas. |
| `education` | Repite el nivel educativo ya representado mediante `education-num`. |
| `race`, `sex` | Se excluyen como predictores por la limitación ética del ejercicio. |

**Por qué no escalar:** Random Forest divide mediante umbrales de cada predictor; no necesita igualar sus escalas como métodos basados en distancias.

**Para qué deduplicar:** evitar que copias idénticas de un vector de predictores **con la misma etiqueta** aparezcan en distintas particiones. Perfiles iguales con etiquetas diferentes no se fusionan: no son duplicados de un ejemplo completo.

**Limitación:** eliminar repeticiones cambia frecuencias y proporciones. El modelo evalúa una muestra de perfiles deduplicados, no la población censal completa. Excluir atributos protegidos no elimina sus proxies.

### Celda 11 — Definir train/test y preprocesamiento

**Tipo:** código. **Depende de:** `df`, `TARGET`, `SEMILLA` y `MAX_FILAS`.

```python
columnas_excluidas = ["fnlwgt", "education", "race", "sex"]  # Excluye peso censal, educación redundante y atributos protegidos de Adult versión 2.
datos_modelado = df.drop(columns=columnas_excluidas)  # Conserva únicamente predictores justificados y el target.
duplicados_modelado = int(datos_modelado.duplicated().sum())  # Cuenta perfiles y etiquetas idénticos después de seleccionar atributos.
datos_modelado = datos_modelado.drop_duplicates().copy()  # Evita que copias exactas de un ejemplo aparezcan en distintas particiones.
if len(datos_modelado) > MAX_FILAS:  # Reduce el modelado solo cuando el dataset supera el presupuesto.
    datos_modelado, _ = train_test_split(datos_modelado, train_size=MAX_FILAS, stratify=datos_modelado[TARGET], random_state=SEMILLA)  # Obtiene una muestra reproducible conservando las clases.
X = datos_modelado.drop(columns=TARGET)  # Separa las características disponibles para predecir.
y = datos_modelado[TARGET].map({"<=50K": 0, ">50K": 1}).astype("int8")  # Codifica explícitamente la clase positiva como 1.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEMILLA)  # Reserva test antes de buscar hiperparámetros.
columnas_numericas = X_train.select_dtypes(include=np.number).columns.tolist()  # Detecta las variables numéricas sin estimar estadísticas de test.
columnas_categoricas = X_train.select_dtypes(exclude=np.number).columns.tolist()  # Detecta las categorías que transformará el pipeline.
pipeline_categorico = Pipeline([("imputador", SimpleImputer(strategy="most_frequent")), ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])  # Imputa y codifica categorías dentro de cada fold.
preprocesador = ColumnTransformer([("numericas", SimpleImputer(strategy="median"), columnas_numericas), ("categoricas", pipeline_categorico, columnas_categoricas)])  # Aplica cada transformación al tipo correcto sin un pipeline numérico de un solo paso.
print(f"Perfiles repetidos excluidos={duplicados_modelado:,} | Muestra={len(X):,} | train={len(X_train):,} | test={len(X_test):,}")  # Documenta el efecto de deduplicación y muestreo.
print(f"Proporción >50K: train={y_train.mean():.3f}, test={y_test.mean():.3f}")  # Comprueba el balance de la partición estratificada.
```

**Cómo funciona la preparación:**

1. Excluye cuatro atributos y cuenta duplicados después de esa selección.
2. Conserva una copia por combinación de predictores y target.
3. Si el dataset excede 8 000 filas, obtiene una muestra estratificada.
4. Separa `X` —predictores— de `y` —clase numérica—.
5. Reserva 25 % como test con otra partición estratificada.
6. Define transformaciones a partir de tipos, sin ajustarlas todavía.

**Qué significa estratificar:** `stratify` intenta mantener la proporción de clases en los conjuntos resultantes. No balancea artificialmente a 50/50 ni genera ejemplos.

**Qué aprende cada transformación cuando se ajuste:**

- `SimpleImputer(strategy="median")`: la mediana de cada columna numérica.
- `SimpleImputer(strategy="most_frequent")`: la categoría más frecuente de cada columna categórica.
- `OneHotEncoder`: las categorías observadas y su orden de indicadores.
- `ColumnTransformer`: aplica cada rama al conjunto correcto de atributos.

Una categoría como `workclass="Private"` pasa a indicadores numéricos. `handle_unknown="ignore"` permite procesar categorías no vistas; sus indicadores de esa variable quedan en cero, pero eso no garantiza una predicción confiable.

**Resultado esperado:** se excluyen **8 460 repeticiones**, se utilizan **8 000 filas**, y la partición deja **6 000 train / 2 000 test**. Hay cinco predictores numéricos y cinco categóricos. La proporción positiva de ambas particiones se imprime como aproximadamente `0.256`.

**Punto clave contra fuga:** aquí se **define** `preprocesador`; todavía no se ejecuta `fit`. Sus estadísticas se aprenderán dentro de los folds del pipeline.

### Celda 12 — Elegir modelo, métrica y espacio de búsqueda

**Tipo:** texto.

**Qué explicar:** se utiliza solo `RandomForestClassifier`, siguiendo la práctica bayesiana de la Unidad 2.

**Por qué este modelo:** permite relaciones no lineales, trabaja con los indicadores preparados y ofrece importancia interna y compatibilidad con TreeSHAP.

**Métrica de selección:**

`score = (accuracy + recall de la clase >50K) / 2`

Accuracy mide los aciertos totales; recall positivo mide qué proporción de ingresos altos se detecta. El promedio les da igual peso. No es F1 ni balanced accuracy y no incluye directamente precision.

**Para qué comparar búsquedas:** no se comparan tres algoritmos predictivos; se comparan tres maneras de encontrar configuraciones del **mismo** algoritmo.

### Celda 13 — Ejecutar las tres búsquedas

**Tipo:** código. **Depende de:** los conjuntos de la celda 11 y el preprocesador aún no ajustado.

```python
def accuracy_recall(y_real, y_predicho):  # Define el criterio común de selección de hiperparámetros.
    exactitud = accuracy_score(y_real, y_predicho)  # Calcula la proporción global de aciertos.
    sensibilidad = recall_score(y_real, y_predicho, pos_label=1, zero_division=0)  # Calcula la fracción de positivos detectados.
    return (exactitud + sensibilidad) / 2  # Da el mismo peso a ambas métricas.

scoring_combinado = make_scorer(accuracy_recall)  # Convierte la función en un scorer que utiliza las predicciones de cada fold.
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMILLA)  # Usa exactamente las mismas particiones estratificadas en los tres métodos.
clasificador = RandomForestClassifier(random_state=SEMILLA, n_jobs=1)  # Fija semilla y evita paralelismo dentro de cada ajuste.
pipeline_modelo = Pipeline([("preprocesador", preprocesador), ("modelo", clasificador)])  # Incluye el aprendizaje del preprocesamiento dentro de CV.
espacio_discreto = {"modelo__n_estimators": [80, 140], "modelo__max_depth": [12, 20], "modelo__min_samples_leaf": [1, 3], "modelo__class_weight": [None, "balanced_subsample"]}  # Define 16 configuraciones del mismo algoritmo.
espacio_bayesiano = {nombre: Categorical(valores) for nombre, valores in espacio_discreto.items()}  # Reutiliza las mismas opciones sin duplicar la definición del espacio.
opciones_cv = dict(estimator=pipeline_modelo, scoring=scoring_combinado, cv=cv, refit=False, n_jobs=2, error_score="raise", return_train_score=False)  # Comparte condiciones y evita ajustes finales redundantes.
buscadores = [("Grid Search", GridSearchCV(param_grid=espacio_discreto, **opciones_cv)), ("Randomized Search", RandomizedSearchCV(param_distributions=espacio_discreto, n_iter=N_ITER, random_state=SEMILLA, **opciones_cv)), ("Optimización bayesiana", BayesSearchCV(search_spaces=espacio_bayesiano, n_iter=N_ITER, random_state=SEMILLA, optimizer_kwargs={"n_initial_points": 4}, **opciones_cv))]  # Configura exhaustividad, muestreo y exploración guiada con la misma API.
resultados_busqueda = []  # Inicializa los registros medidos de cada método.
for nombre, busqueda in buscadores:  # Ejecuta secuencialmente los tres métodos para evitar competencia entre búsquedas.
    inicio = time.perf_counter()  # Registra el comienzo del ajuste completo.
    busqueda.fit(X_train, y_train)  # Evalúa configuraciones usando solo los datos de entrenamiento.
    duracion = time.perf_counter() - inicio  # Mide tiempo de pared incluyendo el costo del buscador.
    indice = busqueda.best_index_  # Localiza la configuración con mayor score combinado medio.
    configuraciones_unicas = len({tuple(sorted(parametros.items())) for parametros in busqueda.cv_results_["params"]})  # Cuenta cobertura real incluso si Bayes repite propuestas.
    resultados_busqueda.append({"Método": nombre, "Score combinado CV": busqueda.best_score_, "Desviación entre folds": busqueda.cv_results_["std_test_score"][indice], "Tiempo (s)": duracion, "Evaluaciones": len(busqueda.cv_results_["params"]), "Configuraciones únicas": configuraciones_unicas, "Mejores hiperparámetros": busqueda.best_params_})  # Guarda calidad, variabilidad, costo y configuración.
    print(f"{nombre}: score CV={busqueda.best_score_:.4f}, tiempo={duracion:.1f} s, configuraciones únicas={configuraciones_unicas}")  # Informa resultados reales de cada método.
```

**Cómo interpretar la función de score:** `make_scorer` adapta `accuracy_recall` a la interfaz de los buscadores. No se entrenan modelos dentro de esa función: recibe etiquetas reales y predichas de la validación.

**Cómo funciona la CV de tres folds:** train se divide en tres partes; para cada configuración, el pipeline aprende en dos partes y se evalúa en la tercera. Se repite cambiando la parte reservada. Las medianas, modas, categorías y árboles se aprenden nuevamente dentro de cada ajuste, sin utilizar la parte validada.

| Hiperparámetro | Opciones | Qué controla |
|---|---|---|
| `n_estimators` | 80, 140 | Cantidad de árboles: estabilidad y costo. |
| `max_depth` | 12, 20 | Profundidad máxima: complejidad. |
| `min_samples_leaf` | 1, 3 | Mínimo de ejemplos por hoja: regularización. |
| `class_weight` | `None`, `"balanced_subsample"` | Ponderación de clases, calculada en la muestra de cada árbol cuando se activa. |

El prefijo `modelo__` indica al pipeline que el hiperparámetro pertenece al paso llamado `modelo`. Hay `2 × 2 × 2 × 2 = 16` combinaciones.

| Búsqueda | Cómo explora | Costo de CV en este experimento |
|---|---|---|
| Grid Search | Revisa las 16 combinaciones del espacio. | 16 × 3 = 48 ajustes. |
| Randomized Search | Selecciona 12 combinaciones al azar, sin reemplazo porque todas las opciones son listas finitas. | 12 × 3 = 36 ajustes. |
| BayesSearchCV | Realiza 12 propuestas: cuatro iniciales y las posteriores guiadas por un modelo sustituto. | 12 × 3 = 36 ajustes; puede repetir configuraciones. |

**Por qué `refit=False`:** cada búsqueda solo evalúa CV; no entrena además su propio modelo definitivo. El modelo final se entrenará una vez después de elegir, como pide la consigna.

**Por qué `error_score="raise"`:** si un ajuste falla, el problema se muestra y detiene la búsqueda. No se publica una comparación que oculte configuraciones fallidas.

**Para qué sirve el bucle:** utiliza el mismo procedimiento de ajuste, medición y almacenamiento para los tres métodos, evitando tres bloques redundantes.

**Resultados guardados:**

| Método | Mejor score CV | Desviación entre folds | Tiempo aproximado | Evaluaciones | Configuraciones únicas |
|---|---:|---:|---:|---:|---:|
| Grid Search | 0.8268 | 0.0042 | 11.8 s | 16 | 16 |
| Randomized Search | 0.8268 | 0.0042 | 8.5 s | 12 | 12 |
| Optimización bayesiana | 0.8268 | 0.0042 | 15.9 s | 12 | 10 |

**Cómo interpretar los avisos bayesianos:** el espacio es pequeño y el optimizador puede proponer puntos ya explorados. Los avisos no son errores de ajuste; la cobertura única permite ver que 12 propuestas no equivalen siempre a 12 configuraciones diferentes.

### Celda 14 — Definir la regla de elección

**Tipo:** texto.

**Qué explicar:** el mayor score medio de CV gana; un empate exacto se resuelve por menor tiempo.

**Por qué:** el criterio debe definirse antes de observar test, no adaptarse a sus resultados.

**Para qué:** seleccionar una configuración por una mezcla explícita de aciertos y detección positiva. La desviación entre folds describe variación; no demuestra significancia estadística.

### Celda 15 — Comparar y seleccionar

**Tipo:** código. **Depende de:** `resultados_busqueda` de la celda 13.

```python
tabla_comparacion = pd.DataFrame(resultados_busqueda).sort_values(["Score combinado CV", "Tiempo (s)"], ascending=[False, True]).reset_index(drop=True)  # Ordena por calidad y usa costo solo para desempatar.
display(tabla_comparacion)  # Presenta score, dispersión, tiempo, cobertura e hiperparámetros.
nombre_ganador = tabla_comparacion.loc[0, "Método"]  # Identifica el método elegido exclusivamente por CV.
mejores_parametros = tabla_comparacion.loc[0, "Mejores hiperparámetros"]  # Recupera la configuración que se entrenará de nuevo.
ganador = tabla_comparacion.iloc[0]  # Recupera las métricas medidas del método seleccionado.
mas_rapido = tabla_comparacion.loc[tabla_comparacion["Tiempo (s)"].idxmin()]  # Identifica el menor tiempo independientemente del score.
display(Markdown(f"**Resultado y justificación:** se elige **{nombre_ganador}**, con promedio accuracy-recall CV **{ganador['Score combinado CV']:.4f}** y desviación entre folds {ganador['Desviación entre folds']:.4f}; requiere {ganador['Tiempo (s)']:.1f} s. El más rápido es **{mas_rapido['Método']}**, con {mas_rapido['Tiempo (s)']:.1f} s y score {mas_rapido['Score combinado CV']:.4f}. La tabla permite verificar si ahorrar tiempo conserva el poder predictivo observado. Se prioriza el promedio de aciertos y detección positiva, no la velocidad por sí sola. Los empates de score se resuelven por tiempo y no indican mayor poder predictivo."))  # Explica el resultado sin fijar un ganador histórico.
```

**Cómo funciona:** transforma los registros en una tabla, ordena por score descendente y tiempo ascendente, y recupera los parámetros de la primera fila. No entrena ni consulta test.

**Resultado esperado:** los métodos empatan en el mejor score observado. Se elige **Randomized Search** porque fue el más rápido en esa ejecución.

**Configuración ganadora:**

| Hiperparámetro | Valor |
|---|---|
| `n_estimators` | 140 |
| `max_depth` | 12 |
| `min_samples_leaf` | 3 |
| `class_weight` | `"balanced_subsample"` |

**Qué no afirmar:** Randomized Search no mostró un score superior a los otros métodos; fue más rápido con score empatado. En otra ejecución puede cambiar el desempate porque cambian los tiempos. Tampoco se debe inferir que todos los métodos eligieron exactamente el mismo diccionario solo porque sus scores empatan.

**Objetos preparados:** `nombre_ganador` y `mejores_parametros`.

### Celda 16 — Diseñar la evaluación final

**Tipo:** texto.

**Qué explicar:** se crea un modelo nuevo con los parámetros seleccionados, se ajusta en train completo y se evalúa en test.

**Por qué:** CV sirve para seleccionar; test sirve para estimar generalización después de la selección.

**Para qué:** obtener accuracy, recall, reporte por clase, matriz de confusión y ROC-AUC sin volver a optimizar con test. Usar test posteriormente para diagnósticos XAI no autoriza a modificar el modelo según esas explicaciones.

### Celda 17 — Entrenar y evaluar el modelo final

**Tipo:** código. **Depende de:** la configuración ganadora, el pipeline base y los conjuntos train/test.

```python
modelo_final = clone(pipeline_modelo).set_params(**mejores_parametros)  # Crea una instancia no ajustada con la configuración ganadora.
modelo_final.fit(X_train, y_train)  # Entrena un nuevo pipeline con todo el conjunto de entrenamiento.
indice_positivo = list(modelo_final.classes_).index(1)  # Identifica explícitamente la columna de probabilidad correspondiente a >50K.
y_prob = modelo_final.predict_proba(X_test)[:, indice_positivo]  # Obtiene probabilidades una sola vez para clase, ROC y AUC.
y_pred = (y_prob > 0.5).astype("int8")  # Reproduce la decisión binaria de Random Forest, incluyendo empates a favor de clase 0.
accuracy_test = accuracy_score(y_test, y_pred)  # Calcula aciertos globales en datos no usados para ajustar.
recall_test = recall_score(y_test, y_pred, pos_label=1, zero_division=0)  # Calcula detección de ingresos altos en test.
auc_test = roc_auc_score(y_test, y_prob)  # Calcula una sola vez el área ROC con probabilidades positivas.
print(f"Accuracy={accuracy_test:.4f} | Recall >50K={recall_test:.4f} | Score combinado={(accuracy_test + recall_test) / 2:.4f} | ROC-AUC={auc_test:.4f}")  # Reporta las métricas finales, distintas de CV.
print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["<=50K", ">50K"], zero_division=0))  # Incluye precision, recall, F1 y soporte por clase.
matriz = confusion_matrix(y_test, y_pred, labels=[0, 1])  # Calcula los cuatro tipos de resultado en un orden conocido.
tn, fp, fn, tp = matriz.ravel()  # Extrae verdaderos/falsos negativos y positivos para interpretación.
fig, ejes = plt.subplots(1, 2, figsize=(12, 5))  # Agrupa matriz y curva ROC sin figuras redundantes.
ConfusionMatrixDisplay(matriz, display_labels=["<=50K", ">50K"]).plot(ax=ejes[0], cmap="Blues", values_format="d", colorbar=False)  # Visualiza conteos de errores y aciertos.
RocCurveDisplay.from_predictions(y_test, y_prob, pos_label=1, ax=ejes[1], name="Random Forest")  # Traza ROC usando probabilidades ya calculadas.
ejes[1].plot([0, 1], [0, 1], "--", color="gray", label="Azar")  # Añade la referencia de un clasificador sin discriminación.
ejes[0].set_title("Matriz de confusión — test")  # Identifica la partición evaluada.
ejes[1].set_title(f"ROC — test (AUC={auc_test:.3f})")  # Vincula la curva con el AUC calculado.
ejes[1].legend(loc="lower right")  # Muestra la identificación del modelo y la diagonal.
plt.tight_layout()  # Ajusta ambas figuras a la misma fila.
plt.show()  # Presenta los resultados de la evaluación.
display(Markdown(f"**Interpretación:** TN={tn}, FP={fp}, FN={fn} y TP={tp}. Se omiten {fn} personas de ingreso alto y se clasifican {fp} personas de ingreso bajo como ingreso alto. El recall {recall_test:.1%} equivale a TP/(TP+FN); accuracy {accuracy_test:.1%} equivale a (TP+TN)/N. La tasa de falsos positivos es {fp / (fp + tn):.1%}. AUC={auc_test:.4f} resume la discriminación sobre distintos umbrales; no significa que ese porcentaje de casos quede bien clasificado. La ponderación de clases puede favorecer recall a costa de precision: se contrasta con el reporte por clase."))  # Explica cifras reales, fórmulas y el costo de los errores.
```

**Cómo funciona el entrenamiento:** `clone` crea un pipeline sin estado aprendido; `set_params` incorpora la configuración ganadora; `fit` aprende transformaciones y árboles con las 6 000 filas de train.

**Cómo funciona la predicción:** se calcula una sola vez `predict_proba(X_test)` y se utiliza su columna de clase positiva. `y_prob > 0.5` produce la clase 1; un empate queda en clase 0, reproduciendo la decisión del Random Forest binario utilizado.

**Qué mide cada resultado:**

| Medida | Interpretación |
|---|---|
| Accuracy | `(TP + TN) / N`: proporción total de aciertos. |
| Recall positivo | `TP / (TP + FN)`: proporción de ingresos altos detectados. |
| Precision positiva | `TP / (TP + FP)`: proporción de predicciones positivas que son correctas. |
| F1 | Combina precision y recall; no es la métrica usada para elegir hiperparámetros. |
| ROC-AUC | Capacidad de ordenar positivos por encima de negativos mediante sus scores; no es el porcentaje de clases acertadas. |

**Resultados guardados:** accuracy **0.8045**, recall **0.8376**, score combinado test **0.8210** y ROC-AUC **0.9044**.

| Clase real / predicha | `<=50K` | `>50K` |
|---|---:|---:|
| `<=50K` | TN = 1181 | FP = 308 |
| `>50K` | FN = 83 | TP = 428 |

**Cómo interpretarlos:** se detectan 428 de los 511 positivos y se omiten 83. Se clasifican como positivos 308 casos realmente negativos. La precision positiva es aproximadamente 0.58: mejorar recall no implica eliminar falsos positivos.

**Referencia correcta para el baseline:** en este test hay 1489 negativos de 2000 registros; predecir siempre negativo daría accuracy **0.7445** y recall cero. El baseline aproximado de 76.1 % explicado en el EDA pertenece al dataset original, cuya distribución cambia después de deduplicar y muestrear.

**Cómo leer ROC:** cada punto corresponde a un umbral diferente. El eje horizontal mide falsos positivos y el vertical verdaderos positivos. La diagonal representa ausencia de discriminación; la decisión publicada utiliza 0.5, no un umbral elegido a partir de test.

### Celda 18 — Entender Feature Importance

**Tipo:** texto.

**Qué explicar:** la importancia interna resume la reducción de impureza que producen los cortes de los árboles.

**Por qué:** es una primera interpretación global disponible en el modelo entrenado, sin nuevos ajustes.

**Para qué:** identificar indicadores utilizados por el bosque y contrastarlos con medidas externas. Su sesgo por oportunidades de corte y la correlación entre variables impiden interpretarla como causalidad.

### Celda 19 — Mostrar importancia interna

**Tipo:** código. **Depende de:** `modelo_final` de la celda 17.

```python
modelo_rf = modelo_final.named_steps["modelo"]  # Recupera el Random Forest ya entrenado sin reajustarlo.
preprocesador_final = modelo_final.named_steps["preprocesador"]  # Recupera las transformaciones aprendidas exclusivamente con train.
nombres_features = preprocesador_final.get_feature_names_out()  # Obtiene el orden real de columnas numéricas y one-hot.
importancia_nativa = pd.Series(modelo_rf.feature_importances_, index=nombres_features, name="Importancia nativa").sort_values(ascending=False)  # Alinea cada importancia interna con su variable transformada.
display(importancia_nativa.head(15).to_frame())  # Muestra los quince indicadores con mayor reducción de impureza.
importancia_nativa.head(15).sort_values().plot.barh(figsize=(9, 5), color="steelblue")  # Visualiza el ranking superior con el mayor valor arriba.
plt.xlabel("Disminución normalizada de impureza")  # Especifica la escala de la medida.
plt.title("Feature Importance — variables transformadas")  # Aclara la unidad de comparación.
plt.tight_layout()  # Ajusta las etiquetas de los indicadores.
plt.show()  # Presenta la importancia global interna.
```

**Cómo funciona:** recupera los pasos ya ajustados del pipeline. `get_feature_names_out()` proporciona el orden real de los indicadores; la serie vincula ese orden con `feature_importances_`.

**Resultado esperado:** tabla y gráfico de los quince indicadores principales. En la ejecución guardada destacan `marital-status_Married-civ-spouse`, `relationship_Husband` y `education-num`.

**Por qué aparecen nombres largos:** `numericas__` y `categoricas__` identifican la rama del preprocesador; una categoría como `Husband` se representa mediante su propio indicador.

**Qué no comparar directamente:** este ranking trabaja con variables transformadas. La siguiente técnica trabaja con atributos originales.

### Celda 20 — Entender Permutation Importance

**Tipo:** texto.

**Qué explicar:** alterar una variable permite observar cuánto depende de ella el desempeño del modelo ya entrenado.

**Por qué:** la reducción de impureza describe construcción de árboles; la permutación mide pérdida de desempeño en datos reservados.

**Para qué:** contrastar una medida interna con una medida externa utilizando el mismo score combinado de selección.

### Celda 21 — Permutar predictores y medir la pérdida

**Tipo:** código. **Depende de:** `modelo_final`, `X_test`, `y_test` y `scoring_combinado`.

```python
resultado_permutacion = permutation_importance(modelo_final, X_test, y_test, scoring=scoring_combinado, n_repeats=3, random_state=SEMILLA, n_jobs=2)  # Evalúa la pérdida del criterio de selección al alterar predictores originales.
importancia_permutacion = pd.DataFrame({"Caída media": resultado_permutacion.importances_mean, "Desviación": resultado_permutacion.importances_std}, index=X_test.columns).sort_values("Caída media", ascending=False)  # Resume sensibilidad y variabilidad por atributo.
display(importancia_permutacion)  # Presenta todas las variables originales, incluidas caídas pequeñas o negativas.
importancia_permutacion.sort_values("Caída media").plot.barh(y="Caída media", xerr="Desviación", figsize=(9, 5), color="darkcyan", legend=False)  # Grafica medias y dispersión sin volver a calcular permutaciones.
plt.xlabel("Caída del promedio accuracy-recall")  # Identifica la diferencia de score, no una proporción normalizada.
plt.title("Permutation Importance — variables originales")  # Distingue este ranking del desglosado one-hot.
plt.tight_layout()  # Ajusta nombres y barras de error.
plt.show()  # Presenta el diagnóstico global externo.
```

**Cómo funciona:** se modifica el orden de los valores de una columna de test, se vuelve a puntuar y se compara con el desempeño sin alterarla. Se hacen tres repeticiones por atributo.

**Por qué se utiliza el pipeline completo:** la columna categórica se permuta antes de codificarla; así se mide el atributo original, no cada indicador one-hot por separado.

**Resultado esperado:** una tabla con caída media y desviación, junto con su gráfico. En la ejecución guardada, los atributos principales son `marital-status`, `relationship` y `capital-gain`.

**Cómo leer las barras:** una caída grande implica mayor dependencia predictiva en esta medición. Una caída negativa indica que al permutar mejoró ligeramente el score; no demuestra que ese atributo sea perjudicial en toda población. La desviación corresponde a las repeticiones, no a un intervalo de confianza.

**Limitación:** variables correlacionadas pueden compensarse. El resultado es diagnóstico; no se elimina una variable ni se repite la selección a partir de test.

### Celda 22 — Preparar una explicación SHAP global

**Tipo:** texto.

**Qué explicar:** SHAP reparte entre los indicadores la diferencia entre una salida del modelo y su valor base.

**Por qué:** la importancia absoluta ordena variables, pero no siempre muestra si sus valores aumentan o reducen una predicción.

**Para qué:** utilizar `bar` para magnitud global y `beeswarm` para dirección y dispersión. Se explican 200 casos para limitar costo, no porque el modelo se haya entrenado con solo 200.

### Celda 23 — Calcular SHAP global y contrastar técnicas

**Tipo:** código. **Depende de:** los objetos ajustados y nombres de la celda 19, test y la importancia por permutación.

```python
X_shap_original = X_test.sample(n=min(200, len(X_test)), random_state=SEMILLA)  # Selecciona únicamente los casos que se explicarán.
X_shap = pd.DataFrame(preprocesador_final.transform(X_shap_original), columns=nombres_features, index=X_shap_original.index)  # Transforma la muestra una sola vez con el preprocesador final.
explainer = shap.TreeExplainer(modelo_rf, feature_perturbation="tree_path_dependent")  # Usa la estructura y los conteos del bosque como referencia explicativa.
explicacion_positiva = explainer(X_shap)[:, :, indice_positivo]  # Calcula y extrae contribuciones de probabilidad para la clase >50K.
shap.plots.bar(explicacion_positiva, max_display=15)  # Resume importancia global como magnitud absoluta media por indicador.
shap.plots.beeswarm(explicacion_positiva, max_display=15)  # Muestra dirección, dispersión y valores de las variables en los mismos casos.
importancia_shap = pd.Series(np.abs(explicacion_positiva.values).mean(axis=0), index=nombres_features).sort_values(ascending=False)  # Calcula el ranking numérico a partir de explicaciones ya obtenidas.
display(Markdown(f"**Contraste de resultados:** los tres indicadores principales por impureza son {', '.join(importancia_nativa.head(3).index)}; por media absoluta SHAP, {', '.join(importancia_shap.head(3).index)}. Los atributos originales más sensibles a permutación son {', '.join(importancia_permutacion.head(3).index)}. Coincidencias respaldan una dependencia descriptiva del modelo; diferencias pueden deberse a correlación, al desglose one-hot y a que impureza, caída de score y contribuciones a probabilidad miden cosas distintas. En beeswarm el signo, no el ranking absoluto, permite distinguir qué valores elevan o reducen la salida."))  # Interpreta los rankings calculados sin afirmar coincidencias prefijadas.
```

**Cómo funciona:**

1. Toma una muestra reproducible de test **antes** de transformarla.
2. Utiliza el preprocesador final ya ajustado.
3. `TreeExplainer` utiliza los árboles y sus conteos como referencia mediante `tree_path_dependent`.
4. Selecciona en el tensor SHAP la salida correspondiente a la clase positiva.
5. Obtiene dos gráficas y redacta el contraste con las técnicas anteriores.

**Por qué existe un eje de clases:** el Random Forest devuelve probabilidades de ambas clases. La explicación tiene ejes de casos, indicadores y clases; extraer `indice_positivo` evita interpretar por error la salida negativa.

**Cómo leer `bar`:** utiliza la media del valor SHAP absoluto; un indicador puede ser globalmente importante aunque sus aportes cambien de signo entre personas.

**Cómo leer `beeswarm`:** cada punto representa un caso. A la derecha el aporte aumenta la salida respecto de la base; a la izquierda la reduce. El color representa el valor del indicador: rojo alto, azul bajo.

**Resultados guardados:** impureza y SHAP absoluto destacan los mismos tres indicadores principales: estado civil casado, rol familiar Husband y nivel educativo. Permutación destaca estado civil, relación familiar y ganancia de capital.

| Técnica | Unidad de análisis | Qué mide |
|---|---|---|
| Feature Importance | Indicadores transformados | Reducción normalizada de impureza. |
| Permutation Importance | Atributos originales | Caída del score combinado. |
| SHAP global | Indicadores transformados | Magnitud media de contribución a la probabilidad positiva. |

**Qué no hacer:** comparar numéricamente sus valores como si estuvieran en la misma escala. Las coincidencias apoyan una dependencia descriptiva; no prueban efectos causales.

### Celda 24 — Diseñar un caso nuevo

**Tipo:** texto.

**Qué explicar:** el caso se escribe manualmente como un perfil hipotético con todos los predictores, fuera de train y test.

**Por qué:** copiar una fila de test demostraría inferencia sobre un caso reservado, pero no la construcción de una observación externa.

**Para qué:** enseñar cómo pasar de los valores originales a una probabilidad y a una explicación individual. No se inventa una etiqueta real ni se calcula accuracy para el perfil sintético.

### Celda 25 — Predecir y explicar el perfil externo

**Tipo:** código. **Depende de:** el preprocesador final, el bosque, los nombres transformados y el explicador SHAP.

```python
X_nuevo = pd.DataFrame([{"age": 37, "workclass": "Private", "education-num": 13, "marital-status": "Married-civ-spouse", "occupation": "Prof-specialty", "relationship": "Husband", "capital-gain": 1234, "capital-loss": 0, "hours-per-week": 41, "native-country": "United-States"}])[X_train.columns]  # Define un perfil nuevo completo y ordena sus atributos como en entrenamiento.
if df[X_train.columns].eq(X_nuevo.iloc[0]).all(axis=1).any():  # Detecta si el perfil hipotético coincide con algún vector del dataset completo.
    raise ValueError("El perfil propuesto ya existe en Adult; cambia sus atributos para demostrar una predicción nueva.")  # Evita presentar una observación conocida como nueva.
X_nuevo_transformado = pd.DataFrame(preprocesador_final.transform(X_nuevo), columns=nombres_features)  # Aplica una sola vez la imputación y codificación ya aprendidas.
probabilidad_nueva = float(modelo_rf.predict_proba(X_nuevo_transformado.to_numpy())[0, indice_positivo])  # Obtiene la probabilidad del pipeline a partir de su transformación ya calculada.
prediccion_nueva = int(probabilidad_nueva > 0.5)  # Reproduce la decisión binaria sin una segunda inferencia.
explicacion_local = explainer(X_nuevo_transformado)[:, :, indice_positivo][0]  # Obtiene la explicación SHAP positiva del único caso nuevo.
probabilidad_reconstruida = float(explicacion_local.base_values + explicacion_local.values.sum())  # Reconstruye la salida con el valor base y las contribuciones.
np.testing.assert_allclose(probabilidad_reconstruida, probabilidad_nueva, atol=1e-6, rtol=1e-6)  # Comprueba que SHAP explica numéricamente la probabilidad publicada.
display(X_nuevo.T.rename(columns={0: "Perfil hipotético"}))  # Presenta los valores originales de la observación nueva.
print(f"Predicción: {'>50K' if prediccion_nueva else '<=50K'} | P(>50K)={probabilidad_nueva:.4f} | Etiqueta real: no disponible")  # Informa la salida sin fingir validación de un caso sintético.
shap.plots.waterfall(explicacion_local, max_display=15)  # Visualiza cómo cada contribución lleva de la base a la salida final.
contribuciones_locales = pd.Series(explicacion_local.values, index=nombres_features)  # Alinea los SHAP locales con sus indicadores para explicarlos.
aumentan = contribuciones_locales[contribuciones_locales > 0].nlargest(3).round(4).to_dict()  # Identifica las mayores contribuciones que elevan P(>50K).
reducen = contribuciones_locales[contribuciones_locales < 0].nsmallest(3).round(4).to_dict()  # Identifica las mayores contribuciones que reducen P(>50K).
display(Markdown(f"**Interpretación local:** la base es {float(explicacion_local.base_values):.4f}; las contribuciones suman {explicacion_local.values.sum():+.4f}, dando P(>50K)={probabilidad_reconstruida:.4f}. Los mayores aportes positivos son `{aumentan}` y los negativos `{reducen}`, expresados en unidades de probabilidad. El modelo clasifica este perfil como **{'>50K' if prediccion_nueva else '<=50K'}** con umbral 0.5. Esto explica una decisión concreta y no permite medir su corrección: el perfil es hipotético y no tiene ingreso observado."))  # Explica signo, escala, identidad aditiva y limitación del caso nuevo.
```

**Cómo funciona la inferencia:**

1. Crea un DataFrame de una fila y reordena sus columnas como en train.
2. Comprueba que ese vector no exista en el dataset original.
3. Aplica una vez las transformaciones aprendidas.
4. Consulta el bosque sobre la matriz preparada y decide con umbral 0.5.
5. Calcula SHAP positivo del mismo caso.
6. Reconstruye la probabilidad y comprueba que coincide con la publicada.

**Por qué se consulta aquí el bosque y no se transforma dos veces mediante el pipeline:** ya se dispone de la matriz transformada por el paso final. La inferencia utiliza exactamente ese bosque y esa transformación; se reutiliza la matriz para SHAP.

**Identidad que se comprueba:**

`P(>50K) = valor base + suma de contribuciones SHAP`

En la ejecución guardada: **0.5000 + 0.3083 = 0.8083**, con cifras redondeadas. La comprobación utiliza valores completos y tolerancia numérica de `1e-6`.

**Resultado esperado:** clase `>50K`, probabilidad **0.8083** y etiqueta real **no disponible**. El waterfall muestra el recorrido desde la base hasta la salida.

**Interpretación de aportes locales:** los mayores positivos incluyen casado (+0.0770), nivel educativo (+0.0764) y Husband (+0.0575). Entre los negativos aparecen ganancia de capital (−0.0249), el indicador de ocupación Exec-managerial (−0.0101) y pérdida de capital (−0.0073).

**Detalle educativo:** un indicador one-hot con valor cero también puede tener contribución SHAP; la ausencia de esa categoría puede diferir de la referencia del explicador. El valor base no tiene por qué coincidir con la prevalencia positiva del dataset original.

**Qué demuestra la comprobación:** coherencia entre la explicación y la salida del modelo. No demuestra corrección del ingreso predicho, calibración de probabilidad ni causalidad.

### Celda 26 — Preparar la conclusión

**Tipo:** texto.

**Qué explicar:** la conclusión debe sintetizar selección, desempeño, explicabilidad y límites con cifras de esta ejecución.

**Por qué:** una conclusión copiada manualmente puede quedar desactualizada cuando cambien datos o configuraciones.

**Para qué:** distinguir resultados comprobados de afirmaciones que el experimento no permite, como generalización universal o ausencia de sesgo.

### Celda 27 — Redactar una conclusión calculada

**Tipo:** código. **Depende de:** la selección y las métricas finales ya calculadas.

```python
display(Markdown(f"**Conclusión de esta ejecución:** {nombre_ganador} selecciona `{mejores_parametros}` por CV. El modelo entrenado de nuevo alcanza accuracy **{accuracy_test:.4f}**, recall positivo **{recall_test:.4f}** y ROC-AUC **{auc_test:.4f}** en test. La comparación anterior detalla score, tiempo y cobertura de los tres métodos. Las tres técnicas XAI explican dependencia global desde perspectivas distintas y SHAP local reconstruye la probabilidad de un caso externo. Los resultados se limitan a la muestra de perfiles deduplicados, a un solo split y a datos de 1994; no prueban causalidad, equidad ni validez para decisiones reales."))  # Resume únicamente métricas y límites verificados en esta ejecución.
```

**Cómo funciona:** una cadena formateada incorpora objetos existentes; no vuelve a entrenar, evaluar ni calcular SHAP.

**Resultado esperado:** identifica Randomized Search, muestra sus parámetros y publica accuracy 0.8045, recall 0.8376 y ROC-AUC 0.9044.

**Para qué sirve:** asegurar que el resumen se mantenga vinculado con los cálculos, incluyendo límites del split único, la deduplicación y la antigüedad del censo.

### Celda 28 — Cierre y entregables

**Tipo:** texto.

**Qué escribir:** una lista de cumplimiento de dataset, EDA reutilizado, preparación, tres búsquedas, métrica combinada, entrenamiento nuevo, evaluación y XAI global/local.

**Por qué:** terminar el cálculo no equivale a completar toda la entrega académica.

**Para qué:** revisar que notebook y reporte expliquen tanto el código como sus resultados. La consigna exige un **reporte PDF**: el Word actualizado se debe exportar a PDF para esa entrega, después de completar los datos personales de portada.

## 5. Dependencias entre las celdas de código

Esta tabla ayuda a comprender un `NameError`: normalmente falta ejecutar una celda anterior que crea el objeto requerido.

| Celda | Objetos principales que recibe | Objetos principales que deja |
|---:|---|---|
| 3 | Entorno e internet | Dependencias instaladas. |
| 4 | Dependencias | Imports, `SEMILLA`, `MAX_FILAS`, `N_ITER`. |
| 6 | Imports y semilla disponible | `TARGET`, `datos_openml`, `df`. |
| 8 | `df`, `TARGET` | `AutoEDA`, `eda`, tablas diagnósticas. |
| 9 | `eda` | Interpretación del EDA. |
| 11 | `df`, constantes | `X_train`, `X_test`, `y_train`, `y_test`, `preprocesador`. |
| 13 | Particiones y preprocesador | Scorer, CV, pipeline, buscadores ajustados y registros de búsqueda. |
| 15 | Registros de búsqueda | Tabla, `nombre_ganador`, `mejores_parametros`. |
| 17 | Pipeline y parámetros ganadores | `modelo_final`, probabilidades, clases, matriz y métricas test. |
| 19 | Modelo final | `modelo_rf`, `preprocesador_final`, nombres e importancia interna. |
| 21 | Modelo y test | Importancia por permutación. |
| 23 | Bosque, preprocesador y test | `explainer`, muestra y explicación SHAP global. |
| 25 | Objetos finales y explicador | Perfil nuevo, probabilidad y explicación local. |
| 27 | Nombre, parámetros y métricas | Conclusión de la ejecución. |

## 6. Decisiones que evitan pasos innecesarios

- El dataset se descarga una sola vez; no se mezcla con un CSV de otra versión.
- La clase EDA se incluye directamente: no se escribe un módulo para importarlo inmediatamente.
- El EDA diagnostica, pero no realiza una segunda limpieza automática.
- No se agregan regresión, clustering ni redes neuronales.
- No se escala para un modelo que no necesita comparar distancias entre variables.
- Las búsquedas comparten un único espacio de opciones y un único bucle de ejecución.
- `refit=False` evita tres ajustes definitivos que después se descartarían.
- Se entrena una instancia final nueva, como exige la actividad.
- Las probabilidades de test se reutilizan para decisión binaria y evaluación.
- La muestra SHAP se selecciona antes de transformarla, evitando preparar todo test para explicar solo 200 casos.
- El caso nuevo se transforma una vez y su matriz se utiliza tanto en inferencia como en SHAP.
- Las interpretaciones y conclusiones reutilizan resultados existentes, no repiten cálculos.

Estas decisiones no eliminan comprobaciones de consistencia, como validar el target, detectar si el perfil nuevo ya existe o verificar la aditividad de SHAP.

## 7. Errores frecuentes y cómo entenderlos

| Situación | Qué revisar |
|---|---|
| `ModuleNotFoundError` | Ejecutar la celda 3 en el kernel correcto; reiniciar si el entorno lo solicita. |
| Error al descargar OpenML | Comprobar internet y disponibilidad del servicio. No sustituir silenciosamente el dataset por otro. |
| `NameError` | Revisar el mapa de dependencias y ejecutar las celdas anteriores en orden. |
| Error de etiquetas válidas | Revisar la descarga y el target; no eliminar clases inesperadas para forzar el resultado. |
| Categoría no vista | El codificador la admite, pero su predicción requiere cautela; no equivale a validar cualquier texto. |
| Avisos de propuestas bayesianas repetidas | Son compatibles con un espacio pequeño; consultar configuraciones únicas y no ocultar errores de ajuste. |
| Advertencia de barra de progreso | Es un aviso de interfaz, no una falla del modelo ni una razón para añadir paquetes al trabajo. |
| El perfil ya aparece en Adult | Modificar sus valores con coherencia y ejecutar nuevamente la celda 25. |
| No coincide base + SHAP con la probabilidad | Revisar clase seleccionada, orden de columnas, preprocesador y modelo; no quitar la comprobación. |
| Cambia el método ganador | Revisar si hubo empate y cambió el tiempo. La velocidad depende de CPU, carga y arranque de procesos. |
| Cambian métricas o nombres transformados | Revisar datos, versiones, semilla, atributos, muestreo y orden de ejecución; actualizar también el reporte. |

## 8. Autoevaluación antes de entregar

El aprendizaje queda completo cuando se puede responder, con los objetos y salidas del notebook:

1. Qué se predice, cuál es la clase positiva y de dónde vienen los 48 842 registros.
2. Por qué el EDA utiliza el dataset completo y las métricas finales solo 2 000 filas de test.
3. Por qué 52 duplicados completos no equivalen a 8 460 duplicados tras seleccionar atributos.
4. Cuándo se aprenden medianas, modas y categorías, y cómo el pipeline evita fuga entre folds.
5. En qué se diferencian algoritmo, hiperparámetro y método de búsqueda.
6. Cómo se calcula el promedio accuracy-recall y por qué no basta la accuracy.
7. Por qué un empate de score no demuestra superioridad predictiva y puede resolverse por tiempo.
8. Cómo se obtienen accuracy y recall a partir de TN, FP, FN y TP.
9. Por qué las tres técnicas globales miden cosas distintas y no prueban causalidad.
10. Cómo se reconstruye una probabilidad con SHAP y por qué un perfil hipotético no tiene accuracy medible.
11. Qué debe actualizarse en el reporte si se vuelve a ejecutar o modificar el notebook.

## 9. Fuentes y relación con las prácticas

La guía sigue el código real del notebook y las instrucciones locales. Las referencias externas explican los datos y las técnicas; **las métricas publicadas provienen de la ejecución del notebook**, no de esas fuentes.

- Dataset: [Adult en UCI](https://archive.ics.uci.edu/dataset/2/adult), Becker y Kohavi (1996), [DOI](https://doi.org/10.24432/C5XW20).
- Descarga utilizada: [Adult versión 2 en OpenML, ID 1590](https://www.openml.org/d/1590).
- EDA reutilizado: `Actividad Integradora 1/herramientas_EDA_(3).ipynb`, sección de reutilización mediante clase.
- Unidad 1: `Practica1-EDA.ipynb`, `Practica2_1_FeatureImportance.ipynb` y `Practica2_2_SHAPvalues.ipynb`.
- Unidad 2: `Practica4.ipynb`, `Practica5.ipynb`, `Practica6A_OptBayesiana(1).ipynb` y `Practica6B_OptBayesiana.ipynb`.
- Unida 3 y Unidad 4: `Practica7.ipynb` y `Practica8.ipynb`, revisadas como contexto, no incorporadas como modelos adicionales.
- Herramientas: [scikit-learn](https://scikit-learn.org/stable/user_guide.html), [BayesSearchCV](https://scikit-optimize.readthedocs.io/en/stable/modules/generated/skopt.BayesSearchCV.html) y [TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).

**Límite del ejercicio:** datos históricos, muestra deduplicada, un split y un solo algoritmo. Los resultados y explicaciones no autorizan decisiones reales sobre personas ni demuestran causalidad, equidad o calibración.
