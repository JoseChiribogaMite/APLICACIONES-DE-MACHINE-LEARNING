# -*- coding: utf-8 -*-
"""
herramientas_eda.py — Actividad Integradora 1 · Aplicaciones de Machine Learning
================================================================================
Herramienta PROPIA de Análisis Exploratorio de Datos (EDA) 100 % automatizada,
orientada a modelos de REGRESIÓN. Cubre los 10 puntos de EDA de clase y emite un
diagnóstico estadístico, visual y de calidad de datos para detectar:

  · MULTICOLINEALIDAD  → matriz de correlación de Pearson + VIF (statsmodels).
  · SESGO (bias)       → skewness y curtosis de predictores y del target.
  · VALORES ATÍPICOS   → regla del rango intercuartílico (IQR) de Tukey + boxplots.

⚠️  PRINCIPIO METODOLÓGICO (lectura obligatoria antes de usar):
Esta clase SOLO DIAGNOSTICA Y REPORTA. Nunca elimina filas, imputa nulos ni
transforma variables de forma automática. Toda alerta estadística se presenta
como una OBSERVACIÓN (🔎) cuya acción de limpieza corresponde decidir al
analista, de forma documentada y con criterio de negocio.
"""

# ================== PASO 1. IMPORTAR LIBRERÍAS ==================
import math                                                     # módulo estándar: provee math.ceil() para calcular filas de subgráficos
import pandas as pd                                             # pandas: estructuras tabulares (DataFrame/Series) y operaciones vectorizadas
import numpy as np                                              # numpy: tipos numéricos base y funciones de álgebra/matemáticas
import matplotlib.pyplot as plt                                 # matplotlib: motor de renderizado de figuras (se usa el submódulo pyplot)
import seaborn as sns                                           # seaborn: gráficos estadísticos de alto nivel (histplot, boxplot, heatmap...)

try:                                                            # bloque tolerante: si statsmodels no está instalado, el resto de la clase sigue operativo
    from statsmodels.stats.outliers_influence import variance_inflation_factor  # statsmodels: función VIF para diagnosticar multicolinealidad
except ImportError:                                             # se captura la ausencia del paquete para degradar con elegancia (sin romper el EDA)
    variance_inflation_factor = None                            # marcador None: los métodos de VIF avisarán que el paquete no está disponible

# Constantes de los criterios estadísticos usados como umbrales de OBSERVACIÓN
UMBRAL_SKEW = 1.0      # |skewness| > 1.0  → distribución asimétrica que conviene evaluar (no transformar automáticamente)
UMBRAL_CORR = 0.80     # |r de Pearson| >= 0.80 → par de predictores con correlación lineal alta
UMBRAL_VIF_MOD = 5.0   # VIF > 5  → indicio moderado de multicolinealidad
UMBRAL_VIF_ALTO = 10.0 # VIF > 10 → indicio severo de multicolinealidad
MULTIPLO_IQR = 1.5     # factor de Tukey: Q1 - 1.5·IQR y Q3 + 1.5·IQR delimitan los valores "atípicos leves"
MULTIPLO_IQR_EXTREMO = 3.0  # factor 3.0·IQR delimita los valores "atípicos extremos"


def _mostrar(objeto):
    """Muestra un DataFrame/objeto con la representación enriquecida si está disponible (notebook) o texto plano (script)."""
    try:                                                        # intenta primero el camino interactivo de Jupyter/Colab
        from IPython.display import display                     # display() renderiza tablas HTML interactivas dentro del notebook
        display(objeto)                                         # despliega el objeto usando su representación gráfica nativa
    except ImportError:                                         # fuera de un notebook no existe IPython.display
        print(objeto)                                           # respaldo textual: print() funciona en cualquier entorno Python


class AutoEDA:
    """Clase reutilizable que ejecuta los 10 puntos de EDA de clase sobre un DataFrame, sin modificar los datos originales."""

    def __init__(self, df, target_col=None, figsize=(10, 6)):
        """Constructor: recibe el DataFrame y (opcionalmente) el nombre de la variable objetivo.

        Parámetros
        ----------
        df : pd.DataFrame
            Dataset a diagnosticar. Se trabaja sobre una COPIA para no mutar los datos del usuario.
        target_col : str, opcional
            Nombre de la variable dependiente (target). Si es numérica se asume un problema de REGRESIÓN.
        figsize : tuple, opcional
            Tamaño (ancho, alto) en pulgadas de las figuras de matplotlib.
        """
        self.df = df.copy()                                     # copia defensiva: ningún método de la clase altera el DataFrame original
        self.target_col = target_col                            # guarda el nombre del target para los análisis bivariados (puntos 8 y 10)
        self.figsize = figsize                                  # guarda el tamaño por defecto de las figuras generadas
        self.num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()   # columnas numéricas: enteros y flotantes (incluye booleanos 0/1)
        self.cat_cols = self.df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()  # columnas categóricas/texto
        self.hallazgos = []                                     # bitácora: aquí se registran todas las OBSERVACIONES para la interpretación final

    # ----------------------------------------------------------------
    def _registrar(self, punto, tipo, detalle):
        """Registra una observación en la bitácora interna y la imprime con formato consistente.

        Parámetros
        ----------
        punto : int
            Número del punto de EDA (2 a 10) al que pertenece la observación.
        tipo : str
            Etiqueta corta de la observación (ej.: 'outliers', 'sesgo', 'multicolinealidad').
        detalle : str
            Texto explicativo con la evidencia numérica y la acción a EVALUAR (no a ejecutar).
        """
        self.hallazgos.append({'punto': punto, 'tipo': tipo, 'detalle': detalle})  # almacena la observación para el resumen final
        print(f"🔎 OBSERVACIÓN [P{punto} · {tipo}]: {detalle}")                     # imprime la observación: es un aviso, NUNCA una acción automática

    # ================== PASO 2. CONOCER EL DATASET ==================
    def _paso2_conocer_dataset(self):
        """Punto 2 — Conocer el dataset: dimensiones, memoria, catálogo de columnas y muestras (head/tail/sample)."""
        print("\n=== PASO 2. CONOCER EL DATASET ===")           # encabezado del bloque para lectura ágil del reporte
        print(f"Dimensiones: {self.df.shape[0]} filas × {self.df.shape[1]} columnas")   # .shape devuelve (n_filas, n_columnas)
        print(f"Memoria RAM estimada: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")  # deep=True mide también los objetos internos de las cadenas
        print(f"Catálogo de columnas ({len(self.df.columns)}): {self.df.columns.tolist()}")     # .columns lista los nombres de todas las variables
        print("--- Primeras 3 filas (head) ---")                # head() inspecciona el inicio del dataset
        _mostrar(self.df.head(3))                               # muestra las 3 primeras filas para validar formato y semántica
        print("--- Últimas 3 filas (tail) ---")                 # tail() inspecciona el final (detecta registros truncados o mal cargados)
        _mostrar(self.df.tail(3))                               # muestra las 3 últimas filas
        print("--- Muestra aleatoria (sample) ---")             # sample() inspecciona filas al azar, útiles en datasets ordenados por fecha
        _mostrar(self.df.sample(3, random_state=42))            # random_state=42 fija la semilla para que la muestra sea reproducible

    # ================== PASO 3. TIPO DE VARIABLE ==================
    def _paso3_tipos_variable(self):
        """Punto 3 — Tipo de variable: clasifica cada columna (numérica vs. categórica) mediante dtypes/select_dtypes."""
        print("\n=== PASO 3. TIPO DE VARIABLE ===")             # encabezado del bloque
        tabla_tipos = pd.DataFrame({                           # construye una tabla resumen de tipos
            'tipo': self.df.dtypes.astype(str),                 # .dtypes devuelve el tipo pandas de cada columna (int64, float64, object...)
            'valores_unicos': self.df.nunique(),                # .nunique() cuenta valores distintos: útil para detectar IDs de alta cardinalidad
            'faltantes': self.df.isnull().sum(),                # conteo rápido de nulos como vista previa del punto 5
        })
        _mostrar(tabla_tipos)                                   # muestra la tabla de tipos
        print(f"Numéricas ({len(self.num_cols)}): {self.num_cols}")   # num_cols se calculó con select_dtypes(include=[np.number])
        print(f"Categóricas ({len(self.cat_cols)}): {self.cat_cols}") # cat_cols se calculó con select_dtypes(include=['object','string','category'])
        for c in self.cat_cols:                                 # recorre las categóricas buscando identificadores
            if self.df[c].nunique() == len(self.df):            # si hay tantos valores únicos como filas, la columna identifica cada registro
                self._registrar(3, 'identificador',             # registra la observación en la bitácora
                                f"'{c}' tiene {self.df[c].nunique()} valores únicos (uno por fila): "
                                f"evaluar si es un ID sin valor predictivo.")

    # ================== PASO 4. CALIDAD DE DATOS ==================
    def _paso4_calidad_datos(self):
        """Punto 4 — Calidad de datos: duplicados exactos y valores centinela ('?', '-', 'N/A', espacios sobrantes)."""
        print("\n=== PASO 4. CALIDAD DE DATOS ===")             # encabezado del bloque
        n_dup = self.df.duplicated().sum()                      # .duplicated() marca las filas idénticas a una anterior; .sum() las cuenta
        print(f"Filas duplicadas exactas: {n_dup} ({100 * n_dup / len(self.df):.2f}%)")  # porcentaje sobre el total de filas
        if n_dup > 0:                                           # solo emite observación si realmente existen duplicados
            self._registrar(4, 'duplicados',                    # registra la observación (no los elimina: decisión humana)
                            f"{n_dup} filas duplicadas detectadas: evaluar si se eliminan con "
                            f"df.drop_duplicates() o si corresponden a registros legítimos repetidos.")
        centinelas = ['?', '-', 'n/a', 'na', 'none', 'null', '']  # lista de valores típicos que "disfrazan" datos ausentes
        for c in self.cat_cols:                                 # la búsqueda de centinelas se aplica a columnas de texto
            normalizada = self.df[c].astype(str).str.strip().str.lower()  # convierte a texto, quita espacios y pasa a minúsculas para comparar
            n_raros = normalizada.isin(centinelas).sum()        # cuenta coincidencias exactas con los valores centinela
            n_espacios = (self.df[c].astype(str) != self.df[c].astype(str).str.strip()).sum()  # detecta textos con espacios al inicio o final
            if n_raros > 0 or n_espacios > 0:                   # si hay hallazgos en esta columna...
                self._registrar(4, 'valores_raros',             # ...se registra la observación con el detalle numérico
                                f"'{c}': {n_raros} valor(es) centinela ('?', '-', 'N/A'...) y "
                                f"{n_espacios} texto(s) con espacios sobrantes: evaluar normalización manual.")
            else:                                               # si la columna está limpia...
                print(f"✓ '{c}': sin valores centinela ni espacios sobrantes.")  # ...se reporta explícitamente como verificada

    # ================== PASO 5. VALORES FALTANTES ==================
    def _paso5_valores_faltantes(self):
        """Punto 5 — Valores faltantes: conteo, porcentaje y clasificación de severidad (sin imputar nada)."""
        print("\n=== PASO 5. VALORES FALTANTES ===")            # encabezado del bloque
        nulos = self.df.isnull().sum()                          # .isnull() marca cada celda vacía; .sum() cuenta los nulos por columna
        pct = (nulos / len(self.df)) * 100                      # convierte el conteo a porcentaje sobre el total de filas
        tabla_nulos = pd.DataFrame({'nulos': nulos, 'porcentaje(%)': pct.round(2)})  # tabla resumen ordenable por severidad
        _mostrar(tabla_nulos)                                   # muestra la tabla completa de nulos
        for c in self.df.columns:                               # recorre todas las columnas del dataset
            if nulos[c] > 0:                                    # solo clasifica las columnas que tienen nulos
                if pct[c] < 5:                                  # umbral 1: menos del 5 % de nulos
                    severidad = 'baja → evaluar imputación simple (media/mediana/moda)'
                elif pct[c] <= 30:                              # umbral 2: entre 5 % y 30 % de nulos
                    severidad = 'media → evaluar imputación avanzada o modelado específico'
                else:                                           # más del 30 % de nulos
                    severidad = 'alta → evaluar descarte de la columna o tratamiento especial'
                self._registrar(5, 'nulos',                     # registra la observación indicando severidad y opciones a EVALUAR
                                f"'{c}': {nulos[c]} nulos ({pct[c]:.2f}%) → severidad {severidad}. "
                                f"La herramienta NO imputa: la estrategia la define el analista.")
        if nulos.sum() == 0:                                    # caso más frecuente en datasets limpios
            print("✓ No se detectaron valores nulos (NaN/None): no se requiere estrategia de imputación.")  # confirma la ausencia de nulos

    # ================== PASO 6. OUTLIERS ==================
    def _paso6_outliers(self):
        """Punto 6 — Outliers: regla del IQR de Tukey (1.5× y 3.0×) sobre cada variable numérica + boxplots."""
        print("\n=== PASO 6. OUTLIERS (REGLA DEL IQR) ===")     # encabezado del bloque
        filas = []                                              # acumulador de resúmenes por variable
        for c in self.num_cols:                                 # la regla del IQR solo aplica a variables numéricas
            q1 = self.df[c].quantile(0.25)                      # primer cuartil Q1: valor bajo el cual queda el 25 % de los datos
            q3 = self.df[c].quantile(0.75)                      # tercer cuartil Q3: valor bajo el cual queda el 75 % de los datos
            iqr = q3 - q1                                       # rango intercuartílico: dispersión del 50 % central
            lim_inf = q1 - MULTIPLO_IQR * iqr                   # límite inferior de Tukey (1.5 × IQR)
            lim_sup = q3 + MULTIPLO_IQR * iqr                   # límite superior de Tukey (1.5 × IQR)
            n_out = ((self.df[c] < lim_inf) | (self.df[c] > lim_sup)).sum()   # cuenta valores fuera del intervalo [lim_inf, lim_sup]
            n_ext = ((self.df[c] < q1 - MULTIPLO_IQR_EXTREMO * iqr) |        # umbral 3.0 × IQR para atípicos extremos
                     (self.df[c] > q3 + MULTIPLO_IQR_EXTREMO * iqr)).sum()
            filas.append({'variable': c, 'Q1': round(q1, 2), 'Q3': round(q3, 2), 'IQR': round(iqr, 2),
                          'atipicos': n_out, 'atipicos_%': round(100 * n_out / len(self.df), 2),
                          'extremos': n_ext})                    # agrega el resumen de la variable a la lista
            if n_out > 0:                                       # emite observación solo si la variable tiene atípicos
                self._registrar(6, 'outliers',                  # registra la observación: sugiere VERIFICAR, nunca borrar
                                f"'{c}': {n_out} valores atípicos ({100 * n_out / len(self.df):.2f}%) y {n_ext} extremos. "
                                f"Verificar si son anomalías reales o errores de captura antes de decidir su tratamiento.")
        _mostrar(pd.DataFrame(filas))                           # muestra la tabla resumen de outliers
        ncols = 3                                               # número de boxplots por fila en la cuadrícula
        nrows = math.ceil(len(self.num_cols) / ncols)           # calcula cuántas filas de subgráficos se necesitan
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))  # crea la cuadrícula de ejes
        axes = axes.flatten()                                   # convierte la matriz de ejes en lista plana para iterar fácil
        for i, c in enumerate(self.num_cols):                   # recorre cada variable numérica con su subgráfico
            sns.boxplot(data=self.df, x=c, ax=axes[i])          # boxplot horizontal: mediana, caja Q1–Q3, bigotes y puntos atípicos
            axes[i].set_title(f'Boxplot de {c}')                # titula el subgráfico con el nombre de la variable
        for j in range(len(self.num_cols), len(axes)):          # recorre los ejes sobrantes (si sobran celdas en la cuadrícula)
            axes[j].axis('off')                                 # apaga los ejes vacíos para una figura limpia
        plt.tight_layout()                                      # ajusta márgenes para que títulos y etiquetas no se solapen
        plt.show()                                              # renderiza la figura (en notebook: salida en línea; en script: ventana/archivo)

    # ================== PASO 7. ESTADÍSTICA DESCRIPTIVA ==================
    def _paso7_estadistica_descriptiva(self):
        """Punto 7 — Estadística descriptiva: tendencia central, dispersión, asimetría y curtosis de las numéricas."""
        print("\n=== PASO 7. ESTADÍSTICA DESCRIPTIVA ===")      # encabezado del bloque
        if not self.num_cols:                                   # si no hay variables numéricas no hay nada que describir
            print("⚠ No hay variables numéricas en el dataset.")  # aviso explícito en lugar de fallar
            return                                              # termina el método sin error
        desc = self.df[self.num_cols].describe().T              # describe() calcula count, mean, std, min, cuartiles y max; .T transpone
        desc['rango'] = desc['max'] - desc['min']               # amplitud total: distancia entre el máximo y el mínimo
        desc['cv'] = desc.apply(                                # coeficiente de variación CV = std/media (dispersión relativa)
            lambda fila: (fila['std'] / fila['mean']) if fila['mean'] != 0 else np.nan,  # evita dividir entre cero cuando media=0
            axis=1).round(2)                                    # axis=1 aplica la función fila por fila; .round(2) limita decimales
        desc['skewness'] = self.df[self.num_cols].skew().round(2)   # skew(): asimetría muestral (>0 cola derecha, <0 cola izquierda)
        desc['curtosis'] = self.df[self.num_cols].kurt().round(2)   # kurt(): curtosis EXCESIVA (0 = normal; >0 colas pesadas)
        _mostrar(desc[['mean', 'std', 'min', '50%', 'max', 'rango', 'cv', 'skewness', 'curtosis']])  # muestra las columnas relevantes
        print("· '50%' es la mediana (percentil 50): si difiere mucho de la media, la distribución está sesgada.")  # ayuda interpretativa

    # ================== PASO 8. DESBALANCE / TARGET ==================
    def _paso8_target(self):
        """Punto 8 — Variable objetivo: distribución (regresión, target continuo) o balance de clases (clasificación)."""
        print("\n=== PASO 8. VARIABLE OBJETIVO (TARGET) ===")   # encabezado del bloque
        if self.target_col is None:                             # la clase también debe funcionar sin target (análisis exploratorio puro)
            print("ℹ No se declaró target_col: se omite el punto 8 (pasar target_col='nombre' al instanciar).")
            return                                              # termina el método sin error
        if self.target_col not in self.df.columns:              # valida que el nombre del target exista realmente en el DataFrame
            print(f"⚠ El target '{self.target_col}' no existe entre las columnas: {self.df.columns.tolist()}")
            return                                              # termina el método sin error
        if self.target_col in self.num_cols:                    # RAMA REGRESIÓN: el target es continuo
            s = self.df[self.target_col]                        # Serie con los valores del target
            print(f"Target continuo '{self.target_col}': media={s.mean():.2f} · mediana={s.median():.2f} · "
                  f"min={s.min():.2f} · max={s.max():.2f} · skew={s.skew():.2f}")  # resumen de forma y ubicación
            if abs(s.skew()) > UMBRAL_SKEW:                     # compara la asimetría con el umbral de observación
                self._registrar(8, 'sesgo_target',              # registra la observación sobre el target
                                f"El target '{self.target_col}' tiene |skew|={abs(s.skew()):.2f} > {UMBRAL_SKEW}: "
                                f"la media se aleja de la mediana. Evaluar transformación log(1+x) o Box-Cox "
                                f"SOLO si el modelo lo requiere (decisión humana, no automática).")
            plt.figure(figsize=self.figsize)                    # crea una figura nueva con el tamaño por defecto
            sns.histplot(s, kde=True, bins=30)                  # histograma con curva de densidad KDE: revela la forma de la distribución
            plt.axvline(s.mean(), color='red', linestyle='--', label=f'media={s.mean():.0f}')    # línea vertical en la media
            plt.axvline(s.median(), color='green', linestyle='--', label=f'mediana={s.median():.0f}')  # línea vertical en la mediana
            plt.title(f'Distribución del target: {self.target_col}')  # título descriptivo de la figura
            plt.legend()                                        # muestra la leyenda con media y mediana
            plt.show()                                          # renderiza la figura
        else:                                                   # RAMA CLASIFICACIÓN: el target es categórico
            conteos = self.df[self.target_col].value_counts(normalize=True) * 100  # value_counts(normalize=True) devuelve proporciones por clase
            print(f"Distribución de clases del target '{self.target_col}' (%):")  # encabezado de la tabla de balance
            _mostrar(conteos.round(2))                          # muestra los porcentajes de cada clase
            if conteos.min() < 20:                              # regla práctica: clase minoritaria por debajo del 20 % = desbalance
                self._registrar(8, 'desbalance',                # registra la observación de desbalance
                                f"La clase minoritaria '{conteos.idxmin()}' representa {conteos.min():.1f}%: "
                                f"evaluar técnicas de balanceo (SMOTE, class_weight) según el modelo elegido.")
            plt.figure(figsize=self.figsize)                    # crea la figura
            sns.countplot(data=self.df, x=self.target_col)      # countplot: barras con la frecuencia de cada clase
            plt.title(f'Balance de clases: {self.target_col}')  # título de la figura
            plt.show()                                          # renderiza la figura

    # ================== PASO 9. ANÁLISIS UNIVARIADO ==================
    def _paso9_analisis_univariado(self):
        """Punto 9 — Análisis univariado: histogramas + KDE y diagnóstico de sesgo (skewness) y curtosis por variable."""
        print("\n=== PASO 9. ANÁLISIS UNIVARIADO (DISTRIBUCIONES) ===")  # encabezado del bloque
        if not self.num_cols:                                   # guarda: sin numéricas no aplica
            print("⚠ No hay variables numéricas en el dataset.")
            return                                              # termina el método sin error
        tabla_forma = pd.DataFrame({                           # tabla comparativa de forma de cada distribución
            'skewness': self.df[self.num_cols].skew().round(2),     # asimetría por variable
            'curtosis': self.df[self.num_cols].kurt().round(2),     # curtosis excesiva por variable
        })
        _mostrar(tabla_forma)                                   # muestra la tabla de forma
        for c in self.num_cols:                                 # recorre cada variable numérica
            if abs(tabla_forma.loc[c, 'skewness']) > UMBRAL_SKEW:  # aplica el umbral de sesgo severo
                self._registrar(9, 'sesgo',                     # registra la observación de asimetría
                                f"'{c}': |skew|={abs(tabla_forma.loc[c, 'skewness']):.2f} > {UMBRAL_SKEW} → "
                                f"distribución asimétrica. Evaluar transformación (log(1+x), Box-Cox) según el modelo: "
                                f"no es una acción automática.")
        ncols = 3                                               # número de histogramas por fila
        nrows = math.ceil(len(self.num_cols) / ncols)           # filas necesarias de la cuadrícula
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))  # crea la cuadrícula de ejes
        axes = axes.flatten()                                   # aplana los ejes para iterar
        for i, c in enumerate(self.num_cols):                   # recorre cada variable con su subgráfico
            sns.histplot(self.df[c], kde=True, bins=30, ax=axes[i], color='skyblue')  # histograma + curva KDE de la variable
            axes[i].set_title(f'Distribución de {c} (skew={tabla_forma.loc[c, "skewness"]})')  # título con la asimetría
        for j in range(len(self.num_cols), len(axes)):          # apaga ejes sobrantes
            axes[j].axis('off')                                 # oculta celdas vacías de la cuadrícula
        plt.tight_layout()                                      # ajusta márgenes
        plt.show()                                              # renderiza la figura

    # ================== PASO 10. BIVARIADO / MULTIVARIADO ==================
    def _paso10_bivariado_multivariado(self):
        """Punto 10 — Bivariado/multivariado: relación con el target, correlación (heatmap) y multicolinealidad (VIF)."""
        print("\n=== PASO 10. ANÁLISIS BIVARIADO Y MULTIVARIADO ===")  # encabezado del bloque
        # --- 10a. Predictores NUMÉRICOS vs target: scatter con recta de tendencia ---
        if self.target_col and self.target_col in self.num_cols:  # solo aplica si hay target numérico declarado
            corr_target = self.df[self.num_cols].corr()[self.target_col].drop(self.target_col)  # correlación Pearson de cada numérica con el target
            corr_target = corr_target.sort_values(key=abs, ascending=False)  # ordena por valor absoluto: las más influyentes primero
            print(f"Correlación de Pearson de cada predictor numérico con '{self.target_col}':")  # encabezado del ranking
            _mostrar(corr_target.round(3))                      # muestra el ranking de correlaciones con el target
            for c in [x for x in self.num_cols if x != self.target_col]:  # recorre los predictores numéricos (excluye el target)
                plt.figure(figsize=self.figsize)                # figura por cada relación predictor–target
                sns.regplot(data=self.df, x=c, y=self.target_col,  # regplot: nube de puntos + recta de regresión lineal
                            scatter_kws={'alpha': 0.35},        # transparencia de los puntos para ver densidad
                            line_kws={'color': 'red'})          # recta de tendencia en rojo
                plt.title(f'{c} vs {self.target_col} (r={corr_target[c]:.3f})')  # título con el coeficiente de Pearson
                plt.show()                                      # renderiza la figura
        # --- 10b. Predictores CATEGÓRICOS vs target: boxplots agrupados ---
        if self.target_col and self.target_col in self.df.columns:  # aplica a cualquier target declarado (numérico o categórico)
            for c in self.cat_cols:                             # recorre cada variable categórica
                plt.figure(figsize=self.figsize)                # figura por categoría
                sns.boxplot(data=self.df, x=c, y=self.target_col)  # boxplot del target por cada categoría: revela efectos de grupo
                plt.title(f'{self.target_col} por {c}')         # título descriptivo
                plt.show()                                      # renderiza la figura
        # --- 10c. Matriz de correlación (heatmap) y pares colineales ---
        if len(self.num_cols) > 1:                              # se necesitan al menos 2 numéricas para correlacionar
            corr = self.df[self.num_cols].corr()                # matriz de correlación lineal de Pearson entre todas las numéricas
            plt.figure(figsize=(8, 6))                          # figura cuadrada para el heatmap
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',  # heatmap anotado: color frío para negativos, cálido para positivos
                        center=0, linewidths=0.5)               # center=0 ancla la escala de color en correlación nula
            plt.title('Matriz de correlación de Pearson (variables numéricas)')  # título del heatmap
            plt.tight_layout()                                  # ajusta márgenes
            plt.show()                                          # renderiza el heatmap
            for i in range(len(self.num_cols)):                 # doble bucle para revisar TODOS los pares sin repetir
                for j in range(i + 1, len(self.num_cols)):      # j inicia en i+1: solo la mitad triangular superior
                    v = corr.iloc[i, j]                         # coeficiente del par (i, j)
                    if abs(v) >= UMBRAL_CORR:                   # aplica el umbral de alta correlación
                        self._registrar(10, 'multicolinealidad',  # registra el par colineal como observación
                                        f"'{self.num_cols[i]}' y '{self.num_cols[j]}' con |r|={abs(v):.2f} ≥ {UMBRAL_CORR}: "
                                        f"evaluar eliminar una de las dos o combinarlas antes de un modelo lineal.")
            if not any(abs(corr.iloc[i, j]) >= UMBRAL_CORR      # verifica si NO hubo ningún par por encima del umbral
                       for i in range(len(self.num_cols)) for j in range(i + 1, len(self.num_cols))):
                print(f"✓ Ningún par de variables numéricas supera |r| = {UMBRAL_CORR}.")  # confirma ausencia de correlación extrema
        # --- 10d. VIF: Factor de Inflación de la Varianza ---
        if variance_inflation_factor is None:                   # si statsmodels no se instaló en el entorno...
            print("⚠ VIF no disponible: instala statsmodels (pip install statsmodels) para el diagnóstico de multicolinealidad.")
            return                                              # termina solo este bloque, no el resto del análisis
        predictores = [c for c in self.num_cols if c != self.target_col]  # excluye el target del cálculo de VIF
        if len(predictores) > 1:                                # el VIF exige al menos 2 predictores
            X = self.df[predictores].dropna()                   # VIF requiere matriz sin nulos ni constantes; dropna garantiza lo primero
            try:                                                # protege contra columnas constantes (varianza cero) que invalidan el VIF
                tabla_vif = pd.DataFrame({                      # construye la tabla de resultados
                    'variable': predictores,                    # nombre de cada predictor
                    'VIF': [round(variance_inflation_factor(X.values, i), 2) for i in range(len(predictores))],  # VIF_i = 1/(1-R²_i) de cada predictor frente a los demás
                })
                _mostrar(tabla_vif)                             # muestra la tabla VIF
                for i, c in enumerate(predictores):             # recorre los resultados para clasificar la severidad
                    if tabla_vif.loc[i, 'VIF'] > UMBRAL_VIF_ALTO:      # VIF > 10: multicolinealidad severa
                        self._registrar(10, 'vif',              # registra la observación severa
                                        f"'{c}': VIF={tabla_vif.loc[i, 'VIF']:.2f} > {UMBRAL_VIF_ALTO} → multicolinealidad severa: "
                                        f"evaluar eliminar la variable o combinarla con otra.")
                    elif tabla_vif.loc[i, 'VIF'] > UMBRAL_VIF_MOD:     # VIF entre 5 y 10: multicolinealidad moderada
                        self._registrar(10, 'vif',              # registra la observación moderada
                                        f"'{c}': VIF={tabla_vif.loc[i, 'VIF']:.2f} > {UMBRAL_VIF_MOD} → multicolinealidad moderada: "
                                        f"monitorear su efecto sobre los coeficientes del modelo.")
                if (tabla_vif['VIF'] <= UMBRAL_VIF_MOD).all():  # si todos los VIF quedan por debajo del umbral moderado
                    print(f"✓ VIF de todos los predictores ≤ {UMBRAL_VIF_MOD}: sin indicios de multicolinealidad.")  # confirma diagnóstico sano
            except ValueError:                                  # captura el caso de columnas constantes u otros fallos numéricos
                print("⚠ No se pudo calcular el VIF (posible columna constante o singularidad numérica).")  # aviso sin romper el flujo

    # ================== ORQUESTADOR PRINCIPAL ==================
    def run_analysis(self):
        """Método público: ejecuta los 10 puntos de EDA en orden y de forma 100 % automatizada (una sola llamada)."""
        print("=" * 72)                                         # línea separadora superior del reporte
        print(" DIAGNÓSTICO AUTOMATIZADO DE EDA · ORIENTADO A REGRESIÓN ".center(72))  # título centrado del reporte
        print("=" * 72)                                         # línea separadora
        print("⚠  Esta herramienta SOLO diagnostica y reporta: no elimina filas, no imputa")
        print("   nulos y no transforma variables. Cada acción de limpieza es una decisión")
        print("   HUMANA que debe documentarse en la sección de interpretación.\n")
        print("PASO 1 (importar librerías) se cumple en la cabecera de este módulo:")  # aclaración del punto 1
        print("pandas, numpy, matplotlib, seaborn y statsmodels (para el VIF).\n")
        self._paso2_conocer_dataset()                           # ejecuta el punto 2: conocer el dataset
        self._paso3_tipos_variable()                            # ejecuta el punto 3: tipo de variable
        self._paso4_calidad_datos()                             # ejecuta el punto 4: calidad de datos
        self._paso5_valores_faltantes()                         # ejecuta el punto 5: valores faltantes
        self._paso6_outliers()                                  # ejecuta el punto 6: outliers (IQR + boxplots)
        self._paso7_estadistica_descriptiva()                   # ejecuta el punto 7: estadística descriptiva
        self._paso8_target()                                    # ejecuta el punto 8: target / balance de clases
        self._paso9_analisis_univariado()                       # ejecuta el punto 9: análisis univariado
        self._paso10_bivariado_multivariado()                   # ejecuta el punto 10: bivariado/multivariado + VIF
        print("=" * 72)                                         # línea separadora inferior
        print(f" DIAGNÓSTICO COMPLETADO · {len(self.hallazgos)} observaciones registradas ".center(72))  # total de observaciones
        print("=" * 72)                                         # línea separadora
        print("Para la interpretación final ejecuta: eda.obtener_hallazgos()")  # indica cómo recuperar la bitácora

    def obtener_hallazgos(self):
        """Devuelve la bitácora completa de observaciones como DataFrame, insumo para la sección de interpretación."""
        if not self.hallazgos:                                  # caso sin observaciones registradas
            return pd.DataFrame(columns=['punto', 'tipo', 'detalle'])  # devuelve la tabla vacía con las mismas columnas
        return pd.DataFrame(self.hallazgos)                     # convierte la lista de dictados en un DataFrame tabular
