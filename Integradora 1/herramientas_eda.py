import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AutoEDA:
    def __init__(self, df: pd.DataFrame, target_col: str = None):
        """
        Inicializa la clase con el DataFrame y la columna objetivo (opcional).
        """
        self.df = df
        self.target_col = target_col
        self.num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    def _diagnose_missing_and_duplicates(self):
        print("=== 1. DIAGNÓSTICO DE CALIDAD DE DATOS ===")
        # Duplicados
        duplicates = self.df.duplicated().sum()
        print(f" Filas duplicadas: {duplicates} ({duplicates / len(self.df) * 100:.2f}%)")

        # Valores nulos
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print(" No se encontraron valores nulos (NaN/None).\n")
        else:
            print(" Valores nulos por columna:")
            for col, val in missing.items():
                print(f"  - {col}: {val} nulos ({val / len(self.df) * 100:.2f}%)")
            print()

    def _summary_statistics(self):
        print("=== 2. RESUMEN ESTADÍSTICO ===")
        print(f"Variables numéricas ({len(self.num_cols)}): {self.num_cols}")
        print(f"Variables categóricas ({len(self.cat_cols)}): {self.cat_cols}\n")

        print("Estadísticas de variables numéricas:")
        print(self.df[self.num_cols].describe().T[['mean', 'std', 'min', '50%', 'max']])
        print()

    def _detect_outliers(self):
        print("=== 3. DETECCIÓN DE OUTLIERS (IQR) ===")
        for col in self.num_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            print(f"  - {col}: {outliers} outliers detectados ({outliers / len(self.df) * 100:.2f}%)")
        print()

    def _correlation_analysis(self):
        print("=== 4. MATRIZ DE CORRELACIÓN DE PEARSON ===")
        if len(self.num_cols) > 1:
            corr = self.df[self.num_cols].corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title("Matriz de Correlación")
            plt.tight_layout()
            plt.show()
        else:
            print(" No hay suficientes variables numéricas para calcular correlaciones.")
        print()

    def run_analysis(self):
        """
        MÉTODO PRINCIPAL: Dispara el flujo completo de análisis del dataset.
        """
        print(f"INICIANDO ANÁLISIS AUTOMÁTICO DE DATOS ({self.df.shape[0]} filas, {self.df.shape[1]} columnas)\n" + "="*60)
        self._diagnose_missing_and_duplicates()
        self._summary_statistics()
        self._detect_outliers()
        self._correlation_analysis()
        print("="*60 + "\nANÁLISIS COMPLETADO CON ÉXITO.")


