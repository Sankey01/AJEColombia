"""
Data Cleaning Module

Módulo para limpieza y normalización de datos históricos de pagos.
Implementa validación, detección de outliers y normalización de features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging


class DataCleaner:
    """
    Clase principal para limpieza y normalización de datos.

    Implementa validación de tipos, detección de outliers,
    normalización de montos y validación de categorías.
    """

    def __init__(self, outlier_threshold: float = 1.5):
        """
        Inicializa el limpiador de datos.

        Args:
            outlier_threshold: Umbral para detección de outliers usando IQR (default: 1.5)
        """
        self.outlier_threshold = outlier_threshold
        self.cleaning_report: Dict = {}

        # Categorías válidas por columna
        self.valid_categories = {
            'categoria_gasto': ['Tecnología', 'Materiales', 'Logística', 'Servicios'],
            'método_pago': ['Efectivo', 'Cheque', 'Transferencia'],
            'indicadores_economicos': ['Creciente', 'Estable', 'Decreciente'],
            'estacionalidad': ['Alta', 'Baja']
        }

        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta pipeline completo de limpieza de datos.

        Args:
            df: DataFrame original

        Returns:
            DataFrame limpio y validado
        """
        self.logger.info("Iniciando proceso de limpieza de datos")

        df_cleaned = df.copy()

        # Almacenar estadísticas antes de limpieza
        self._record_initial_statistics(df_cleaned)

        # 1. Validar y convertir tipos de datos
        df_cleaned = self._validate_data_types(df_cleaned)

        # 2. Manejar valores nulos
        df_cleaned = self._handle_missing_values(df_cleaned)

        # 3. Normalizar montos
        df_cleaned = self.normalize_amounts(df_cleaned)

        # 4. Validar categorías
        df_cleaned = self.validate_categories(df_cleaned)

        # 5. Detectar y marcar outliers
        df_cleaned = self.detect_outliers(df_cleaned)

        # 6. Validar rangos de valores numéricos
        df_cleaned = self._validate_numerical_ranges(df_cleaned)

        # Almacenar estadísticas después de limpieza
        self._record_final_statistics(df_cleaned)

        self.logger.info("Proceso de limpieza completado")

        return df_cleaned

    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Valida y convierte tipos de datos apropiados.

        Args:
            df: DataFrame a validar

        Returns:
            DataFrame con tipos correctos
        """
        self.logger.info("Validando tipos de datos")

        df_validated = df.copy()

        # Convertir fecha a datetime
        df_validated['fecha'] = pd.to_datetime(df_validated['fecha'], errors='coerce')

        # Validar columnas numéricas
        numeric_columns = ['proveedor_id', 'monto', 'tiempo_procesamiento', 'score_cumplimiento']
        for col in numeric_columns:
            df_validated[col] = pd.to_numeric(df_validated[col], errors='coerce')

        # Validar columnas categóricas (convertir a string)
        categorical_columns = ['categoria_gasto', 'método_pago', 'indicadores_economicos', 'estacionalidad']
        for col in categorical_columns:
            df_validated[col] = df_validated[col].astype(str).str.strip()

        return df_validated

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maneja valores nulos en el dataset.

        Args:
            df: DataFrame con posibles valores nulos

        Returns:
            DataFrame sin valores nulos
        """
        self.logger.info("Manejando valores nulos")

        df_clean = df.copy()

        # Contar valores nulos antes
        null_counts_before = df_clean.isnull().sum()

        # Para columnas numéricas: imputar con mediana
        numeric_columns = ['monto', 'tiempo_procesamiento', 'score_cumplimiento']
        for col in numeric_columns:
            if df_clean[col].isnull().any():
                median_value = df_clean[col].median()
                df_clean[col].fillna(median_value, inplace=True)
                self.logger.warning(
                    f"Imputados {null_counts_before[col]} valores nulos en '{col}' con mediana: {median_value}")

        # Para proveedor_id: eliminar registros (no se puede imputar)
        if df_clean['proveedor_id'].isnull().any():
            rows_before = len(df_clean)
            df_clean = df_clean.dropna(subset=['proveedor_id'])
            rows_dropped = rows_before - len(df_clean)
            self.logger.warning(f"Eliminados {rows_dropped} registros con proveedor_id nulo")

        # Para fecha: eliminar registros (crítico para análisis temporal)
        if df_clean['fecha'].isnull().any():
            rows_before = len(df_clean)
            df_clean = df_clean.dropna(subset=['fecha'])
            rows_dropped = rows_before - len(df_clean)
            self.logger.warning(f"Eliminados {rows_dropped} registros con fecha nula")

        # Para columnas categóricas: imputar con moda
        categorical_columns = ['categoria_gasto', 'método_pago', 'indicadores_economicos', 'estacionalidad']
        for col in categorical_columns:
            if df_clean[col].isnull().any() or (df_clean[col] == 'nan').any():
                mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col].fillna(mode_value, inplace=True)
                df_clean[col] = df_clean[col].replace('nan', mode_value)
                self.logger.warning(f"Imputados valores nulos en '{col}' con moda: {mode_value}")

        return df_clean

    def normalize_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza la columna de montos.

        Redondea a 2 decimales y valida valores positivos.

        Args:
            df: DataFrame con columna 'monto'

        Returns:
            DataFrame con montos normalizados
        """
        self.logger.info("Normalizando montos")

        df_normalized = df.copy()

        # Redondear a 2 decimales
        df_normalized['monto'] = df_normalized['monto'].round(2)

        # Validar que montos sean positivos
        negative_amounts = (df_normalized['monto'] <= 0).sum()
        if negative_amounts > 0:
            self.logger.warning(f"Encontrados {negative_amounts} montos negativos o cero - serán eliminados")
            df_normalized = df_normalized[df_normalized['monto'] > 0]

        return df_normalized

    def validate_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Valida que las categorías contengan valores válidos.

        Args:
            df: DataFrame con columnas categóricas

        Returns:
            DataFrame con categorías validadas
        """
        self.logger.info("Validando categorías")

        df_validated = df.copy()

        for column, valid_values in self.valid_categories.items():
            if column in df_validated.columns:
                invalid_mask = ~df_validated[column].isin(valid_values)
                invalid_count = invalid_mask.sum()

                if invalid_count > 0:
                    invalid_values = df_validated.loc[invalid_mask, column].unique()
                    self.logger.warning(
                        f"Columna '{column}': {invalid_count} valores inválidos encontrados: {invalid_values}"
                    )

                    # Reemplazar valores inválidos con el más frecuente válido
                    mode_value = df_validated[column].mode()[0] if not df_validated[column].mode().empty else \
                    valid_values[0]
                    df_validated.loc[invalid_mask, column] = mode_value
                    self.logger.info(f"Valores inválidos en '{column}' reemplazados con: {mode_value}")

        return df_validated

    def detect_outliers(
            self,
            df: pd.DataFrame,
            columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Detecta outliers usando método IQR.

        Agrega columna booleana indicando si es outlier en alguna variable numérica.

        Args:
            df: DataFrame a analizar
            columns: Lista de columnas a analizar (default: ['monto', 'tiempo_procesamiento'])

        Returns:
            DataFrame con columna 'is_outlier' agregada
        """
        self.logger.info("Detectando outliers")

        df_outliers = df.copy()

        if columns is None:
            columns = ['monto', 'tiempo_procesamiento']

        outlier_mask = pd.Series(False, index=df_outliers.index)
        outlier_details = {}

        for col in columns:
            if col in df_outliers.columns:
                Q1 = df_outliers[col].quantile(0.25)
                Q3 = df_outliers[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - self.outlier_threshold * IQR
                upper_bound = Q3 + self.outlier_threshold * IQR

                col_outliers = (df_outliers[col] < lower_bound) | (df_outliers[col] > upper_bound)
                outlier_mask = outlier_mask | col_outliers

                outlier_count = col_outliers.sum()
                outlier_details[col] = {
                    'count': int(outlier_count),
                    'percentage': round(outlier_count / len(df_outliers) * 100, 2),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2)
                }

                self.logger.info(
                    f"Columna '{col}': {outlier_count} outliers ({outlier_details[col]['percentage']}%) "
                    f"fuera de rango [{lower_bound:.2f}, {upper_bound:.2f}]"
                )

        df_outliers['is_outlier'] = outlier_mask

        total_outliers = outlier_mask.sum()
        self.logger.info(
            f"Total de registros con outliers: {total_outliers} ({total_outliers / len(df_outliers) * 100:.2f}%)")

        # Guardar detalles en reporte
        self.cleaning_report['outliers'] = outlier_details
        self.cleaning_report['total_outliers'] = int(total_outliers)

        return df_outliers

    def _validate_numerical_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Valida rangos de variables numéricas.

        Args:
            df: DataFrame a validar

        Returns:
            DataFrame con valores en rangos válidos
        """
        self.logger.info("Validando rangos numéricos")

        df_validated = df.copy()

        # Validar score_cumplimiento en rango [0, 1]
        invalid_score = ((df_validated['score_cumplimiento'] < 0) |
                         (df_validated['score_cumplimiento'] > 1))
        invalid_count = invalid_score.sum()

        if invalid_count > 0:
            self.logger.warning(f"Encontrados {invalid_count} valores de score_cumplimiento fuera de rango [0,1]")
            df_validated.loc[invalid_score, 'score_cumplimiento'] = df_validated['score_cumplimiento'].clip(0, 1)

        # Validar tiempo_procesamiento > 0
        invalid_time = df_validated['tiempo_procesamiento'] <= 0
        invalid_count = invalid_time.sum()

        if invalid_count > 0:
            self.logger.warning(f"Encontrados {invalid_count} valores de tiempo_procesamiento <= 0")
            df_validated = df_validated[~invalid_time]

        return df_validated

    def _record_initial_statistics(self, df: pd.DataFrame) -> None:
        """
        Registra estadísticas antes de limpieza.

        Args:
            df: DataFrame original
        """
        self.cleaning_report['initial'] = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'null_values': int(df.isnull().sum().sum()),
            'duplicated_rows': int(df.duplicated().sum()),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2)
        }

    def _record_final_statistics(self, df: pd.DataFrame) -> None:
        """
        Registra estadísticas después de limpieza.

        Args:
            df: DataFrame limpio
        """
        self.cleaning_report['final'] = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'null_values': int(df.isnull().sum().sum()),
            'duplicated_rows': int(df.duplicated().sum()),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2)
        }

        # Calcular diferencias
        self.cleaning_report['changes'] = {
            'records_removed': self.cleaning_report['initial']['total_records'] - len(df),
            'null_values_removed': self.cleaning_report['initial']['null_values'] - self.cleaning_report['final'][
                'null_values']
        }

    def get_cleaning_report(self) -> Dict:
        """
        Retorna reporte completo de limpieza.

        Returns:
            Diccionario con estadísticas de limpieza
        """
        return self.cleaning_report

    def print_cleaning_report(self) -> None:
        """
        Imprime reporte de limpieza en consola.
        """
        print("\n" + "=" * 80)
        print("REPORTE DE LIMPIEZA DE DATOS")
        print("=" * 80)

        if 'initial' in self.cleaning_report:
            print("\n📊 ESTADÍSTICAS INICIALES:")
            for key, value in self.cleaning_report['initial'].items():
                print(f"  - {key}: {value:,}")

        if 'final' in self.cleaning_report:
            print("\n✅ ESTADÍSTICAS FINALES:")
            for key, value in self.cleaning_report['final'].items():
                print(f"  - {key}: {value:,}")

        if 'changes' in self.cleaning_report:
            print("\n📈 CAMBIOS REALIZADOS:")
            for key, value in self.cleaning_report['changes'].items():
                print(f"  - {key}: {value:,}")

        if 'outliers' in self.cleaning_report:
            print("\n⚠️  OUTLIERS DETECTADOS:")
            for col, details in self.cleaning_report['outliers'].items():
                print(f"  - {col}:")
                print(f"    * Cantidad: {details['count']} ({details['percentage']}%)")
                print(f"    * Rango válido: [{details['lower_bound']}, {details['upper_bound']}]")

        print("\n" + "=" * 80)

    def export_cleaned_data(
            self,
            df: pd.DataFrame,
            output_path: str,
            include_outliers: bool = True
    ) -> None:
        """
        Exporta dataset limpio a CSV.

        Args:
            df: DataFrame limpio
            output_path: Ruta de salida para el CSV
            include_outliers: Si incluir registros marcados como outliers (default: True)
        """
        self.logger.info(f"Exportando datos limpios a: {output_path}")

        df_export = df.copy()

        if not include_outliers and 'is_outlier' in df_export.columns:
            df_export = df_export[df_export['is_outlier'] == False]
            self.logger.info(f"Outliers excluidos. Registros exportados: {len(df_export)}")

        # Crear directorio si no existe
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Exportar a CSV
        df_export.to_csv(output_path, index=False, encoding='utf-8')

        self.logger.info(f"Exportación completada: {len(df_export)} registros guardados")


def load_and_clean_data(
        input_path: str,
        output_path: Optional[str] = None,
        outlier_threshold: float = 1.5
) -> Tuple[pd.DataFrame, Dict]:
    """
    Función helper para cargar y limpiar datos en un solo paso.

    Args:
        input_path: Ruta del archivo CSV de entrada
        output_path: Ruta opcional para guardar datos limpios
        outlier_threshold: Umbral para detección de outliers

    Returns:
        Tuple con (DataFrame limpio, Reporte de limpieza)
    """
    # Cargar datos
    df = pd.read_csv(input_path)

    # Limpiar datos
    cleaner = DataCleaner(outlier_threshold=outlier_threshold)
    df_cleaned = cleaner.clean_data(df)

    # Imprimir reporte
    cleaner.print_cleaning_report()

    # Exportar si se especifica ruta
    if output_path:
        cleaner.export_cleaned_data(df_cleaned, output_path)

    return df_cleaned, cleaner.get_cleaning_report()