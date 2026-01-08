"""
Data Preprocessing Module

Módulo para preprocesamiento y feature engineering de datos limpios.
Implementa encoding, normalización y creación de features temporales.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

from .config import (
    PREPROCESSING_CONFIG,
    CLEANED_DATA_PATH,
    PREPROCESSED_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH
)


class DataPreprocessor:
    """
    Clase principal para preprocesamiento y feature engineering.

    Implementa encoding de variables categóricas, normalización de numéricas,
    creación de features temporales y split train/test temporal.
    """

    def __init__(self, config=None):
        """
        Inicializa el preprocesador de datos.

        Args:
            config: Objeto de configuración (default: PREPROCESSING_CONFIG)
        """
        self.config = config if config else PREPROCESSING_CONFIG

        self.scalers: Dict = {}
        self.encoders: Dict = {}
        self.ordinal_mappings: Dict = {}
        self.feature_names: List[str] = []

        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def preprocess_pipeline(
        self,
        df: pd.DataFrame,
        fit_transform: bool = True
    ) -> pd.DataFrame:
        """
        Ejecuta pipeline completo de preprocesamiento.

        Args:
            df: DataFrame limpio
            fit_transform: Si es True, ajusta y transforma. Si es False, solo transforma.

        Returns:
            DataFrame preprocesado
        """
        self.logger.info("Iniciando pipeline de preprocesamiento")

        df_processed = df.copy()

        # 1. Crear features temporales
        df_processed = self.create_temporal_features(df_processed)

        # 2. Crear variable objetivo (target) binaria
        df_processed = self.create_target_variable(df_processed)

        # 3. Encoding de variables nominales (One-Hot)
        df_processed = self.encode_nominal_features(df_processed, fit=fit_transform)

        # 4. Encoding de variables ordinales
        df_processed = self.encode_ordinal_features(df_processed, fit=fit_transform)

        # 5. Normalizar variables numéricas
        df_processed = self.scale_numerical_features(df_processed, fit=fit_transform)

        # 6. Remover columnas originales que ya fueron transformadas
        df_processed = self._remove_original_columns(df_processed)

        # Almacenar nombres de features finales
        self.feature_names = [col for col in df_processed.columns
                             if col not in ['risk_target', self.config.target_column]]

        self.logger.info(f"Preprocesamiento completado. Features finales: {len(self.feature_names)}")

        return df_processed

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features temporales a partir de la columna fecha.

        Args:
            df: DataFrame con columna 'fecha'

        Returns:
            DataFrame con features temporales agregadas
        """
        self.logger.info("Creando features temporales")

        df_temporal = df.copy()

        # Asegurar que fecha es datetime
        if not pd.api.types.is_datetime64_any_dtype(df_temporal['fecha']):
            df_temporal['fecha'] = pd.to_datetime(df_temporal['fecha'])

        # Extraer componentes temporales según configuración
        if 'year' in self.config.temporal_features:
            df_temporal['year'] = df_temporal['fecha'].dt.year

        if 'month' in self.config.temporal_features:
            df_temporal['month'] = df_temporal['fecha'].dt.month

        if 'quarter' in self.config.temporal_features:
            df_temporal['quarter'] = df_temporal['fecha'].dt.quarter

        if 'day_of_week' in self.config.temporal_features:
            df_temporal['day_of_week'] = df_temporal['fecha'].dt.dayofweek

        if 'day_of_month' in self.config.temporal_features:
            df_temporal['day_of_month'] = df_temporal['fecha'].dt.day

        if 'week_of_year' in self.config.temporal_features:
            df_temporal['week_of_year'] = df_temporal['fecha'].dt.isocalendar().week

        if 'is_month_start' in self.config.temporal_features:
            df_temporal['is_month_start'] = df_temporal['fecha'].dt.is_month_start.astype(int)

        if 'is_month_end' in self.config.temporal_features:
            df_temporal['is_month_end'] = df_temporal['fecha'].dt.is_month_end.astype(int)

        if 'is_quarter_start' in self.config.temporal_features:
            df_temporal['is_quarter_start'] = df_temporal['fecha'].dt.is_quarter_start.astype(int)

        if 'is_quarter_end' in self.config.temporal_features:
            df_temporal['is_quarter_end'] = df_temporal['fecha'].dt.is_quarter_end.astype(int)

        created_features = [col for col in df_temporal.columns if col not in df.columns]
        self.logger.info(f"Features temporales creadas: {len(created_features)}")

        return df_temporal

    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea variable objetivo binaria de riesgo de mora.

        score_cumplimiento < threshold = Alto Riesgo (1)
        score_cumplimiento >= threshold = Bajo Riesgo (0)

        Args:
            df: DataFrame con columna score_cumplimiento

        Returns:
            DataFrame con columna 'risk_target' agregada
        """
        self.logger.info(f"Creando variable objetivo con umbral: {self.config.risk_threshold}")

        df_target = df.copy()

        # Crear variable binaria
        df_target['risk_target'] = (
            df_target[self.config.target_column] < self.config.risk_threshold
        ).astype(int)

        # Estadísticas de la variable objetivo
        risk_distribution = df_target['risk_target'].value_counts()
        total = len(df_target)

        self.logger.info("Distribución de riesgo:")
        self.logger.info(f"  Alto Riesgo (1): {risk_distribution.get(1, 0)} ({risk_distribution.get(1, 0)/total*100:.2f}%)")
        self.logger.info(f"  Bajo Riesgo (0): {risk_distribution.get(0, 0)} ({risk_distribution.get(0, 0)/total*100:.2f}%)")

        return df_target

    def encode_nominal_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Aplica One-Hot Encoding a variables nominales.

        Args:
            df: DataFrame con variables nominales
            fit: Si ajustar el encoder (True) o solo transformar (False)

        Returns:
            DataFrame con variables nominales codificadas
        """
        self.logger.info("Aplicando One-Hot Encoding a variables nominales")

        df_encoded = df.copy()

        for column in self.config.nominal_columns:
            if column in df_encoded.columns:
                if fit:
                    # Crear columnas dummy
                    dummies = pd.get_dummies(
                        df_encoded[column],
                        prefix=column,
                        drop_first=False  # Mantener todas las categorías
                    )

                    # Guardar nombres de columnas para uso posterior
                    self.encoders[column] = list(dummies.columns)

                    # Agregar dummies al DataFrame
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)

                    self.logger.info(f"  {column}: {len(dummies.columns)} columnas creadas")
                else:
                    # Usar columnas guardadas previamente
                    if column in self.encoders:
                        dummies = pd.get_dummies(
                            df_encoded[column],
                            prefix=column,
                            drop_first=False
                        )

                        # Asegurar que todas las columnas esperadas existen
                        for expected_col in self.encoders[column]:
                            if expected_col not in dummies.columns:
                                dummies[expected_col] = 0

                        # Mantener solo las columnas que se vieron en entrenamiento
                        dummies = dummies[self.encoders[column]]

                        df_encoded = pd.concat([df_encoded, dummies], axis=1)

        return df_encoded

    def encode_ordinal_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Aplica Ordinal Encoding a variables ordinales.

        Args:
            df: DataFrame con variables ordinales
            fit: Si ajustar el encoder (True) o solo transformar (False)

        Returns:
            DataFrame con variables ordinales codificadas
        """
        self.logger.info("Aplicando Ordinal Encoding a variables ordinales")

        df_encoded = df.copy()

        for column, categories in self.config.ordinal_columns.items():
            if column in df_encoded.columns:
                if fit:
                    # Crear mapping de categorías a números
                    mapping = {cat: idx for idx, cat in enumerate(categories)}
                    self.ordinal_mappings[column] = mapping

                    # Aplicar mapping
                    df_encoded[f'{column}_encoded'] = df_encoded[column].map(mapping)

                    # Manejar valores no vistos (si existen)
                    if df_encoded[f'{column}_encoded'].isnull().any():
                        self.logger.warning(f"Valores no vistos en '{column}', se asignarán a categoría media")
                        median_value = len(categories) // 2
                        df_encoded[f'{column}_encoded'].fillna(median_value, inplace=True)

                    self.logger.info(f"  {column}: {mapping}")
                else:
                    # Usar mapping guardado
                    if column in self.ordinal_mappings:
                        mapping = self.ordinal_mappings[column]
                        df_encoded[f'{column}_encoded'] = df_encoded[column].map(mapping)

                        # Manejar valores no vistos
                        if df_encoded[f'{column}_encoded'].isnull().any():
                            median_value = len(mapping) // 2
                            df_encoded[f'{column}_encoded'].fillna(median_value, inplace=True)

        return df_encoded

    def scale_numerical_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normaliza variables numéricas usando el método configurado.

        Args:
            df: DataFrame con variables numéricas
            fit: Si ajustar el scaler (True) o solo transformar (False)

        Returns:
            DataFrame con variables numéricas normalizadas
        """
        self.logger.info(f"Normalizando variables numéricas con método: {self.config.scaler_method}")

        df_scaled = df.copy()

        # Seleccionar tipo de scaler
        scaler_map = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler
        }

        scaler_class = scaler_map.get(self.config.scaler_method, StandardScaler)

        for column in self.config.numerical_columns:
            if column in df_scaled.columns:
                if fit:
                    # Crear y ajustar scaler
                    scaler = scaler_class()
                    df_scaled[f'{column}_scaled'] = scaler.fit_transform(
                        df_scaled[[column]]
                    )
                    self.scalers[column] = scaler

                    self.logger.info(f"  {column}: normalizado")
                else:
                    # Usar scaler guardado
                    if column in self.scalers:
                        scaler = self.scalers[column]
                        df_scaled[f'{column}_scaled'] = scaler.transform(
                            df_scaled[[column]]
                        )

        return df_scaled

    def _remove_original_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remueve columnas originales que ya fueron transformadas.

        Args:
            df: DataFrame con columnas originales y transformadas

        Returns:
            DataFrame solo con columnas transformadas y necesarias
        """
        df_clean = df.copy()

        # Columnas a remover
        columns_to_remove = []

        # Columnas nominales (ya codificadas con One-Hot)
        columns_to_remove.extend(self.config.nominal_columns)

        # Columnas ordinales (mantener solo la versión encoded)
        for column in self.config.ordinal_columns.keys():
            if column in df_clean.columns:
                columns_to_remove.append(column)

        # Columnas numéricas (mantener solo la versión scaled)
        for column in self.config.numerical_columns:
            if column in df_clean.columns:
                columns_to_remove.append(column)

        # Remover columnas
        columns_to_remove = [col for col in columns_to_remove if col in df_clean.columns]
        df_clean = df_clean.drop(columns=columns_to_remove)

        self.logger.info(f"Columnas originales removidas: {len(columns_to_remove)}")

        return df_clean

    def split_train_test(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide dataset en train y test usando split temporal o aleatorio.

        Args:
            df: DataFrame preprocesado

        Returns:
            Tuple con (train_df, test_df)
        """
        self.logger.info(f"Dividiendo dataset (test_size={self.config.test_size})")

        if self.config.temporal_split:
            # Split temporal (últimas fechas para test)
            self.logger.info("Usando split temporal")

            # Ordenar por fecha
            df_sorted = df.sort_values('fecha').reset_index(drop=True)

            # Calcular índice de corte
            split_idx = int(len(df_sorted) * (1 - self.config.test_size))

            train_df = df_sorted.iloc[:split_idx].copy()
            test_df = df_sorted.iloc[split_idx:].copy()

            self.logger.info(f"  Train: {len(train_df)} registros ({train_df['fecha'].min()} a {train_df['fecha'].max()})")
            self.logger.info(f"  Test: {len(test_df)} registros ({test_df['fecha'].min()} a {test_df['fecha'].max()})")
        else:
            # Split aleatorio estratificado
            self.logger.info("Usando split aleatorio estratificado")

            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=df['risk_target']
            )

            self.logger.info(f"  Train: {len(train_df)} registros")
            self.logger.info(f"  Test: {len(test_df)} registros")

        return train_df, test_df

    def get_feature_names(self) -> List[str]:
        """
        Retorna lista de nombres de features finales.

        Returns:
            Lista de nombres de features
        """
        return self.feature_names

    def export_preprocessed_data(
        self,
        df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> None:
        """
        Exporta datos preprocesados a CSV.

        Args:
            df: DataFrame preprocesado
            output_path: Ruta de salida (default: PREPROCESSED_DATA_PATH)
        """
        if output_path is None:
            output_path = PREPROCESSED_DATA_PATH

        self.logger.info(f"Exportando datos preprocesados a: {output_path}")

        # Crear directorio si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Exportar
        df.to_csv(output_path, index=False, encoding='utf-8')

        self.logger.info(f"Exportación completada: {len(df)} registros")


def preprocess_and_split_data(
    input_path: Optional[str] = None,
    export_files: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, DataPreprocessor]:
    """
    Función helper para preprocesar y dividir datos en un solo paso.

    Args:
        input_path: Ruta del archivo CSV limpio (default: CLEANED_DATA_PATH)
        export_files: Si exportar archivos train/test (default: True)

    Returns:
        Tuple con (train_df, test_df, preprocessor)
    """
    if input_path is None:
        input_path = CLEANED_DATA_PATH

    # Cargar datos limpios
    df = pd.read_csv(input_path)

    # Preprocesar
    preprocessor = DataPreprocessor()
    df_preprocessed = preprocessor.preprocess_pipeline(df, fit_transform=True)

    # Dividir en train/test
    train_df, test_df = preprocessor.split_train_test(df_preprocessed)

    # Exportar si se solicita
    if export_files:
        # Exportar dataset completo preprocesado
        preprocessor.export_preprocessed_data(df_preprocessed)

        # Exportar train
        train_df.to_csv(TRAIN_DATA_PATH, index=False, encoding='utf-8')
        print(f"Train data guardado en: {TRAIN_DATA_PATH}")

        # Exportar test
        test_df.to_csv(TEST_DATA_PATH, index=False, encoding='utf-8')
        print(f"Test data guardado en: {TEST_DATA_PATH}")

    return train_df, test_df, preprocessor