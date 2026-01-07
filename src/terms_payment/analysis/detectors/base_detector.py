"""
Base Detector Module

Define la interfaz abstracta que todos los detectores de anomalías deben implementar.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List


class BaseDetector(ABC):
    """
    Clase base abstracta para todos los detectores de anomalías.

    Todos los detectores deben heredar de esta clase e implementar
    los métodos abstractos: detect() y get_statistics()
    """

    def __init__(self, name: str):
        """
        Inicializa el detector base.

        Args:
            name: Nombre del detector
        """
        self.name = name
        self.anomalies_found: int = 0
        self.total_records: int = 0

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta anomalías en el DataFrame proporcionado.

        Args:
            df: DataFrame con los datos a analizar

        Returns:
            DataFrame con columnas adicionales indicando anomalías:
            - has_anomaly: bool - Si el registro tiene anomalía
            - anomaly_type: str - Tipo específico de anomalía
            - anomaly_description: str - Descripción detallada
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas sobre las anomalías detectadas.

        Returns:
            Diccionario con estadísticas relevantes del detector
        """
        pass

    def _add_anomaly_columns(
        self,
        df: pd.DataFrame,
        anomaly_mask: pd.Series,
        anomaly_type: str,
        descriptions: pd.Series
    ) -> pd.DataFrame:
        """
        Agrega columnas estandarizadas de anomalías al DataFrame.

        Args:
            df: DataFrame original
            anomaly_mask: Serie booleana indicando qué registros tienen anomalía
            anomaly_type: Tipo de anomalía (ej: 'DUPLICATE', 'INVALID_APPROVER')
            descriptions: Serie con descripciones detalladas por registro

        Returns:
            DataFrame con columnas de anomalía agregadas
        """
        df_result = df.copy()

        # Agregar columna de anomalía detectada
        df_result[f'{self.name}_has_anomaly'] = anomaly_mask

        # Agregar tipo de anomalía
        df_result[f'{self.name}_type'] = anomaly_mask.apply(
            lambda x: anomaly_type if x else None
        )

        # Agregar descripción
        df_result[f'{self.name}_description'] = descriptions

        return df_result

    def _calculate_statistics(
        self,
        df: pd.DataFrame,
        anomaly_column: str
    ) -> Dict:
        """
        Calcula estadísticas básicas sobre las anomalías detectadas.

        Args:
            df: DataFrame con anomalías detectadas
            anomaly_column: Nombre de la columna con flag de anomalía

        Returns:
            Diccionario con estadísticas
        """
        total = len(df)
        anomalies = df[anomaly_column].sum() if anomaly_column in df.columns else 0

        stats = {
            'detector_name': self.name,
            'total_records': total,
            'anomalies_found': int(anomalies),
            'anomaly_percentage': round((anomalies / total * 100) if total > 0 else 0, 2),
            'clean_records': total - int(anomalies),
            'clean_percentage': round(((total - anomalies) / total * 100) if total > 0 else 0, 2)
        }

        # Almacenar para uso posterior
        self.anomalies_found = int(anomalies)
        self.total_records = total

        return stats