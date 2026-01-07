"""
Amount Anomalies Detector Module

Detecta montos atípicos usando análisis estadístico:
- Z-score global (>3 desviaciones estándar)
- Z-score por proveedor
- Z-score por aprobador
"""

import pandas as pd
import numpy as np
from typing import Dict
from .base_detector import BaseDetector


class AmountAnomaliesDetector(BaseDetector):
    """
    Detector de montos atípicos usando Z-score.

    Identifica transacciones con montos que se desvían
    significativamente de la media.
    """

    def __init__(self, z_threshold: float = 3.0):
        """
        Inicializa el detector de montos.

        Args:
            z_threshold: Umbral de Z-score para considerar anomalía (default: 3.0)
        """
        super().__init__(name="amount")
        self.z_threshold = z_threshold
        self.anomaly_breakdown: Dict = {}

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta montos atípicos en el DataFrame.

        Args:
            df: DataFrame con los datos de transacciones

        Returns:
            DataFrame con columnas de anomalía agregadas
        """
        df_work = df.copy()

        # Calcular Z-scores
        df_work['z_score_global'] = self._calculate_global_z_score(df_work)
        df_work['z_score_by_provider'] = self._calculate_z_score_by_group(
            df_work, 'proveedor_id'
        )
        df_work['z_score_by_approver'] = self._calculate_z_score_by_group(
            df_work, 'aprobador'
        )

        # Detectar anomalías
        anomaly_global = abs(df_work['z_score_global']) > self.z_threshold
        anomaly_provider = abs(df_work['z_score_by_provider']) > self.z_threshold
        anomaly_approver = abs(df_work['z_score_by_approver']) > self.z_threshold

        # Combinar anomalías (al menos una debe ser atípica)
        anomaly_mask = anomaly_global | anomaly_provider | anomaly_approver

        # Generar descripciones
        descriptions = self._generate_descriptions(
            df_work,
            anomaly_global,
            anomaly_provider,
            anomaly_approver
        )

        # Agregar columnas de anomalía
        result = self._add_anomaly_columns(
            df_work,
            anomaly_mask,
            'ATYPICAL_AMOUNT',
            descriptions
        )

        # Calcular estadísticas
        self._calculate_statistics(result, f'{self.name}_has_anomaly')

        # Guardar desglose
        self._save_breakdown(anomaly_global, anomaly_provider, anomaly_approver)

        return result

    def _calculate_global_z_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula Z-score global para todos los montos.

        Args:
            df: DataFrame con los datos

        Returns:
            Serie con Z-scores
        """
        mean = df['monto'].mean()
        std = df['monto'].std()

        return (df['monto'] - mean) / std

    def _calculate_z_score_by_group(
        self,
        df: pd.DataFrame,
        group_column: str
    ) -> pd.Series:
        """
        Calcula Z-score por grupo (proveedor o aprobador).

        Args:
            df: DataFrame con los datos
            group_column: Columna para agrupar

        Returns:
            Serie con Z-scores por grupo
        """
        z_scores = []

        for idx, row in df.iterrows():
            group_value = row[group_column]
            group_data = df[df[group_column] == group_value]['monto']

            if len(group_data) > 1:
                mean = group_data.mean()
                std = group_data.std()

                if std > 0:
                    z_score = (row['monto'] - mean) / std
                else:
                    z_score = 0
            else:
                z_score = 0

            z_scores.append(z_score)

        return pd.Series(z_scores, index=df.index)

    def _generate_descriptions(
        self,
        df: pd.DataFrame,
        anomaly_global: pd.Series,
        anomaly_provider: pd.Series,
        anomaly_approver: pd.Series
    ) -> pd.Series:
        """
        Genera descripciones detalladas para cada anomalía.

        Args:
            df: DataFrame original
            anomaly_global: Máscara de anomalías globales
            anomaly_provider: Máscara de anomalías por proveedor
            anomaly_approver: Máscara de anomalías por aprobador

        Returns:
            Serie con descripciones
        """
        descriptions = []

        for idx in df.index:
            issues = []

            if anomaly_global[idx]:
                z_score = df.loc[idx, 'z_score_global']
                monto = df.loc[idx, 'monto']
                issues.append(
                    f"Monto atípico global: ${monto:.2f} (Z-score: {z_score:.2f})"
                )

            if anomaly_provider[idx]:
                z_score = df.loc[idx, 'z_score_by_provider']
                provider = df.loc[idx, 'proveedor_id']
                issues.append(
                    f"Atípico para proveedor {provider} (Z-score: {z_score:.2f})"
                )

            if anomaly_approver[idx]:
                z_score = df.loc[idx, 'z_score_by_approver']
                approver = df.loc[idx, 'aprobador']
                issues.append(
                    f"Atípico para aprobador {approver} (Z-score: {z_score:.2f})"
                )

            descriptions.append('; '.join(issues) if issues else None)

        return pd.Series(descriptions, index=df.index)

    def _save_breakdown(
        self,
        anomaly_global: pd.Series,
        anomaly_provider: pd.Series,
        anomaly_approver: pd.Series
    ) -> None:
        """
        Guarda el desglose de anomalías por tipo.

        Args:
            anomaly_global: Máscara de anomalías globales
            anomaly_provider: Máscara de anomalías por proveedor
            anomaly_approver: Máscara de anomalías por aprobador
        """
        self.anomaly_breakdown = {
            'global_outliers': int(anomaly_global.sum()),
            'provider_outliers': int(anomaly_provider.sum()),
            'approver_outliers': int(anomaly_approver.sum())
        }

    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas sobre montos atípicos.

        Returns:
            Diccionario con estadísticas
        """
        base_stats = {
            'detector_name': self.name,
            'total_records': self.total_records,
            'anomalies_found': self.anomalies_found,
            'anomaly_percentage': round((self.anomalies_found / self.total_records * 100) if self.total_records > 0 else 0, 2),
            'z_threshold': self.z_threshold
        }

        # Agregar desglose
        base_stats['breakdown'] = self.anomaly_breakdown

        return base_stats

    def get_anomaly_breakdown(self) -> Dict:
        """
        Retorna el desglose de anomalías por tipo.

        Returns:
            Diccionario con conteo por tipo de anomalía
        """
        return self.anomaly_breakdown