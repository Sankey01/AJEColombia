"""
Temporal Anomalies Detector Module

Detecta anomalías temporales relacionadas con fechas:
- Inconsistencia entre fechas y condiciones de pago
- Fechas de vencimiento anteriores a fecha de factura
- Diferencias excesivas entre fechas
"""

import pandas as pd
from typing import Dict
from .base_detector import BaseDetector


class TemporalAnomaliesDetector(BaseDetector):
    """
    Detector de anomalías temporales en fechas.

    Valida la consistencia entre fechas de factura, vencimiento
    y las condiciones de pago establecidas.
    """

    def __init__(self, tolerance_days: int = 5):
        """
        Inicializa el detector temporal.

        Args:
            tolerance_days: Días de tolerancia para diferencias (default: 5)
        """
        super().__init__(name="temporal")
        self.tolerance_days = tolerance_days
        self.anomaly_breakdown: Dict = {}

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta anomalías temporales en el DataFrame.

        Args:
            df: DataFrame con los datos de transacciones

        Returns:
            DataFrame con columnas de anomalía agregadas
        """
        # Convertir fechas si no están en formato datetime
        df_work = df.copy()
        df_work['fecha_factura'] = pd.to_datetime(df_work['fecha_factura'])
        df_work['fecha_vencimiento'] = pd.to_datetime(df_work['fecha_vencimiento'])

        # Calcular diferencia real de días
        df_work['dias_diferencia_real'] = (
            df_work['fecha_vencimiento'] - df_work['fecha_factura']
        ).dt.days

        # Extraer días esperados de condiciones_pago
        df_work['dias_esperados'] = self._extract_expected_days(df_work)

        # Detectar diferentes tipos de anomalías temporales
        inconsistent_dates = self._detect_inconsistent_dates(df_work)
        inverted_dates = self._detect_inverted_dates(df_work)
        excessive_difference = self._detect_excessive_difference(df_work)

        # Combinar todas las anomalías
        anomaly_mask = (
            inconsistent_dates |
            inverted_dates |
            excessive_difference
        )

        # Generar descripciones
        descriptions = self._generate_descriptions(
            df_work,
            inconsistent_dates,
            inverted_dates,
            excessive_difference
        )

        # Agregar columnas de anomalía
        result = self._add_anomaly_columns(
            df_work,
            anomaly_mask,
            'TEMPORAL_ANOMALY',
            descriptions
        )

        # Calcular estadísticas
        self._calculate_statistics(result, f'{self.name}_has_anomaly')

        # Guardar desglose
        self._save_breakdown(
            inconsistent_dates,
            inverted_dates,
            excessive_difference
        )

        return result

    def _extract_expected_days(self, df: pd.DataFrame) -> pd.Series:
        """
        Extrae el número de días de la columna condiciones_pago.

        Args:
            df: DataFrame con los datos

        Returns:
            Serie con días esperados
        """
        # Extraer número de "30 días", "60 días", "90 días"
        return df['condiciones_pago'].str.extract(r'(\d+)')[0].astype(int)

    def _detect_inconsistent_dates(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta cuando la diferencia real no coincide con la esperada.

        Args:
            df: DataFrame con los datos

        Returns:
            Serie booleana con True donde hay inconsistencia
        """
        # Diferencia entre lo esperado y lo real
        difference = abs(df['dias_diferencia_real'] - df['dias_esperados'])

        # Anomalía si la diferencia supera la tolerancia
        return difference > self.tolerance_days

    def _detect_inverted_dates(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta fechas de vencimiento anteriores a fecha de factura.

        Args:
            df: DataFrame con los datos

        Returns:
            Serie booleana con True donde hay fechas invertidas
        """
        return df['fecha_vencimiento'] < df['fecha_factura']

    def _detect_excessive_difference(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta diferencias extremas (más de 2 años).

        Args:
            df: DataFrame con los datos

        Returns:
            Serie booleana con True donde hay diferencias extremas
        """
        # 2 años = 730 días (aproximado)
        return df['dias_diferencia_real'] > 730

    def _generate_descriptions(
        self,
        df: pd.DataFrame,
        inconsistent_dates: pd.Series,
        inverted_dates: pd.Series,
        excessive_difference: pd.Series
    ) -> pd.Series:
        """
        Genera descripciones detalladas para cada anomalía.

        Args:
            df: DataFrame original
            inconsistent_dates: Máscara de fechas inconsistentes
            inverted_dates: Máscara de fechas invertidas
            excessive_difference: Máscara de diferencias extremas

        Returns:
            Serie con descripciones
        """
        descriptions = []

        for idx in df.index:
            issues = []

            if inverted_dates[idx]:
                issues.append(
                    f"Fecha vencimiento ({df.loc[idx, 'fecha_vencimiento'].date()}) "
                    f"anterior a fecha factura ({df.loc[idx, 'fecha_factura'].date()})"
                )

            if inconsistent_dates[idx] and not inverted_dates[idx]:
                real = df.loc[idx, 'dias_diferencia_real']
                expected = df.loc[idx, 'dias_esperados']
                diff = abs(real - expected)
                issues.append(
                    f"Diferencia real ({real} días) vs esperada ({expected} días): "
                    f"{diff} días de diferencia"
                )

            if excessive_difference[idx] and not inverted_dates[idx]:
                days = df.loc[idx, 'dias_diferencia_real']
                issues.append(f"Diferencia excesiva: {days} días (>2 años)")

            descriptions.append('; '.join(issues) if issues else None)

        return pd.Series(descriptions, index=df.index)

    def _save_breakdown(
        self,
        inconsistent_dates: pd.Series,
        inverted_dates: pd.Series,
        excessive_difference: pd.Series
    ) -> None:
        """
        Guarda el desglose de anomalías por tipo.

        Args:
            inconsistent_dates: Máscara de fechas inconsistentes
            inverted_dates: Máscara de fechas invertidas
            excessive_difference: Máscara de diferencias extremas
        """
        self.anomaly_breakdown = {
            'inconsistent_dates': int(inconsistent_dates.sum()),
            'inverted_dates': int(inverted_dates.sum()),
            'excessive_difference': int(excessive_difference.sum())
        }

    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas detalladas sobre anomalías temporales.

        Returns:
            Diccionario con estadísticas
        """
        base_stats = {
            'detector_name': self.name,
            'total_records': self.total_records,
            'anomalies_found': self.anomalies_found,
            'anomaly_percentage': round((self.anomalies_found / self.total_records * 100) if self.total_records > 0 else 0, 2),
            'tolerance_days': self.tolerance_days
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