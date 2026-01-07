"""
Business Rules Detector Module

Detecta violaciones a reglas de negocio:
- Inconsistencia entre condiciones_pago y dias_credito
"""

import pandas as pd
from typing import Dict
from .base_detector import BaseDetector


class BusinessRulesDetector(BaseDetector):
    """
    Detector de violaciones a reglas de negocio.

    Valida que los datos cumplan con las reglas de negocio establecidas.
    """

    def __init__(self):
        super().__init__(name="business_rules")

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta violaciones a reglas de negocio en el DataFrame.

        Args:
            df: DataFrame con los datos de transacciones

        Returns:
            DataFrame con columnas de anomalía agregadas
        """
        # Extraer días esperados de condiciones_pago
        df_work = df.copy()
        df_work['dias_esperados'] = self._extract_expected_days(df_work)

        # Detectar inconsistencias
        anomaly_mask = self._detect_payment_terms_inconsistency(df_work)

        # Generar descripciones
        descriptions = self._generate_descriptions(df_work, anomaly_mask)

        # Agregar columnas de anomalía
        result = self._add_anomaly_columns(
            df_work,
            anomaly_mask,
            'PAYMENT_TERMS_INCONSISTENCY',
            descriptions
        )

        # Calcular estadísticas
        self._calculate_statistics(result, f'{self.name}_has_anomaly')

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

    def _detect_payment_terms_inconsistency(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta cuando condiciones_pago no coincide con dias_credito.

        Según Opción A: Deben coincidir exactamente (sin tolerancia).

        Args:
            df: DataFrame con los datos

        Returns:
            Serie booleana con True donde hay inconsistencia
        """
        return df['dias_esperados'] != df['dias_credito']

    def _generate_descriptions(
        self,
        df: pd.DataFrame,
        anomaly_mask: pd.Series
    ) -> pd.Series:
        """
        Genera descripciones detalladas para cada anomalía.

        Args:
            df: DataFrame original
            anomaly_mask: Máscara de anomalías

        Returns:
            Serie con descripciones
        """
        descriptions = []

        for idx in df.index:
            if anomaly_mask[idx]:
                expected = df.loc[idx, 'dias_esperados']
                actual = df.loc[idx, 'dias_credito']
                condition = df.loc[idx, 'condiciones_pago']

                desc = (
                    f"Inconsistencia: condiciones '{condition}' "
                    f"(esperado: {expected} días) vs dias_credito otorgados: {actual} días"
                )
                descriptions.append(desc)
            else:
                descriptions.append(None)

        return pd.Series(descriptions, index=df.index)

    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas sobre violaciones de reglas de negocio.

        Returns:
            Diccionario con estadísticas
        """
        return {
            'detector_name': self.name,
            'total_records': self.total_records,
            'anomalies_found': self.anomalies_found,
            'anomaly_percentage': round((self.anomalies_found / self.total_records * 100) if self.total_records > 0 else 0, 2)
        }