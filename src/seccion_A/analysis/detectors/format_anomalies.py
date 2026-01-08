"""
Format Anomalies Detector Module

Detecta anomalías en el formato de los datos:
- Montos con decimales excesivos
- Días de crédito negativos o cero
- Montos negativos o cero
"""

import pandas as pd
from typing import Dict, List
from .base_detector import BaseDetector


class FormatAnomaliesDetector(BaseDetector):
    """
    Detector de anomalías de formato en los datos.

    Identifica problemas de formato en columnas numéricas.
    """

    def __init__(self):
        super().__init__(name="format")
        self.anomaly_breakdown: Dict = {}

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta anomalías de formato en el DataFrame.

        Args:
            df: DataFrame con los datos de transacciones

        Returns:
            DataFrame con columnas de anomalía agregadas
        """
        # Detectar diferentes tipos de anomalías de formato
        excessive_decimals = self._detect_excessive_decimals(df)
        negative_credit_days = self._detect_negative_credit_days(df)
        zero_credit_days = self._detect_zero_credit_days(df)
        invalid_amounts = self._detect_invalid_amounts(df)

        # Combinar todas las anomalías
        anomaly_mask = (
            excessive_decimals |
            negative_credit_days |
            zero_credit_days |
            invalid_amounts
        )

        # Generar descripciones
        descriptions = self._generate_descriptions(
            df,
            excessive_decimals,
            negative_credit_days,
            zero_credit_days,
            invalid_amounts
        )

        # Agregar columnas de anomalía
        result = self._add_anomaly_columns(
            df,
            anomaly_mask,
            'FORMAT_ANOMALY',
            descriptions
        )

        # Calcular estadísticas
        self._calculate_statistics(result, f'{self.name}_has_anomaly')

        # Guardar desglose de anomalías
        self._save_breakdown(
            excessive_decimals,
            negative_credit_days,
            zero_credit_days,
            invalid_amounts
        )

        return result

    def _detect_excessive_decimals(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta montos con más de 2 decimales.

        Args:
            df: DataFrame con los datos

        Returns:
            Serie booleana con True donde hay decimales excesivos
        """
        def count_decimals(value: float) -> int:
            """Cuenta el número de decimales de un valor"""
            value_str = str(value)
            if '.' in value_str:
                return len(value_str.split('.')[1])
            return 0

        decimal_counts = df['monto'].apply(count_decimals)
        return decimal_counts > 2

    def _detect_negative_credit_days(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta días de crédito negativos.

        Args:
            df: DataFrame con los datos

        Returns:
            Serie booleana con True donde hay días negativos
        """
        return df['dias_credito'] < 0

    def _detect_zero_credit_days(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta días de crédito igual a cero.

        Args:
            df: DataFrame con los datos

        Returns:
            Serie booleana con True donde hay días en cero
        """
        return df['dias_credito'] == 0

    def _detect_invalid_amounts(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta montos negativos o cero.

        Args:
            df: DataFrame con los datos

        Returns:
            Serie booleana con True donde hay montos inválidos
        """
        return df['monto'] <= 0

    def _generate_descriptions(
        self,
        df: pd.DataFrame,
        excessive_decimals: pd.Series,
        negative_credit_days: pd.Series,
        zero_credit_days: pd.Series,
        invalid_amounts: pd.Series
    ) -> pd.Series:
        """
        Genera descripciones detalladas para cada anomalía.

        Args:
            df: DataFrame original
            excessive_decimals: Máscara de decimales excesivos
            negative_credit_days: Máscara de días negativos
            zero_credit_days: Máscara de días en cero
            invalid_amounts: Máscara de montos inválidos

        Returns:
            Serie con descripciones
        """
        descriptions = []

        for idx in df.index:
            issues = []

            if excessive_decimals[idx]:
                decimal_count = len(str(df.loc[idx, 'monto']).split('.')[1]) if '.' in str(df.loc[idx, 'monto']) else 0
                issues.append(f"Monto con {decimal_count} decimales")

            if negative_credit_days[idx]:
                days = df.loc[idx, 'dias_credito']
                issues.append(f"Días de crédito negativos: {days}")

            if zero_credit_days[idx]:
                issues.append("Días de crédito igual a cero")

            if invalid_amounts[idx]:
                amount = df.loc[idx, 'monto']
                if amount < 0:
                    issues.append(f"Monto negativo: {amount}")
                else:
                    issues.append("Monto igual a cero")

            descriptions.append('; '.join(issues) if issues else None)

        return pd.Series(descriptions, index=df.index)

    def _save_breakdown(
        self,
        excessive_decimals: pd.Series,
        negative_credit_days: pd.Series,
        zero_credit_days: pd.Series,
        invalid_amounts: pd.Series
    ) -> None:
        """
        Guarda el desglose de anomalías por tipo.

        Args:
            excessive_decimals: Máscara de decimales excesivos
            negative_credit_days: Máscara de días negativos
            zero_credit_days: Máscara de días en cero
            invalid_amounts: Máscara de montos inválidos
        """
        self.anomaly_breakdown = {
            'excessive_decimals': int(excessive_decimals.sum()),
            'negative_credit_days': int(negative_credit_days.sum()),
            'zero_credit_days': int(zero_credit_days.sum()),
            'invalid_amounts': int(invalid_amounts.sum())
        }

    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas detalladas sobre anomalías de formato.

        Returns:
            Diccionario con estadísticas
        """
        base_stats = {
            'detector_name': self.name,
            'total_records': self.total_records,
            'anomalies_found': self.anomalies_found,
            'anomaly_percentage': round((self.anomalies_found / self.total_records * 100) if self.total_records > 0 else 0, 2)
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