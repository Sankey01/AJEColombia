"""
Discount and Penalty Detector Module

Detecta anomalías en descuentos y penalizaciones:
- Descuentos fuera de rango (0-15%)
- Penalizaciones fuera de rango (0-5%)
- Combinaciones ilógicas
"""

import pandas as pd
from typing import Dict
from .base_detector import BaseDetector


class DiscountPenaltyDetector(BaseDetector):
    """
    Detector de anomalías en descuentos y penalizaciones.

    Valida que los porcentajes estén en rangos válidos
    y que las combinaciones sean lógicas.
    """

    def __init__(
        self,
        min_discount: float = 0,
        max_discount: float = 15,
        min_penalty: float = 0,
        max_penalty: float = 5
    ):
        """
        Inicializa el detector de descuentos/penalizaciones.

        Args:
            min_discount: Descuento mínimo válido (default: 0%)
            max_discount: Descuento máximo válido (default: 15%)
            min_penalty: Penalización mínima válida (default: 0%)
            max_penalty: Penalización máxima válida (default: 5%)
        """
        super().__init__(name="discount_penalty")
        self.min_discount = min_discount
        self.max_discount = max_discount
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.anomaly_breakdown: Dict = {}

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta anomalías en descuentos y penalizaciones.

        Args:
            df: DataFrame con los datos de transacciones

        Returns:
            DataFrame con columnas de anomalía agregadas
        """
        # Detectar diferentes tipos de anomalías
        invalid_discount = self._detect_invalid_discount(df)
        invalid_penalty = self._detect_invalid_penalty(df)
        illogical_discount_delayed = self._detect_discount_on_delayed(df)
        illogical_penalty_paid = self._detect_penalty_on_paid(df)

        # Combinar anomalías
        anomaly_mask = (
            invalid_discount |
            invalid_penalty |
            illogical_discount_delayed |
            illogical_penalty_paid
        )

        # Generar descripciones
        descriptions = self._generate_descriptions(
            df,
            invalid_discount,
            invalid_penalty,
            illogical_discount_delayed,
            illogical_penalty_paid
        )

        # Agregar columnas de anomalía
        result = self._add_anomaly_columns(
            df,
            anomaly_mask,
            'DISCOUNT_PENALTY_ANOMALY',
            descriptions
        )

        # Calcular estadísticas
        self._calculate_statistics(result, f'{self.name}_has_anomaly')

        # Guardar desglose
        self._save_breakdown(
            invalid_discount,
            invalid_penalty,
            illogical_discount_delayed,
            illogical_penalty_paid
        )

        return result

    def _detect_invalid_discount(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta descuentos fuera del rango válido.

        Args:
            df: DataFrame con los datos

        Returns:
            Serie booleana con True donde hay descuentos inválidos
        """
        return (
            (df['descuento_pronto_pago'] < self.min_discount) |
            (df['descuento_pronto_pago'] > self.max_discount)
        )

    def _detect_invalid_penalty(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta penalizaciones fuera del rango válido.

        Args:
            df: DataFrame con los datos

        Returns:
            Serie booleana con True donde hay penalizaciones inválidas
        """
        return (
            (df['penalizacion_mora'] < self.min_penalty) |
            (df['penalizacion_mora'] > self.max_penalty)
        )

    def _detect_discount_on_delayed(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta descuentos aplicados a pagos retrasados.

        No tiene sentido dar descuento por pronto pago si está retrasado.

        Args:
            df: DataFrame con los datos

        Returns:
            Serie booleana con True donde hay combinación ilógica
        """
        return (
            (df['estado_pago'] == 'Retrasado') &
            (df['descuento_pronto_pago'] > 0)
        )

    def _detect_penalty_on_paid(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta penalizaciones en pagos marcados como pagados.

        Si está pagado, no debería tener penalización por mora.

        Args:
            df: DataFrame con los datos

        Returns:
            Serie booleana con True donde hay combinación ilógica
        """
        return (
            (df['estado_pago'] == 'Pagado') &
            (df['penalizacion_mora'] > 0)
        )

    def _generate_descriptions(
        self,
        df: pd.DataFrame,
        invalid_discount: pd.Series,
        invalid_penalty: pd.Series,
        illogical_discount_delayed: pd.Series,
        illogical_penalty_paid: pd.Series
    ) -> pd.Series:
        """
        Genera descripciones detalladas para cada anomalía.

        Args:
            df: DataFrame original
            invalid_discount: Máscara de descuentos inválidos
            invalid_penalty: Máscara de penalizaciones inválidas
            illogical_discount_delayed: Máscara de descuentos en retrasados
            illogical_penalty_paid: Máscara de penalizaciones en pagados

        Returns:
            Serie con descripciones
        """
        descriptions = []

        for idx in df.index:
            issues = []

            if invalid_discount[idx]:
                discount = df.loc[idx, 'descuento_pronto_pago']
                issues.append(
                    f"Descuento fuera de rango ({self.min_discount}-{self.max_discount}%): {discount}%"
                )

            if invalid_penalty[idx]:
                penalty = df.loc[idx, 'penalizacion_mora']
                issues.append(
                    f"Penalización fuera de rango ({self.min_penalty}-{self.max_penalty}%): {penalty}%"
                )

            if illogical_discount_delayed[idx]:
                discount = df.loc[idx, 'descuento_pronto_pago']
                issues.append(
                    f"Descuento de {discount}% aplicado a pago retrasado (ilógico)"
                )

            if illogical_penalty_paid[idx]:
                penalty = df.loc[idx, 'penalizacion_mora']
                issues.append(
                    f"Penalización de {penalty}% en pago marcado como pagado (ilógico)"
                )

            descriptions.append('; '.join(issues) if issues else None)

        return pd.Series(descriptions, index=df.index)

    def _save_breakdown(
        self,
        invalid_discount: pd.Series,
        invalid_penalty: pd.Series,
        illogical_discount_delayed: pd.Series,
        illogical_penalty_paid: pd.Series
    ) -> None:
        """
        Guarda el desglose de anomalías por tipo.

        Args:
            invalid_discount: Máscara de descuentos inválidos
            invalid_penalty: Máscara de penalizaciones inválidas
            illogical_discount_delayed: Máscara de descuentos en retrasados
            illogical_penalty_paid: Máscara de penalizaciones en pagados
        """
        self.anomaly_breakdown = {
            'invalid_discounts': int(invalid_discount.sum()),
            'invalid_penalties': int(invalid_penalty.sum()),
            'discount_on_delayed': int(illogical_discount_delayed.sum()),
            'penalty_on_paid': int(illogical_penalty_paid.sum())
        }

    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas sobre anomalías de descuentos/penalizaciones.

        Returns:
            Diccionario con estadísticas
        """
        base_stats = {
            'detector_name': self.name,
            'total_records': self.total_records,
            'anomalies_found': self.anomalies_found,
            'anomaly_percentage': round((self.anomalies_found / self.total_records * 100) if self.total_records > 0 else 0, 2),
            'discount_range': f"{self.min_discount}-{self.max_discount}%",
            'penalty_range': f"{self.min_penalty}-{self.max_penalty}%"
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