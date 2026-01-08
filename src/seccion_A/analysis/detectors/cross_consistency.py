"""
Cross Consistency Detector Module

Detecta inconsistencias entre campos relacionados:
- Estado de pago vs fechas
- Estado de pago vs descuentos/penalizaciones
- Otras validaciones cruzadas
"""

import pandas as pd
from datetime import datetime
from typing import Dict
from .base_detector import BaseDetector


class CrossConsistencyDetector(BaseDetector):
    """
    Detector de consistencia cruzada entre campos.

    Valida que los datos sean consistentes entre sí
    considerando múltiples campos relacionados.
    """

    def __init__(self, reference_date: str = None):
        """
        Inicializa el detector de consistencia cruzada.

        Args:
            reference_date: Fecha de referencia para validaciones (default: hoy)
        """
        super().__init__(name="cross_consistency")
        self.reference_date = pd.to_datetime(reference_date) if reference_date else pd.to_datetime('today')
        self.anomaly_breakdown: Dict = {}

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta inconsistencias cruzadas en el DataFrame.

        Args:
            df: DataFrame con los datos de transacciones

        Returns:
            DataFrame con columnas de anomalía agregadas
        """
        # Convertir fechas si es necesario
        df_work = df.copy()
        df_work['fecha_factura'] = pd.to_datetime(df_work['fecha_factura'])
        df_work['fecha_vencimiento'] = pd.to_datetime(df_work['fecha_vencimiento'])

        # Detectar inconsistencias
        delayed_not_past_due = self._detect_delayed_not_past_due(df_work)
        pending_past_due = self._detect_pending_past_due(df_work)
        paid_with_penalty = self._detect_paid_with_penalty(df_work)
        delayed_with_discount = self._detect_delayed_with_discount(df_work)

        # Combinar anomalías
        anomaly_mask = (
            delayed_not_past_due |
            pending_past_due |
            paid_with_penalty |
            delayed_with_discount
        )

        # Generar descripciones
        descriptions = self._generate_descriptions(
            df_work,
            delayed_not_past_due,
            pending_past_due,
            paid_with_penalty,
            delayed_with_discount
        )

        # Agregar columnas de anomalía
        result = self._add_anomaly_columns(
            df_work,
            anomaly_mask,
            'CROSS_CONSISTENCY_ANOMALY',
            descriptions
        )

        # Calcular estadísticas
        self._calculate_statistics(result, f'{self.name}_has_anomaly')

        # Guardar desglose
        self._save_breakdown(
            delayed_not_past_due,
            pending_past_due,
            paid_with_penalty,
            delayed_with_discount
        )

        return result

    def _detect_delayed_not_past_due(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta pagos marcados como retrasados pero no vencidos.

        Args:
            df: DataFrame con los datos

        Returns:
            Serie booleana con True donde hay inconsistencia
        """
        is_delayed = df['estado_pago'] == 'Retrasado'
        not_past_due = df['fecha_vencimiento'] > self.reference_date

        return is_delayed & not_past_due

    def _detect_pending_past_due(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta pagos pendientes pero ya vencidos.

        Args:
            df: DataFrame con los datos

        Returns:
            Serie booleana con True donde hay inconsistencia
        """
        is_pending = df['estado_pago'] == 'Pendiente'
        is_past_due = df['fecha_vencimiento'] < self.reference_date

        return is_pending & is_past_due

    def _detect_paid_with_penalty(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta pagos marcados como pagados pero con penalización.

        Esto puede ser válido si pagó tarde pero ya pagó,
        pero vale la pena revisarlo.

        Args:
            df: DataFrame con los datos

        Returns:
            Serie booleana con True donde hay posible inconsistencia
        """
        is_paid = df['estado_pago'] == 'Pagado'
        has_penalty = df['penalizacion_mora'] > 0

        return is_paid & has_penalty

    def _detect_delayed_with_discount(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta pagos retrasados con descuento por pronto pago.

        Args:
            df: DataFrame con los datos

        Returns:
            Serie booleana con True donde hay inconsistencia
        """
        is_delayed = df['estado_pago'] == 'Retrasado'
        has_discount = df['descuento_pronto_pago'] > 0

        return is_delayed & has_discount

    def _generate_descriptions(
        self,
        df: pd.DataFrame,
        delayed_not_past_due: pd.Series,
        pending_past_due: pd.Series,
        paid_with_penalty: pd.Series,
        delayed_with_discount: pd.Series
    ) -> pd.Series:
        """
        Genera descripciones detalladas para cada anomalía.

        Args:
            df: DataFrame original
            delayed_not_past_due: Máscara de retrasados no vencidos
            pending_past_due: Máscara de pendientes vencidos
            paid_with_penalty: Máscara de pagados con penalización
            delayed_with_discount: Máscara de retrasados con descuento

        Returns:
            Serie con descripciones
        """
        descriptions = []

        for idx in df.index:
            issues = []

            if delayed_not_past_due[idx]:
                venc = df.loc[idx, 'fecha_vencimiento'].date()
                issues.append(
                    f"Marcado como 'Retrasado' pero vence en fecha futura ({venc})"
                )

            if pending_past_due[idx]:
                venc = df.loc[idx, 'fecha_vencimiento'].date()
                issues.append(
                    f"Marcado como 'Pendiente' pero ya venció ({venc})"
                )

            if paid_with_penalty[idx]:
                penalty = df.loc[idx, 'penalizacion_mora']
                issues.append(
                    f"Marcado como 'Pagado' pero tiene penalización por mora ({penalty}%)"
                )

            if delayed_with_discount[idx]:
                discount = df.loc[idx, 'descuento_pronto_pago']
                issues.append(
                    f"Marcado como 'Retrasado' pero tiene descuento pronto pago ({discount}%)"
                )

            descriptions.append('; '.join(issues) if issues else None)

        return pd.Series(descriptions, index=df.index)

    def _save_breakdown(
        self,
        delayed_not_past_due: pd.Series,
        pending_past_due: pd.Series,
        paid_with_penalty: pd.Series,
        delayed_with_discount: pd.Series
    ) -> None:
        """
        Guarda el desglose de anomalías por tipo.

        Args:
            delayed_not_past_due: Máscara de retrasados no vencidos
            pending_past_due: Máscara de pendientes vencidos
            paid_with_penalty: Máscara de pagados con penalización
            delayed_with_discount: Máscara de retrasados con descuento
        """
        self.anomaly_breakdown = {
            'delayed_not_past_due': int(delayed_not_past_due.sum()),
            'pending_past_due': int(pending_past_due.sum()),
            'paid_with_penalty': int(paid_with_penalty.sum()),
            'delayed_with_discount': int(delayed_with_discount.sum())
        }

    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas sobre inconsistencias cruzadas.

        Returns:
            Diccionario con estadísticas
        """
        base_stats = {
            'detector_name': self.name,
            'total_records': self.total_records,
            'anomalies_found': self.anomalies_found,
            'anomaly_percentage': round((self.anomalies_found / self.total_records * 100) if self.total_records > 0 else 0, 2),
            'reference_date': str(self.reference_date.date())
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