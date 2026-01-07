"""
Approver Anomalies Detector Module

Detecta aprobadores con comportamiento atípico usando análisis estadístico:
- Frecuencia de transacciones anormal
- Montos promedio atípicos
- Patrones sospechosos en nombres
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from .base_detector import BaseDetector


class ApproverAnomaliesDetector(BaseDetector):
    """
    Detector de aprobadores atípicos usando análisis estadístico.

    No usa lista hardcodeada de aprobadores válidos.
    Identifica anomalías basándose en patrones estadísticos.
    """

    def __init__(self, frequency_percentile: float = 10, amount_percentile_low: float = 25, amount_percentile_high: float = 75):
        """
        Inicializa el detector de aprobadores.

        Args:
            frequency_percentile: Percentil para considerar frecuencia baja (default: 10)
            amount_percentile_low: Percentil bajo para montos (default: 25)
            amount_percentile_high: Percentil alto para montos (default: 75)
        """
        super().__init__(name="approver")
        self.frequency_percentile = frequency_percentile
        self.amount_percentile_low = amount_percentile_low
        self.amount_percentile_high = amount_percentile_high
        self.approver_stats: Dict = {}
        self.suspicious_approvers: List = []

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta aprobadores atípicos en el DataFrame.

        Args:
            df: DataFrame con los datos de transacciones

        Returns:
            DataFrame con columnas de anomalía agregadas
        """
        # Analizar estadísticas por aprobador
        self._analyze_approver_statistics(df)

        # Identificar aprobadores atípicos
        atypical_approvers = self._identify_atypical_approvers()

        # Detectar nombres sospechosos
        suspicious_names = self._detect_suspicious_names(df)

        # Combinar ambos criterios
        all_suspicious = list(set(atypical_approvers + suspicious_names))
        self.suspicious_approvers = all_suspicious

        # Crear máscara de anomalías
        anomaly_mask = df['aprobador'].isin(all_suspicious)

        # Generar descripciones
        descriptions = self._generate_descriptions(df, all_suspicious)

        # Agregar columnas de anomalía
        result = self._add_anomaly_columns(
            df,
            anomaly_mask,
            'ATYPICAL_APPROVER',
            descriptions
        )

        # Calcular estadísticas
        self._calculate_statistics(result, f'{self.name}_has_anomaly')

        return result

    def _analyze_approver_statistics(self, df: pd.DataFrame) -> None:
        """
        Analiza estadísticas por cada aprobador.

        Args:
            df: DataFrame con los datos
        """
        approver_groups = df.groupby('aprobador')

        for approver, group in approver_groups:
            self.approver_stats[approver] = {
                'transaction_count': len(group),
                'avg_amount': group['monto'].mean(),
                'std_amount': group['monto'].std(),
                'total_amount': group['monto'].sum(),
                'min_amount': group['monto'].min(),
                'max_amount': group['monto'].max(),
                'median_amount': group['monto'].median()
            }

    def _identify_atypical_approvers(self) -> List[str]:
        """
        Identifica aprobadores con estadísticas atípicas.

        Returns:
            Lista de aprobadores atípicos
        """
        atypical = []

        # Extraer métricas
        transaction_counts = [stats['transaction_count'] for stats in self.approver_stats.values()]
        avg_amounts = [stats['avg_amount'] for stats in self.approver_stats.values()]

        # Calcular percentiles
        freq_threshold = np.percentile(transaction_counts, self.frequency_percentile)
        amount_low_threshold = np.percentile(avg_amounts, self.amount_percentile_low)
        amount_high_threshold = np.percentile(avg_amounts, self.amount_percentile_high)

        # Identificar outliers
        for approver, stats in self.approver_stats.items():
            is_low_frequency = stats['transaction_count'] < freq_threshold
            is_unusual_amount = (
                stats['avg_amount'] < amount_low_threshold or
                stats['avg_amount'] > amount_high_threshold
            )

            # Aprobador atípico si cumple ambos criterios
            if is_low_frequency and is_unusual_amount:
                atypical.append(approver)

        return atypical

    def _detect_suspicious_names(self, df: pd.DataFrame) -> List[str]:
        """
        Detecta nombres de aprobadores con patrones sospechosos.

        Args:
            df: DataFrame con los datos

        Returns:
            Lista de aprobadores con nombres sospechosos
        """
        suspicious = []

        # Patrones sospechosos
        suspicious_patterns = [
            'unauthorized',
            'john doe',
            'jane smith',
            'test',
            'temp',
            'unknown',
            'user'
        ]

        unique_approvers = df['aprobador'].unique()

        for approver in unique_approvers:
            approver_lower = approver.lower()

            # Verificar si contiene algún patrón sospechoso
            for pattern in suspicious_patterns:
                if pattern in approver_lower:
                    suspicious.append(approver)
                    break

        return suspicious

    def _generate_descriptions(
        self,
        df: pd.DataFrame,
        suspicious_approvers: List[str]
    ) -> pd.Series:
        """
        Genera descripciones detalladas para cada anomalía.

        Args:
            df: DataFrame original
            suspicious_approvers: Lista de aprobadores sospechosos

        Returns:
            Serie con descripciones
        """
        descriptions = []

        for idx in df.index:
            approver = df.loc[idx, 'aprobador']

            if approver in suspicious_approvers:
                stats = self.approver_stats.get(approver, {})

                reasons = []

                # Verificar baja frecuencia
                if stats.get('transaction_count', 0) < np.percentile(
                    [s['transaction_count'] for s in self.approver_stats.values()],
                    self.frequency_percentile
                ):
                    reasons.append(
                        f"Baja frecuencia ({stats.get('transaction_count', 0)} transacciones)"
                    )

                # Verificar monto atípico
                avg_amounts = [s['avg_amount'] for s in self.approver_stats.values()]
                if stats.get('avg_amount', 0) < np.percentile(avg_amounts, self.amount_percentile_low):
                    reasons.append(
                        f"Monto promedio bajo (${stats.get('avg_amount', 0):.2f})"
                    )
                elif stats.get('avg_amount', 0) > np.percentile(avg_amounts, self.amount_percentile_high):
                    reasons.append(
                        f"Monto promedio alto (${stats.get('avg_amount', 0):.2f})"
                    )

                # Verificar nombre sospechoso
                if any(pattern in approver.lower() for pattern in ['unauthorized', 'john doe', 'jane smith', 'test', 'temp', 'unknown', 'user']):
                    reasons.append("Nombre sospechoso")

                desc = f"Aprobador atípico: {approver}. " + "; ".join(reasons)
                descriptions.append(desc)
            else:
                descriptions.append(None)

        return pd.Series(descriptions, index=df.index)

    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas sobre aprobadores atípicos.

        Returns:
            Diccionario con estadísticas
        """
        base_stats = {
            'detector_name': self.name,
            'total_records': self.total_records,
            'anomalies_found': self.anomalies_found,
            'anomaly_percentage': round((self.anomalies_found / self.total_records * 100) if self.total_records > 0 else 0, 2),
            'suspicious_approvers_count': len(self.suspicious_approvers),
            'suspicious_approvers': self.suspicious_approvers
        }

        return base_stats

    def get_approver_statistics(self) -> Dict:
        """
        Retorna estadísticas detalladas de todos los aprobadores.

        Returns:
            Diccionario con estadísticas por aprobador
        """
        return self.approver_stats

    def get_approver_report(self) -> pd.DataFrame:
        """
        Genera un reporte detallado de todos los aprobadores.

        Returns:
            DataFrame con reporte de aprobadores
        """
        report_data = []

        for approver, stats in self.approver_stats.items():
            report_data.append({
                'approver': approver,
                'transaction_count': stats['transaction_count'],
                'avg_amount': stats['avg_amount'],
                'std_amount': stats['std_amount'],
                'total_amount': stats['total_amount'],
                'min_amount': stats['min_amount'],
                'max_amount': stats['max_amount'],
                'median_amount': stats['median_amount'],
                'is_suspicious': approver in self.suspicious_approvers
            })

        df_report = pd.DataFrame(report_data)
        df_report = df_report.sort_values('transaction_count', ascending=False)

        return df_report