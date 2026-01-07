"""
Duplicate Detector Module

Detecta transacciones duplicadas basándose en ID_transaccion.
Identifica qué campos difieren entre registros duplicados.
"""

import pandas as pd
from typing import Dict, List
from .base_detector import BaseDetector


class DuplicateDetector(BaseDetector):
    """
    Detector de transacciones duplicadas.

    Identifica registros con el mismo ID_transaccion y analiza
    las diferencias entre ellos.
    """

    def __init__(self):
        super().__init__(name="duplicate")
        self.duplicate_groups: Dict = {}
        self.duplicate_summary: Dict = {}

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta transacciones duplicadas en el DataFrame.

        Args:
            df: DataFrame con los datos de transacciones

        Returns:
            DataFrame con columnas de anomalía agregadas
        """
        # Identificar IDs duplicados
        duplicate_ids = df[df.duplicated(subset=['ID_transaccion'], keep=False)]['ID_transaccion'].unique()

        # Crear máscara de anomalías
        anomaly_mask = df['ID_transaccion'].isin(duplicate_ids)

        # Generar descripciones detalladas
        descriptions = self._generate_descriptions(df, duplicate_ids)

        # Agregar columnas de anomalía
        result = self._add_anomaly_columns(
            df,
            anomaly_mask,
            'DUPLICATE_TRANSACTION',
            descriptions
        )

        # Calcular estadísticas
        self._calculate_statistics(result, f'{self.name}_has_anomaly')

        # Analizar grupos de duplicados
        self._analyze_duplicate_groups(df, duplicate_ids)

        return result

    def _generate_descriptions(self, df: pd.DataFrame, duplicate_ids: List) -> pd.Series:
        """
        Genera descripciones detalladas para cada duplicado.

        Args:
            df: DataFrame original
            duplicate_ids: Lista de IDs duplicados

        Returns:
            Serie con descripciones
        """
        descriptions = []

        for _, row in df.iterrows():
            transaction_id = row['ID_transaccion']

            if transaction_id in duplicate_ids:
                # Obtener todos los registros con este ID
                duplicates = df[df['ID_transaccion'] == transaction_id]
                count = len(duplicates)

                # Identificar diferencias
                differences = self._identify_differences(duplicates)

                desc = f"ID duplicado {count} veces. Diferencias en: {', '.join(differences)}"
                descriptions.append(desc)
            else:
                descriptions.append(None)

        return pd.Series(descriptions, index=df.index)

    def _identify_differences(self, duplicates: pd.DataFrame) -> List[str]:
        """
        Identifica qué campos difieren entre registros duplicados.

        Args:
            duplicates: DataFrame con registros duplicados del mismo ID

        Returns:
            Lista de nombres de campos que difieren
        """
        differences = []

        # Columnas a verificar (excluir ID_transaccion)
        columns_to_check = [col for col in duplicates.columns if col != 'ID_transaccion']

        for col in columns_to_check:
            unique_values = duplicates[col].nunique()
            if unique_values > 1:
                differences.append(col)

        return differences if differences else ['ninguna']

    def _analyze_duplicate_groups(self, df: pd.DataFrame, duplicate_ids: List) -> None:
        """
        Analiza en detalle cada grupo de duplicados.

        Args:
            df: DataFrame original
            duplicate_ids: Lista de IDs duplicados
        """
        self.duplicate_groups = {}

        for dup_id in duplicate_ids:
            group = df[df['ID_transaccion'] == dup_id]

            self.duplicate_groups[dup_id] = {
                'count': len(group),
                'differences': self._identify_differences(group),
                'records': group.to_dict('records'),
                'approvers': group['aprobador'].unique().tolist(),
                'amounts': group['monto'].unique().tolist(),
                'has_different_approvers': group['aprobador'].nunique() > 1,
                'has_different_amounts': group['monto'].nunique() > 1
            }

        # Generar resumen
        self._generate_summary()

    def _generate_summary(self) -> None:
        """Genera un resumen de los duplicados encontrados"""
        total_groups = len(self.duplicate_groups)

        different_approvers = sum(
            1 for g in self.duplicate_groups.values()
            if g['has_different_approvers']
        )

        different_amounts = sum(
            1 for g in self.duplicate_groups.values()
            if g['has_different_amounts']
        )

        both_different = sum(
            1 for g in self.duplicate_groups.values()
            if g['has_different_approvers'] and g['has_different_amounts']
        )

        self.duplicate_summary = {
            'total_duplicate_groups': total_groups,
            'groups_with_different_approvers': different_approvers,
            'groups_with_different_amounts': different_amounts,
            'groups_with_both_different': both_different,
            'total_duplicate_records': self.anomalies_found
        }

    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas detalladas sobre duplicados.

        Returns:
            Diccionario con estadísticas
        """
        base_stats = {
            'detector_name': self.name,
            'total_records': self.total_records,
            'anomalies_found': self.anomalies_found,
            'anomaly_percentage': round((self.anomalies_found / self.total_records * 100) if self.total_records > 0 else 0, 2)
        }

        # Agregar resumen de duplicados
        base_stats.update(self.duplicate_summary)

        return base_stats

    def get_duplicate_groups(self) -> Dict:
        """
        Retorna el diccionario con todos los grupos de duplicados.

        Returns:
            Diccionario con información detallada de cada grupo
        """
        return self.duplicate_groups

    def get_critical_duplicates(self) -> List[int]:
        """
        Retorna lista de IDs con duplicados críticos.

        Críticos = diferentes aprobadores O diferentes montos

        Returns:
            Lista de IDs con duplicados críticos
        """
        critical = []

        for dup_id, info in self.duplicate_groups.items():
            if info['has_different_approvers'] or info['has_different_amounts']:
                critical.append(dup_id)

        return critical

    def export_duplicate_report(self) -> pd.DataFrame:
        """
        Exporta un reporte detallado de duplicados.

        Returns:
            DataFrame con reporte de duplicados
        """
        report_data = []

        for dup_id, info in self.duplicate_groups.items():
            report_data.append({
                'ID_transaccion': dup_id,
                'duplicate_count': info['count'],
                'differences': ', '.join(info['differences']),
                'different_approvers': info['has_different_approvers'],
                'different_amounts': info['has_different_amounts'],
                'approvers': ', '.join(map(str, info['approvers'])),
                'amounts': ', '.join(map(str, info['amounts']))
            })

        return pd.DataFrame(report_data)