"""
Anomaly Detector Orchestrator Module

Orquestador principal que coordina todos los detectores de anomalías
y consolida los resultados.
"""

import pandas as pd
import json
from typing import Dict, List
from pathlib import Path

from .detectors.duplicate_detector import DuplicateDetector
from .detectors.format_anomalies import FormatAnomaliesDetector
from .detectors.temporal_anomalies import TemporalAnomaliesDetector
from .detectors.business_rules import BusinessRulesDetector
from .detectors.approver_anomalies import ApproverAnomaliesDetector
from .detectors.amount_anomalies import AmountAnomaliesDetector
from .detectors.discount_penalty import DiscountPenaltyDetector
from .detectors.cross_consistency import CrossConsistencyDetector


class AnomalyDetectorOrchestrator:
    """
    Orquestador que coordina todos los detectores de anomalías.

    Ejecuta todos los detectores, consolida resultados y genera reportes.
    """

    def __init__(self):
        """Inicializa el orquestador con todos los detectores"""
        self.detectors = {
            'duplicate': DuplicateDetector(),
            'format': FormatAnomaliesDetector(),
            'temporal': TemporalAnomaliesDetector(tolerance_days=5),
            'business_rules': BusinessRulesDetector(),
            'approver': ApproverAnomaliesDetector(),
            'amount': AmountAnomaliesDetector(z_threshold=3.0),
            'discount_penalty': DiscountPenaltyDetector(),
            'cross_consistency': CrossConsistencyDetector()
        }
        self.results: Dict = {}
        self.consolidated_df: pd.DataFrame = None
        self.global_statistics: Dict = {}

    def detect_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta todos los detectores sobre el DataFrame.

        Args:
            df: DataFrame con los datos de transacciones

        Returns:
            DataFrame consolidado con todas las anomalías detectadas
        """
        print("\n" + "=" * 80)
        print("EJECUTANDO DETECCIÓN DE ANOMALÍAS")
        print("=" * 80)

        # Ejecutar cada detector
        for name, detector in self.detectors.items():
            print(f"\n→ Ejecutando detector: {name}...")
            result = detector.detect(df)
            self.results[name] = {
                'dataframe': result,
                'statistics': detector.get_statistics()
            }
            print(f"  ✓ Anomalías encontradas: {detector.anomalies_found}")

        # Consolidar resultados
        self.consolidated_df = self._consolidate_results(df)

        # Calcular estadísticas globales
        self._calculate_global_statistics()

        print("\n" + "=" * 80)
        print("✅ DETECCIÓN COMPLETADA")
        print("=" * 80)

        return self.consolidated_df

    def _consolidate_results(self, df_original: pd.DataFrame) -> pd.DataFrame:
        """
        Consolida los resultados de todos los detectores en un solo DataFrame.

        Args:
            df_original: DataFrame original

        Returns:
            DataFrame consolidado
        """
        df_consolidated = df_original.copy()

        # Agregar columnas de cada detector
        for name, result_data in self.results.items():
            result_df = result_data['dataframe']

            # Copiar columnas de anomalía de cada detector
            anomaly_columns = [col for col in result_df.columns if name in col]
            for col in anomaly_columns:
                df_consolidated[col] = result_df[col]

        # Agregar columna de conteo total de anomalías
        has_anomaly_cols = [col for col in df_consolidated.columns if '_has_anomaly' in col]
        df_consolidated['total_anomalies'] = df_consolidated[has_anomaly_cols].sum(axis=1)

        # Agregar columna indicando si tiene alguna anomalía
        df_consolidated['has_any_anomaly'] = df_consolidated['total_anomalies'] > 0

        return df_consolidated

    def _calculate_global_statistics(self) -> None:
        """Calcula estadísticas globales de todas las anomalías"""
        total_records = len(self.consolidated_df)
        records_with_anomalies = (self.consolidated_df['has_any_anomaly'] == True).sum()

        # Estadísticas por detector
        detector_stats = {}
        for name, result_data in self.results.items():
            detector_stats[name] = result_data['statistics']

        # Estadísticas de registros con múltiples anomalías
        anomaly_distribution = self.consolidated_df['total_anomalies'].value_counts().sort_index().to_dict()

        self.global_statistics = {
            'total_records': int(total_records),
            'records_with_anomalies': int(records_with_anomalies),
            'records_clean': int(total_records - records_with_anomalies),
            'anomaly_percentage': round((records_with_anomalies / total_records * 100) if total_records > 0 else 0, 2),
            'clean_percentage': round(
                ((total_records - records_with_anomalies) / total_records * 100) if total_records > 0 else 0, 2),
            'detector_statistics': detector_stats,
            'anomaly_distribution': {str(k): int(v) for k, v in anomaly_distribution.items()}
        }

    def get_summary(self) -> Dict:
        """
        Retorna un resumen completo de las anomalías detectadas.

        Returns:
            Diccionario con resumen global
        """
        return self.global_statistics

    def get_top_anomalies(self, n: int = 10) -> pd.DataFrame:
        """
        Retorna los registros con más anomalías.

        Args:
            n: Número de registros a retornar

        Returns:
            DataFrame con los top N registros con más anomalías
        """
        return self.consolidated_df.nlargest(n, 'total_anomalies')

    def export_results(
            self,
            output_dir: str = './output',
            export_csv: bool = True,
            export_json: bool = True,
            export_summary: bool = True
    ) -> None:
        """
        Exporta los resultados a archivos.

        Args:
            output_dir: Directorio de salida
            export_csv: Si exportar DataFrame completo a CSV
            export_json: Si exportar estadísticas a JSON
            export_summary: Si exportar resumen en texto
        """
        # Crear directorio si no existe
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n→ Exportando resultados a {output_dir}...")

        # Exportar DataFrame completo
        if export_csv:
            csv_path = output_path / 'anomalies_full.csv'
            self.consolidated_df.to_csv(csv_path, index=False)
            print(f"  ✓ CSV completo: {csv_path}")

        # Exportar solo registros con anomalías
        if export_csv:
            anomalies_only = self.consolidated_df[self.consolidated_df['has_any_anomaly'] == True]
            csv_anomalies_path = output_path / 'anomalies_only.csv'
            anomalies_only.to_csv(csv_anomalies_path, index=False)
            print(f"  ✓ CSV anomalías: {csv_anomalies_path}")

        # Exportar estadísticas en JSON
        if export_json:
            json_path = output_path / 'statistics.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.global_statistics, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Estadísticas JSON: {json_path}")

        # Exportar resumen en texto
        if export_summary:
            summary_path = output_path / 'summary.txt'
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(self._generate_text_summary())
            print(f"  ✓ Resumen: {summary_path}")

        print("  ✅ Exportación completada")

    def _generate_text_summary(self) -> str:
        """
        Genera un resumen en texto plano.

        Returns:
            String con resumen formateado
        """
        stats = self.global_statistics

        summary = []
        summary.append("=" * 80)
        summary.append("RESUMEN DE DETECCIÓN DE ANOMALÍAS")
        summary.append("=" * 80)
        summary.append("")
        summary.append(f"Total de registros analizados: {stats['total_records']:,}")
        summary.append(f"Registros con anomalías: {stats['records_with_anomalies']:,} ({stats['anomaly_percentage']}%)")
        summary.append(f"Registros limpios: {stats['records_clean']:,} ({stats['clean_percentage']}%)")
        summary.append("")
        summary.append("=" * 80)
        summary.append("ANOMALÍAS POR DETECTOR")
        summary.append("=" * 80)

        for detector_name, detector_stats in stats['detector_statistics'].items():
            summary.append("")
            summary.append(f"{detector_name.upper()}:")
            summary.append(
                f"  - Anomalías: {detector_stats['anomalies_found']:,} ({detector_stats['anomaly_percentage']}%)")

            # Agregar breakdown si existe
            if 'breakdown' in detector_stats:
                summary.append("  - Desglose:")
                for key, value in detector_stats['breakdown'].items():
                    summary.append(f"    * {key}: {value:,}")

        summary.append("")
        summary.append("=" * 80)
        summary.append("DISTRIBUCIÓN DE ANOMALÍAS POR REGISTRO")
        summary.append("=" * 80)
        summary.append("")

        for num_anomalies, count in sorted(stats['anomaly_distribution'].items()):
            summary.append(f"Registros con {num_anomalies} anomalías: {count:,}")

        return "\n".join(summary)

    def print_summary(self) -> None:
        """Imprime el resumen en consola"""
        print("\n" + self._generate_text_summary())