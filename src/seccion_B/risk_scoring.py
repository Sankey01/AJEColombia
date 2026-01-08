"""
Risk Scoring Module

Sistema de scoring de riesgo por proveedor con ranking y alertas.
Calcula un score agregado basado en múltiples factores históricos.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging
from datetime import datetime, timedelta

from src.seccion_B.config import CLEANED_DATA_PATH, REPORTS_DIR


class RiskScorer:
    """
    Calculador de risk score por proveedor.

    Genera un score de riesgo (0-1) basado en historial de pagos,
    con sistema de alertas y ranking de seccion_C.
    """

    def __init__(
            self,
            score_weights: Optional[Dict[str, float]] = None,
            risk_thresholds: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        Inicializa el calculador de risk score.

        Args:
            score_weights: Pesos de cada componente del score
            risk_thresholds: Umbrales para clasificación de riesgo
        """
        # Pesos por defecto
        self.score_weights = score_weights or {
            'score_cumplimiento_avg': 0.40,
            'payment_delay_frequency': 0.25,
            'amount_volatility': 0.15,
            'recent_performance': 0.20
        }

        # Validar que pesos suman 1.0
        total_weight = sum(self.score_weights.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Los pesos deben sumar 1.0, suma actual: {total_weight}")

        # Umbrales de riesgo por defecto
        self.risk_thresholds = risk_thresholds or {
            'alto': (0.0, 0.4),  # [0.0, 0.4)
            'medio': (0.4, 0.7),  # [0.4, 0.7)
            'bajo': (0.7, 1.0)  # [0.7, 1.0]
        }

        # Configurar logging
        self.logger = logging.getLogger(__name__)

        # Almacenar resultados
        self.provider_scores: pd.DataFrame = None
        self.alerts: List[Dict] = []

        self.logger.info("RiskScorer inicializado")
        self.logger.info(f"Pesos: {self.score_weights}")
        self.logger.info(f"Umbrales: {self.risk_thresholds}")

    def calculate_provider_scores(
            self,
            df: pd.DataFrame,
            recent_days: int = 90
    ) -> pd.DataFrame:
        """
        Calcula scores de riesgo por proveedor.

        Args:
            df: DataFrame con datos históricos de pagos
            recent_days: Días para análisis de performance reciente

        Returns:
            DataFrame con scores por proveedor
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("CALCULANDO RISK SCORES POR PROVEEDOR")
        self.logger.info("=" * 80)

        # Asegurar que fecha esté en formato datetime
        df = df.copy()
        df['fecha'] = pd.to_datetime(df['fecha'])

        # Calcular componentes del score para cada proveedor
        provider_data = []

        unique_providers = df['proveedor_id'].unique()
        self.logger.info(f"\nProveedores únicos: {len(unique_providers)}")

        for provider_id in unique_providers:
            provider_df = df[df['proveedor_id'] == provider_id]

            # Calcular componentes
            components = self._calculate_score_components(
                provider_df,
                recent_days=recent_days
            )

            # Calcular score final
            final_score = self._calculate_final_score(components)

            # Clasificar riesgo
            risk_category = self._classify_risk(final_score)

            provider_data.append({
                'proveedor_id': provider_id,
                'total_transactions': len(provider_df),
                'score_cumplimiento_avg': components['score_cumplimiento_avg'],
                'payment_delay_frequency': components['payment_delay_frequency'],
                'amount_volatility': components['amount_volatility'],
                'recent_performance': components['recent_performance'],
                'final_risk_score': final_score,
                'risk_category': risk_category,
                'total_amount': provider_df['monto'].sum(),
                'avg_amount': provider_df['monto'].mean(),
                'last_transaction_date': provider_df['fecha'].max(),
                'first_transaction_date': provider_df['fecha'].min()
            })

        # Crear DataFrame
        self.provider_scores = pd.DataFrame(provider_data)

        # Ordenar por risk score (menor = más riesgoso)
        self.provider_scores = self.provider_scores.sort_values(
            'final_risk_score',
            ascending=True
        ).reset_index(drop=True)

        # Agregar ranking
        self.provider_scores['risk_rank'] = range(1, len(self.provider_scores) + 1)

        self.logger.info(f"\nScores calculados para {len(self.provider_scores)} seccion_C")
        self.logger.info(f"Alto Riesgo: {(self.provider_scores['risk_category'] == 'alto').sum()}")
        self.logger.info(f"Medio Riesgo: {(self.provider_scores['risk_category'] == 'medio').sum()}")
        self.logger.info(f"Bajo Riesgo: {(self.provider_scores['risk_category'] == 'bajo').sum()}")

        return self.provider_scores

    def _calculate_score_components(
            self,
            provider_df: pd.DataFrame,
            recent_days: int
    ) -> Dict[str, float]:
        """
        Calcula los componentes individuales del score.

        Args:
            provider_df: DataFrame del proveedor
            recent_days: Días para análisis reciente

        Returns:
            Diccionario con componentes
        """
        # 1. Score de cumplimiento promedio (0-1, mayor = mejor)
        score_cumplimiento_avg = provider_df['score_cumplimiento'].mean()

        # 2. Frecuencia de retrasos (0-1, menor = mejor)
        # Convertir a score donde 1 = sin retrasos, 0 = siempre retrasado
        if 'estado_pago' in provider_df.columns:
            delay_frequency = (provider_df['estado_pago'] == 'Retrasado').mean()
            payment_delay_frequency = 1.0 - delay_frequency
        else:
            # Usar score_cumplimiento como proxy
            payment_delay_frequency = score_cumplimiento_avg

        # 3. Volatilidad de montos (0-1, menor volatilidad = mejor)
        # Normalizar coeficiente de variación
        if len(provider_df) > 1:
            cv = provider_df['monto'].std() / provider_df['monto'].mean()
            # Invertir y normalizar: alta volatilidad (cv>1) = score bajo
            amount_volatility = 1.0 / (1.0 + cv)
        else:
            amount_volatility = 0.5  # Neutral para seccion_C con 1 transacción

        # 4. Performance reciente (últimos N días)
        recent_cutoff = provider_df['fecha'].max() - timedelta(days=recent_days)
        recent_df = provider_df[provider_df['fecha'] >= recent_cutoff]

        if len(recent_df) > 0:
            recent_performance = recent_df['score_cumplimiento'].mean()
        else:
            # Si no hay transacciones recientes, usar promedio histórico
            recent_performance = score_cumplimiento_avg

        return {
            'score_cumplimiento_avg': float(score_cumplimiento_avg),
            'payment_delay_frequency': float(payment_delay_frequency),
            'amount_volatility': float(amount_volatility),
            'recent_performance': float(recent_performance)
        }

    def _calculate_final_score(self, components: Dict[str, float]) -> float:
        """
        Calcula el score final ponderado.

        Args:
            components: Diccionario con componentes individuales

        Returns:
            Score final (0-1)
        """
        final_score = sum(
            components[key] * weight
            for key, weight in self.score_weights.items()
        )

        return float(np.clip(final_score, 0.0, 1.0))

    def _classify_risk(self, score: float) -> str:
        """
        Clasifica el nivel de riesgo según el score.

        Args:
            score: Risk score (0-1)

        Returns:
            Categoría de riesgo: 'alto', 'medio', 'bajo'
        """
        for category, (min_score, max_score) in self.risk_thresholds.items():
            if min_score <= score < max_score:
                return category
            elif category == 'bajo' and score >= min_score:
                return category

        return 'medio'  # Default

    def generate_alerts(
            self,
            alert_thresholds: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Genera alertas basadas en umbrales configurables.

        Args:
            alert_thresholds: Umbrales personalizados para alertas

        Returns:
            Lista de alertas generadas
        """
        if self.provider_scores is None:
            raise ValueError("Debe ejecutar calculate_provider_scores() primero")

        self.logger.info("\n" + "=" * 80)
        self.logger.info("GENERANDO ALERTAS")
        self.logger.info("=" * 80)

        # Umbrales por defecto
        thresholds = alert_thresholds or {
            'critical_score': 0.3,
            'high_amount': 5000,
            'high_transaction_count': 50,
            'recent_activity_days': 30
        }

        self.alerts = []
        current_date = self.provider_scores['last_transaction_date'].max()

        for _, row in self.provider_scores.iterrows():
            provider_alerts = []

            # Alerta 1: Score crítico
            if row['final_risk_score'] < thresholds['critical_score']:
                provider_alerts.append({
                    'type': 'CRITICAL_RISK_SCORE',
                    'severity': 'CRITICA',
                    'message': f"Score de riesgo crítico: {row['final_risk_score']:.3f}",
                    'recommendation': "Revisar historial completo y considerar restricciones"
                })

            # Alerta 2: Alto riesgo con monto elevado
            if (row['risk_category'] == 'alto' and
                    row['total_amount'] > thresholds['high_amount']):
                provider_alerts.append({
                    'type': 'HIGH_RISK_HIGH_AMOUNT',
                    'severity': 'ALTA',
                    'message': f"Alto riesgo con monto acumulado: ${row['total_amount']:,.2f}",
                    'recommendation': "Limitar crédito o requerir garantías adicionales"
                })

            # Alerta 3: Performance reciente degradada
            if (row['recent_performance'] < row['score_cumplimiento_avg'] - 0.15 and
                    row['recent_performance'] < 0.5):
                provider_alerts.append({
                    'type': 'DEGRADED_PERFORMANCE',
                    'severity': 'MEDIA',
                    'message': f"Performance reciente degradada: {row['recent_performance']:.3f} vs histórico {row['score_cumplimiento_avg']:.3f}",
                    'recommendation': "Monitorear de cerca próximas transacciones"
                })

            # Alerta 4: Alta volatilidad
            if row['amount_volatility'] < 0.3:  # Baja puntuación = alta volatilidad
                provider_alerts.append({
                    'type': 'HIGH_VOLATILITY',
                    'severity': 'BAJA',
                    'message': f"Alta volatilidad en montos (score: {row['amount_volatility']:.3f})",
                    'recommendation': "Establecer límites de transacción más estrictos"
                })

            # Alerta 5: Alto volumen de transacciones con riesgo medio/alto
            if (row['total_transactions'] > thresholds['high_transaction_count'] and
                    row['risk_category'] != 'bajo'):
                provider_alerts.append({
                    'type': 'HIGH_VOLUME_RISKY',
                    'severity': 'MEDIA',
                    'message': f"{row['total_transactions']} transacciones con riesgo {row['risk_category']}",
                    'recommendation': "Revisar proceso de aprobación para este proveedor"
                })

            # Alerta 6: Inactividad reciente
            days_since_last = (current_date - row['last_transaction_date']).days
            if days_since_last > thresholds['recent_activity_days']:
                provider_alerts.append({
                    'type': 'INACTIVE_PROVIDER',
                    'severity': 'BAJA',
                    'message': f"Sin actividad en {days_since_last} días",
                    'recommendation': "Actualizar información del proveedor antes de nueva transacción"
                })

            # Agregar alertas si existen
            if provider_alerts:
                self.alerts.append({
                    'proveedor_id': row['proveedor_id'],
                    'risk_score': row['final_risk_score'],
                    'risk_category': row['risk_category'],
                    'alerts': provider_alerts,
                    'alert_count': len(provider_alerts)
                })

        self.logger.info(f"\nAlertas generadas: {len(self.alerts)}")

        # Contar por severidad
        severity_counts = {'CRITICA': 0, 'ALTA': 0, 'MEDIA': 0, 'BAJA': 0}
        for alert in self.alerts:
            for provider_alert in alert['alerts']:
                severity_counts[provider_alert['severity']] += 1

        self.logger.info(f"Por severidad: {severity_counts}")

        return self.alerts

    def get_top_risky_providers(self, n: int = 20) -> pd.DataFrame:
        """
        Obtiene los N seccion_C más riesgosos.

        Args:
            n: Número de seccion_C a retornar

        Returns:
            DataFrame con top N seccion_C
        """
        if self.provider_scores is None:
            raise ValueError("Debe ejecutar calculate_provider_scores() primero")

        return self.provider_scores.head(n)

    def get_providers_by_risk_category(
            self,
            category: str
    ) -> pd.DataFrame:
        """
        Filtra seccion_C por categoría de riesgo.

        Args:
            category: 'alto', 'medio', 'bajo'

        Returns:
            DataFrame filtrado
        """
        if self.provider_scores is None:
            raise ValueError("Debe ejecutar calculate_provider_scores() primero")

        return self.provider_scores[
            self.provider_scores['risk_category'] == category.lower()
            ].copy()

    def get_provider_detail(self, provider_id: int) -> Dict:
        """
        Obtiene información detallada de un proveedor.

        Args:
            provider_id: ID del proveedor

        Returns:
            Diccionario con información detallada
        """
        if self.provider_scores is None:
            raise ValueError("Debe ejecutar calculate_provider_scores() primero")

        provider_row = self.provider_scores[
            self.provider_scores['proveedor_id'] == provider_id
            ]

        if len(provider_row) == 0:
            raise ValueError(f"Proveedor {provider_id} no encontrado")

        provider_info = provider_row.iloc[0].to_dict()

        # Agregar alertas si existen
        provider_alerts = [
            alert for alert in self.alerts
            if alert['proveedor_id'] == provider_id
        ]

        if provider_alerts:
            provider_info['alerts'] = provider_alerts[0]['alerts']
        else:
            provider_info['alerts'] = []

        return provider_info

    def export_reports(
            self,
            output_dir: Optional[str] = None,
            formats: List[str] = ['csv', 'json', 'excel']
    ):
        """
        Exporta reportes de risk scoring.

        Args:
            output_dir: Directorio de salida
            formats: Formatos a exportar ('csv', 'json', 'excel')
        """
        if self.provider_scores is None:
            raise ValueError("Debe ejecutar calculate_provider_scores() primero")

        output_path = Path(output_dir or REPORTS_DIR) / 'risk_scoring'
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPORTANDO REPORTES")
        self.logger.info("=" * 80)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Exportar scores completos
        if 'csv' in formats:
            csv_file = output_path / f'provider_scores_{timestamp}.csv'
            self.provider_scores.to_csv(csv_file, index=False)
            self.logger.info(f"CSV exportado: {csv_file}")

        if 'json' in formats:
            json_file = output_path / f'provider_scores_{timestamp}.json'
            self.provider_scores.to_json(
                json_file,
                orient='records',
                indent=2,
                date_format='iso'
            )
            self.logger.info(f"JSON exportado: {json_file}")

        if 'excel' in formats:
            excel_file = output_path / f'risk_scoring_report_{timestamp}.xlsx'
            self._export_excel_report(excel_file)
            self.logger.info(f"Excel exportado: {excel_file}")

        # 2. Exportar alertas
        if self.alerts:
            alerts_file = output_path / f'alerts_{timestamp}.json'
            with open(alerts_file, 'w', encoding='utf-8') as f:
                json.dump(self.alerts, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Alertas exportadas: {alerts_file}")

        # 3. Exportar resumen por categoría
        summary_file = output_path / f'risk_summary_{timestamp}.json'
        summary = self._generate_summary()
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Resumen exportado: {summary_file}")

        self.logger.info(f"\nTodos los reportes en: {output_path}")

    def _export_excel_report(self, filepath: Path):
        """
        Exporta reporte completo en Excel con múltiples hojas.

        Args:
            filepath: Ruta del archivo Excel
        """
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Hoja 1: Todos los seccion_C
            self.provider_scores.to_excel(
                writer,
                sheet_name='Todos los Proveedores',
                index=False
            )

            # Hoja 2: Top 20 más riesgosos
            self.get_top_risky_providers(20).to_excel(
                writer,
                sheet_name='Top 20 Riesgosos',
                index=False
            )

            # Hoja 3: Alto riesgo
            self.get_providers_by_risk_category('alto').to_excel(
                writer,
                sheet_name='Alto Riesgo',
                index=False
            )

            # Hoja 4: Medio riesgo
            self.get_providers_by_risk_category('medio').to_excel(
                writer,
                sheet_name='Medio Riesgo',
                index=False
            )

            # Hoja 5: Bajo riesgo
            self.get_providers_by_risk_category('bajo').to_excel(
                writer,
                sheet_name='Bajo Riesgo',
                index=False
            )

            # Hoja 6: Resumen
            summary_df = self._create_summary_dataframe()
            summary_df.to_excel(
                writer,
                sheet_name='Resumen',
                index=False
            )

    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Crea DataFrame de resumen para Excel."""
        summary = self._generate_summary()

        data = [
            ['RESUMEN GENERAL', ''],
            ['Total Proveedores', summary['total_providers']],
            ['Alto Riesgo', summary['by_category']['alto']['count']],
            ['Medio Riesgo', summary['by_category']['medio']['count']],
            ['Bajo Riesgo', summary['by_category']['bajo']['count']],
            ['', ''],
            ['ALERTAS', ''],
            ['Total Alertas', summary['total_alerts']],
            ['Críticas', summary['alerts_by_severity']['CRITICA']],
            ['Altas', summary['alerts_by_severity']['ALTA']],
            ['Medias', summary['alerts_by_severity']['MEDIA']],
            ['Bajas', summary['alerts_by_severity']['BAJA']],
        ]

        return pd.DataFrame(data, columns=['Métrica', 'Valor'])

    def _generate_summary(self) -> Dict:
        """Genera resumen estadístico."""
        summary = {
            'total_providers': len(self.provider_scores),
            'by_category': {},
            'total_alerts': len(self.alerts),
            'alerts_by_severity': {'CRITICA': 0, 'ALTA': 0, 'MEDIA': 0, 'BAJA': 0},
            'avg_risk_score': float(self.provider_scores['final_risk_score'].mean()),
            'median_risk_score': float(self.provider_scores['final_risk_score'].median())
        }

        # Por categoría
        for category in ['alto', 'medio', 'bajo']:
            cat_df = self.get_providers_by_risk_category(category)
            summary['by_category'][category] = {
                'count': len(cat_df),
                'percentage': len(cat_df) / len(self.provider_scores) * 100,
                'avg_score': float(cat_df['final_risk_score'].mean()) if len(cat_df) > 0 else 0
            }

        # Contar alertas por severidad
        for alert in self.alerts:
            for provider_alert in alert['alerts']:
                summary['alerts_by_severity'][provider_alert['severity']] += 1

        return summary

    def print_summary(self):
        """Imprime resumen en consola."""
        if self.provider_scores is None:
            print("No hay scores calculados")
            return

        summary = self._generate_summary()

        print("\n" + "=" * 80)
        print("RESUMEN DE RISK SCORING")
        print("=" * 80)

        print(f"\nTotal de seccion_C analizados: {summary['total_providers']}")
        print(f"Score promedio: {summary['avg_risk_score']:.3f}")
        print(f"Score mediana: {summary['median_risk_score']:.3f}")

        print("\nPor Categoría de Riesgo:")
        for category in ['alto', 'medio', 'bajo']:
            cat_data = summary['by_category'][category]
            print(f"  {category.upper():10s}: {cat_data['count']:3d} seccion_C ({cat_data['percentage']:5.1f}%)")

        if self.alerts:
            print(f"\nTotal de alertas: {summary['total_alerts']}")
            print("Por severidad:")
            for severity, count in summary['alerts_by_severity'].items():
                if count > 0:
                    print(f"  {severity:8s}: {count}")

        print("\nTop 5 Proveedores Más Riesgosos:")
        top5 = self.get_top_risky_providers(5)
        for _, row in top5.iterrows():
            print(f"  {row['proveedor_id']:5d} | Score: {row['final_risk_score']:.3f} | "
                  f"Categoría: {row['risk_category']:5s} | "
                  f"Transacciones: {row['total_transactions']:3d}")

        print("=" * 80)


def risk_scoring_pipeline(
        data_path: str = None,
        recent_days: int = 90,
        export_reports: bool = True
) -> RiskScorer:
    """
    Pipeline principal de risk scoring.

    Args:
        data_path: Ruta del archivo de datos limpios
        recent_days: Días para análisis de performance reciente
        export_reports: Si exportar reportes

    Returns:
        RiskScorer con resultados
    """
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE DE RISK SCORING")
    logger.info("=" * 80)

    # Cargar datos
    data_path = data_path or CLEANED_DATA_PATH
    logger.info(f"Cargando datos: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Registros cargados: {len(df)}")

    # Crear scorer
    scorer = RiskScorer()

    # Calcular scores
    scorer.calculate_provider_scores(df, recent_days=recent_days)

    # Generar alertas
    scorer.generate_alerts()

    # Imprimir resumen
    scorer.print_summary()

    # Exportar reportes
    if export_reports:
        scorer.export_reports(formats=['csv', 'json', 'excel'])

    return scorer


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Ejecutar pipeline
    scorer = risk_scoring_pipeline(export_reports=True)