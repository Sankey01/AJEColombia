"""
Risk Scoring Visualization Module

Genera visualizaciones profesionales para risk scoring.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List
import logging


class RiskScoringVisualizer:
    """
    Generador de visualizaciones para risk scoring.

    Crea gráficos profesionales de distribución, ranking y alertas.
    """

    def __init__(self):
        """Inicializa el visualizador."""
        self.logger = logging.getLogger(__name__)
        self._setup_plot_style()

    def _setup_plot_style(self):
        """Configura el estilo de los gráficos."""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10

    def create_all_visualizations(
        self,
        provider_scores: pd.DataFrame,
        alerts: List,
        output_dir: str
    ):
        """
        Crea todas las visualizaciones.

        Args:
            provider_scores: DataFrame con scores
            alerts: Lista de alertas
            output_dir: Directorio de salida
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("\nGenerando visualizaciones...")

        # 1. Distribución de scores
        self.plot_score_distribution(
            provider_scores,
            save_path=output_path / 'score_distribution.png'
        )

        # 2. Top 20 seccion_C más riesgosos
        self.plot_top_risky_providers(
            provider_scores,
            n=20,
            save_path=output_path / 'top_20_risky.png'
        )

        # 3. Distribución por categoría
        self.plot_risk_categories(
            provider_scores,
            save_path=output_path / 'risk_categories.png'
        )

        # 4. Componentes del score (Top 10)
        self.plot_score_components(
            provider_scores,
            n=10,
            save_path=output_path / 'score_components.png'
        )

        # 5. Scatter: Monto vs Score
        self.plot_amount_vs_score(
            provider_scores,
            save_path=output_path / 'amount_vs_score.png'
        )

        # 6. Alertas por severidad
        self.plot_alerts_by_severity(
            alerts,
            save_path=output_path / 'alerts_severity.png'
        )

        # 7. Heatmap de componentes (Top 20)
        self.plot_components_heatmap(
            provider_scores,
            n=20,
            save_path=output_path / 'components_heatmap.png'
        )

        # 8. Timeline de últimas transacciones
        self.plot_activity_timeline(
            provider_scores,
            save_path=output_path / 'activity_timeline.png'
        )

        self.logger.info(f"Visualizaciones guardadas en: {output_path}")

    def plot_score_distribution(
        self,
        provider_scores: pd.DataFrame,
        save_path: Path
    ):
        """Gráfico de distribución de scores."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Histograma
        ax1.hist(
            provider_scores['final_risk_score'],
            bins=30,
            color='steelblue',
            edgecolor='black',
            alpha=0.7
        )

        # Líneas verticales para umbrales
        ax1.axvline(x=0.4, color='red', linestyle='--', lw=2, label='Alto/Medio')
        ax1.axvline(x=0.7, color='green', linestyle='--', lw=2, label='Medio/Bajo')

        ax1.set_xlabel('Risk Score', fontsize=12)
        ax1.set_ylabel('Frecuencia', fontsize=12)
        ax1.set_title('Distribución de Risk Scores', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(
            provider_scores['final_risk_score'],
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color='red', linewidth=2)
        )

        ax2.set_ylabel('Risk Score', fontsize=12)
        ax2.set_title('Box Plot de Risk Scores', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Estadísticas
        mean_score = provider_scores['final_risk_score'].mean()
        median_score = provider_scores['final_risk_score'].median()
        ax2.text(
            1.15, mean_score,
            f'Media: {mean_score:.3f}',
            fontsize=10,
            color='blue'
        )
        ax2.text(
            1.15, median_score,
            f'Mediana: {median_score:.3f}',
            fontsize=10,
            color='red'
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_top_risky_providers(
        self,
        provider_scores: pd.DataFrame,
        n: int,
        save_path: Path
    ):
        """Gráfico de top N seccion_C más riesgosos."""
        top_n = provider_scores.head(n)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Colores según categoría
        colors = top_n['risk_category'].map({
            'alto': 'red',
            'medio': 'orange',
            'bajo': 'green'
        })

        # Barras horizontales
        y_pos = np.arange(len(top_n))
        ax.barh(y_pos, top_n['final_risk_score'], color=colors, edgecolor='black')

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"ID {pid}" for pid in top_n['proveedor_id']])
        ax.invert_yaxis()
        ax.set_xlabel('Risk Score', fontsize=12)
        ax.set_title(f'Top {n} Proveedores Más Riesgosos', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Líneas de referencia
        ax.axvline(x=0.4, color='red', linestyle='--', lw=1, alpha=0.5)
        ax.axvline(x=0.7, color='green', linestyle='--', lw=1, alpha=0.5)

        # Leyenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Alto Riesgo'),
            Patch(facecolor='orange', label='Medio Riesgo'),
            Patch(facecolor='green', label='Bajo Riesgo')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_risk_categories(
        self,
        provider_scores: pd.DataFrame,
        save_path: Path
    ):
        """Gráfico de distribución por categoría."""
        category_counts = provider_scores['risk_category'].value_counts()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Gráfico de barras
        colors = {'alto': 'red', 'medio': 'orange', 'bajo': 'green'}
        category_order = ['alto', 'medio', 'bajo']

        bars = ax1.bar(
            range(len(category_order)),
            [category_counts.get(cat, 0) for cat in category_order],
            color=[colors[cat] for cat in category_order],
            edgecolor='black'
        )

        ax1.set_xticks(range(len(category_order)))
        ax1.set_xticklabels([cat.upper() for cat in category_order])
        ax1.set_ylabel('Cantidad de Proveedores', fontsize=12)
        ax1.set_title('Distribución por Categoría de Riesgo', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Agregar valores en barras
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontweight='bold'
            )

        # Gráfico de pastel
        pie_data = [category_counts.get(cat, 0) for cat in category_order]
        pie_colors = [colors[cat] for cat in category_order]

        wedges, texts, autotexts = ax2.pie(
            pie_data,
            labels=[cat.upper() for cat in category_order],
            colors=pie_colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.1, 0, 0)
        )

        ax2.set_title('Proporción de Proveedores', fontsize=14, fontweight='bold')

        # Estilo de texto
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_score_components(
        self,
        provider_scores: pd.DataFrame,
        n: int,
        save_path: Path
    ):
        """Gráfico de componentes del score para top N."""
        top_n = provider_scores.head(n)

        components = [
            'score_cumplimiento_avg',
            'payment_delay_frequency',
            'amount_volatility',
            'recent_performance'
        ]

        component_labels = [
            'Cumplimiento\nPromedio',
            'Frecuencia\nPagos',
            'Estabilidad\nMontos',
            'Performance\nReciente'
        ]

        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(top_n))
        width = 0.2

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i, (comp, label) in enumerate(zip(components, component_labels)):
            ax.bar(
                x + i * width,
                top_n[comp],
                width,
                label=label,
                color=colors[i]
            )

        ax.set_xlabel('Proveedor ID', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Componentes del Score - Top {n} Proveedores', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([f'{pid}' for pid in top_n['proveedor_id']], rotation=45)
        ax.legend(loc='best', ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_amount_vs_score(
        self,
        provider_scores: pd.DataFrame,
        save_path: Path
    ):
        """Scatter plot de monto total vs risk score."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Colores por categoría
        category_colors = {
            'alto': 'red',
            'medio': 'orange',
            'bajo': 'green'
        }

        for category in ['alto', 'medio', 'bajo']:
            cat_data = provider_scores[provider_scores['risk_category'] == category]
            if len(cat_data) > 0:
                ax.scatter(
                    cat_data['final_risk_score'],
                    cat_data['total_amount'],
                    c=category_colors[category],
                    label=category.upper(),
                    s=100,
                    alpha=0.6,
                    edgecolors='black'
                )

        ax.set_xlabel('Risk Score', fontsize=12)
        ax.set_ylabel('Monto Total ($)', fontsize=12)
        ax.set_title('Monto Total vs Risk Score', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Líneas verticales de umbrales
        ax.axvline(x=0.4, color='red', linestyle='--', lw=1, alpha=0.5)
        ax.axvline(x=0.7, color='green', linestyle='--', lw=1, alpha=0.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_alerts_by_severity(
        self,
        alerts: List,
        save_path: Path
    ):
        """Gráfico de alertas por severidad."""
        if not alerts:
            # Sin alertas, crear gráfico vacío
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5, 0.5,
                'No hay alertas generadas',
                ha='center',
                va='center',
                fontsize=16
            )
            ax.axis('off')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return

        # Contar alertas por severidad
        severity_counts = {'CRITICA': 0, 'ALTA': 0, 'MEDIA': 0, 'BAJA': 0}
        for alert in alerts:
            for provider_alert in alert['alerts']:
                severity_counts[provider_alert['severity']] += 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Gráfico de barras
        severities = list(severity_counts.keys())
        counts = list(severity_counts.values())
        colors = ['darkred', 'red', 'orange', 'yellow']

        bars = ax1.bar(severities, counts, color=colors, edgecolor='black')
        ax1.set_ylabel('Cantidad de Alertas', fontsize=12)
        ax1.set_title('Alertas por Severidad', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Agregar valores
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{int(height)}',
                    ha='center',
                    va='bottom',
                    fontweight='bold'
                )

        # Gráfico de pastel (solo severidades con alertas)
        pie_data = [count for count in counts if count > 0]
        pie_labels = [sev for sev, count in zip(severities, counts) if count > 0]
        pie_colors = [colors[i] for i, count in enumerate(counts) if count > 0]

        if pie_data:
            ax2.pie(
                pie_data,
                labels=pie_labels,
                colors=pie_colors,
                autopct='%1.1f%%',
                startangle=90
            )
            ax2.set_title('Proporción de Alertas', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_components_heatmap(
        self,
        provider_scores: pd.DataFrame,
        n: int,
        save_path: Path
    ):
        """Heatmap de componentes del score."""
        top_n = provider_scores.head(n)

        components = [
            'score_cumplimiento_avg',
            'payment_delay_frequency',
            'amount_volatility',
            'recent_performance'
        ]

        component_labels = [
            'Cumplimiento',
            'Frecuencia Pagos',
            'Estabilidad',
            'Performance'
        ]

        # Crear matriz
        data = top_n[components].T.values

        fig, ax = plt.subplots(figsize=(12, 6))

        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # Etiquetas
        ax.set_yticks(range(len(component_labels)))
        ax.set_yticklabels(component_labels)
        ax.set_xticks(range(len(top_n)))
        ax.set_xticklabels([f'{pid}' for pid in top_n['proveedor_id']], rotation=45)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score', rotation=270, labelpad=20)

        # Título
        ax.set_title(f'Heatmap de Componentes - Top {n}', fontsize=14, fontweight='bold')

        # Agregar valores
        for i in range(len(component_labels)):
            for j in range(len(top_n)):
                text = ax.text(
                    j, i, f'{data[i, j]:.2f}',
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8
                )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_activity_timeline(
        self,
        provider_scores: pd.DataFrame,
        save_path: Path
    ):
        """Timeline de actividad de seccion_C."""
        # Ordenar por última transacción
        sorted_df = provider_scores.sort_values('last_transaction_date', ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Convertir fechas
        dates = pd.to_datetime(sorted_df['last_transaction_date'])

        # Colores por categoría
        colors = sorted_df['risk_category'].map({
            'alto': 'red',
            'medio': 'orange',
            'bajo': 'green'
        })

        # Scatter plot
        y_pos = range(len(sorted_df))
        ax.scatter(dates, y_pos, c=colors, s=100, edgecolors='black', zorder=3)

        # Líneas horizontales
        for i, (idx, row) in enumerate(sorted_df.iterrows()):
            first_date = pd.to_datetime(row['first_transaction_date'])
            last_date = pd.to_datetime(row['last_transaction_date'])
            ax.plot(
                [first_date, last_date],
                [i, i],
                'gray',
                alpha=0.3,
                linewidth=2,
                zorder=1
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"ID {pid}" for pid in sorted_df['proveedor_id']])
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_title('Timeline de Actividad - Top 20 por Última Transacción',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Leyenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Alto Riesgo'),
            Patch(facecolor='orange', label='Medio Riesgo'),
            Patch(facecolor='green', label='Bajo Riesgo')
        ]
        ax.legend(handles=legend_elements, loc='best')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_risk_scoring_visualizations(
    provider_scores: pd.DataFrame,
    alerts: List,
    output_dir: str
):
    """
    Función auxiliar para crear todas las visualizaciones.

    Args:
        provider_scores: DataFrame con scores
        alerts: Lista de alertas
        output_dir: Directorio de salida
    """
    visualizer = RiskScoringVisualizer()
    visualizer.create_all_visualizations(provider_scores, alerts, output_dir)


if __name__ == "__main__":
    # Test
    import sys
    from pathlib import Path

    # Agregar src al path
    BASE_DIR = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(BASE_DIR))

    from risk_scoring import risk_scoring_pipeline
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Ejecutar scoring
    scorer = risk_scoring_pipeline()

    # Crear visualizaciones
    create_risk_scoring_visualizations(
        provider_scores=scorer.provider_scores,
        alerts=scorer.alerts,
        output_dir='output/reports/risk_scoring/visualizations'
    )

    print("\nVisualizaciones creadas exitosamente")