"""
Model Evaluation Module

Módulo para evaluación detallada de modelos de Machine Learning.
Incluye visualizaciones, métricas y análisis de errores.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging
from datetime import datetime

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

from src.seccion_B.config import MODELS_DIR, REPORTS_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH


class ModelEvaluator:
    """
    Evaluador de modelos de Machine Learning.

    Genera visualizaciones, métricas detalladas y análisis de errores.
    """

    def __init__(
            self,
            models_dict: Dict,
            X_train: pd.DataFrame,
            X_test: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series,
            feature_names: List[str]
    ):
        """
        Inicializa el evaluador.

        Args:
            models_dict: Diccionario con modelos entrenados {nombre: modelo}
            X_train: Features de entrenamiento
            X_test: Features de test
            y_train: Target de entrenamiento
            y_test: Target de test
            feature_names: Nombres de las features
        """
        self.models_dict = models_dict
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names

        # Configurar logging
        self.logger = logging.getLogger(__name__)

        # Configurar estilo de gráficos
        self._setup_plot_style()

        # Almacenar resultados
        self.evaluation_results: Dict = {}

        self.logger.info("ModelEvaluator inicializado")
        self.logger.info(f"Modelos a evaluar: {list(models_dict.keys())}")

    def _setup_plot_style(self):
        """Configura el estilo de los gráficos."""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        # Configuración global
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10

    def evaluate_all_models(
            self,
            save_plots: bool = True,
            output_dir: Optional[str] = None
    ) -> Dict:
        """
        Evalúa todos los modelos.

        Args:
            save_plots: Si guardar los gráficos
            output_dir: Directorio de salida

        Returns:
            Diccionario con resultados de evaluación
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("INICIANDO EVALUACION DE MODELOS")
        self.logger.info("=" * 80)

        output_path = Path(output_dir or REPORTS_DIR) / 'evaluation'
        output_path.mkdir(parents=True, exist_ok=True)

        for model_name, model in self.models_dict.items():
            self.logger.info(f"\nEvaluando: {model_name}")

            # Predecir
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]

            # Evaluar
            results = self._evaluate_single_model(
                model_name=model_name,
                model=model,
                y_true=self.y_test,
                y_pred=y_pred,
                y_pred_proba=y_pred_proba
            )

            self.evaluation_results[model_name] = results

            # Guardar gráficos
            if save_plots:
                self._save_model_plots(
                    model_name=model_name,
                    model=model,
                    y_pred=y_pred,
                    y_pred_proba=y_pred_proba,
                    output_path=output_path
                )

        # Comparación entre modelos
        if save_plots:
            self._create_comparison_plots(output_path)

        # Guardar resultados
        self._save_evaluation_results(output_path)

        self.logger.info("\n" + "=" * 80)
        self.logger.info("EVALUACION COMPLETADA")
        self.logger.info("=" * 80)

        return self.evaluation_results

    def _evaluate_single_model(
            self,
            model_name: str,
            model,
            y_true: pd.Series,
            y_pred: np.ndarray,
            y_pred_proba: np.ndarray
    ) -> Dict:
        """
        Evalúa un modelo individual.

        Args:
            model_name: Nombre del modelo
            model: Modelo entrenado
            y_true: Valores reales
            y_pred: Predicciones
            y_pred_proba: Probabilidades predichas

        Returns:
            Diccionario con métricas
        """
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Classification report
        class_report = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        # Curvas ROC y PR
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)

        # Feature importance (si está disponible)
        feature_importance = self._get_feature_importance(model, model_name)

        results = {
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp),
                'total': int(tn + fp + fn + tp)
            },
            'classification_report': class_report,
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': float(roc_auc)
            },
            'pr_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'avg_precision': float(avg_precision)
            },
            'feature_importance': feature_importance,
            'error_analysis': self._analyze_errors(y_true, y_pred, y_pred_proba)
        }

        return results

    def _get_feature_importance(
            self,
            model,
            model_name: str
    ) -> Optional[Dict]:
        """
        Extrae feature importance del modelo.

        Args:
            model: Modelo entrenado
            model_name: Nombre del modelo

        Returns:
            Diccionario con importancias o None
        """
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based seccion_B (RF, XGBoost)
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear seccion_B (LR)
                importances = np.abs(model.coef_[0])
            else:
                return None

            # Crear diccionario ordenado
            importance_dict = dict(zip(self.feature_names, importances))
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )

            return {
                'features': list(importance_dict.keys()),
                'importances': [float(v) for v in importance_dict.values()]
            }

        except Exception as e:
            self.logger.warning(f"No se pudo extraer feature importance de {model_name}: {e}")
            return None

    def _analyze_errors(
            self,
            y_true: pd.Series,
            y_pred: np.ndarray,
            y_pred_proba: np.ndarray
    ) -> Dict:
        """
        Analiza los errores del modelo.

        Args:
            y_true: Valores reales
            y_pred: Predicciones
            y_pred_proba: Probabilidades predichas

        Returns:
            Diccionario con análisis de errores
        """
        # Identificar errores
        errors = y_true != y_pred
        false_positives = (y_true == 0) & (y_pred == 1)
        false_negatives = (y_true == 1) & (y_pred == 0)

        # Estadísticas de probabilidades en errores
        fp_probs = y_pred_proba[false_positives] if false_positives.sum() > 0 else []
        fn_probs = y_pred_proba[false_negatives] if false_negatives.sum() > 0 else []

        return {
            'total_errors': int(errors.sum()),
            'error_rate': float(errors.mean()),
            'false_positives': {
                'count': int(false_positives.sum()),
                'avg_probability': float(np.mean(fp_probs)) if len(fp_probs) > 0 else 0,
                'std_probability': float(np.std(fp_probs)) if len(fp_probs) > 0 else 0
            },
            'false_negatives': {
                'count': int(false_negatives.sum()),
                'avg_probability': float(np.mean(fn_probs)) if len(fn_probs) > 0 else 0,
                'std_probability': float(np.std(fn_probs)) if len(fn_probs) > 0 else 0
            }
        }

    def _save_model_plots(
            self,
            model_name: str,
            model,
            y_pred: np.ndarray,
            y_pred_proba: np.ndarray,
            output_path: Path
    ):
        """
        Guarda los gráficos de un modelo.

        Args:
            model_name: Nombre del modelo
            model: Modelo entrenado
            y_pred: Predicciones
            y_pred_proba: Probabilidades
            output_path: Directorio de salida
        """
        # Crear subdirectorio para el modelo
        model_path = output_path / model_name
        model_path.mkdir(exist_ok=True)

        # 1. Matriz de confusión
        self._plot_confusion_matrix(
            y_true=self.y_test,
            y_pred=y_pred,
            title=f'Confusion Matrix - {model_name}',
            save_path=model_path / 'confusion_matrix.png'
        )

        # 2. Curva ROC
        self._plot_roc_curve(
            y_true=self.y_test,
            y_pred_proba=y_pred_proba,
            title=f'ROC Curve - {model_name}',
            save_path=model_path / 'roc_curve.png'
        )

        # 3. Curva Precision-Recall
        self._plot_precision_recall_curve(
            y_true=self.y_test,
            y_pred_proba=y_pred_proba,
            title=f'Precision-Recall Curve - {model_name}',
            save_path=model_path / 'pr_curve.png'
        )

        # 4. Feature importance (si disponible)
        if self.evaluation_results[model_name]['feature_importance'] is not None:
            self._plot_feature_importance(
                feature_importance=self.evaluation_results[model_name]['feature_importance'],
                title=f'Feature Importance - {model_name}',
                save_path=model_path / 'feature_importance.png'
            )

        # 5. Distribución de probabilidades
        self._plot_probability_distribution(
            y_true=self.y_test,
            y_pred_proba=y_pred_proba,
            title=f'Probability Distribution - {model_name}',
            save_path=model_path / 'probability_distribution.png'
        )

        self.logger.info(f"  Graficos guardados en: {model_path}")

    def _plot_confusion_matrix(
            self,
            y_true: pd.Series,
            y_pred: np.ndarray,
            title: str,
            save_path: Path
    ):
        """Grafica matriz de confusión."""
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Bajo Riesgo (0)', 'Alto Riesgo (1)'],
            yticklabels=['Bajo Riesgo (0)', 'Alto Riesgo (1)'],
            ax=ax,
            cbar_kws={'label': 'Count'}
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12)

        # Agregar porcentajes
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / total * 100
                ax.text(
                    j + 0.5, i + 0.7,
                    f'({percentage:.1f}%)',
                    ha='center',
                    va='center',
                    fontsize=9,
                    color='gray'
                )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_roc_curve(
            self,
            y_true: pd.Series,
            y_pred_proba: np.ndarray,
            title: str,
            save_path: Path
    ):
        """Grafica curva ROC."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Curva ROC
        ax.plot(
            fpr, tpr,
            color='darkorange',
            lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})'
        )

        # Línea diagonal (random classifier)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_precision_recall_curve(
            self,
            y_true: pd.Series,
            y_pred_proba: np.ndarray,
            title: str,
            save_path: Path
    ):
        """Grafica curva Precision-Recall."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(
            recall, precision,
            color='blue',
            lw=2,
            label=f'PR curve (AP = {avg_precision:.3f})'
        )

        # Línea base (proporción de positivos)
        baseline = y_true.mean()
        ax.plot([0, 1], [baseline, baseline], color='red', lw=2, linestyle='--', label=f'Baseline ({baseline:.3f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_importance(
            self,
            feature_importance: Dict,
            title: str,
            save_path: Path,
            top_n: int = 15
    ):
        """Grafica feature importance."""
        features = feature_importance['features'][:top_n]
        importances = feature_importance['importances'][:top_n]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Barras horizontales
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, color='steelblue')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_probability_distribution(
            self,
            y_true: pd.Series,
            y_pred_proba: np.ndarray,
            title: str,
            save_path: Path
    ):
        """Grafica distribución de probabilidades."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Separar por clase real
        proba_class_0 = y_pred_proba[y_true == 0]
        proba_class_1 = y_pred_proba[y_true == 1]

        # Histogramas
        ax.hist(
            proba_class_0,
            bins=30,
            alpha=0.6,
            label='Bajo Riesgo (0)',
            color='green',
            edgecolor='black'
        )
        ax.hist(
            proba_class_1,
            bins=30,
            alpha=0.6,
            label='Alto Riesgo (1)',
            color='red',
            edgecolor='black'
        )

        # Línea de threshold
        ax.axvline(x=0.5, color='black', linestyle='--', lw=2, label='Threshold (0.5)')

        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_comparison_plots(self, output_path: Path):
        """Crea gráficos comparativos entre modelos."""
        self.logger.info("\nGenerando graficos de comparacion...")

        # 1. Comparación de ROC curves
        self._plot_roc_comparison(output_path / 'roc_comparison.png')

        # 2. Comparación de métricas
        self._plot_metrics_comparison(output_path / 'metrics_comparison.png')

        # 3. Comparación de matrices de confusión
        self._plot_confusion_matrices_comparison(output_path / 'confusion_comparison.png')

        self.logger.info(f"  Graficos de comparacion guardados en: {output_path}")

    def _plot_roc_comparison(self, save_path: Path):
        """Compara curvas ROC de todos los modelos."""
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = ['darkorange', 'green', 'purple', 'brown', 'pink']

        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            roc_data = results['roc_curve']
            fpr = np.array(roc_data['fpr'])
            tpr = np.array(roc_data['tpr'])
            roc_auc = roc_data['auc']

            ax.plot(
                fpr, tpr,
                color=colors[i % len(colors)],
                lw=2,
                label=f'{model_name} (AUC = {roc_auc:.3f})'
            )

        # Línea diagonal
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_metrics_comparison(self, save_path: Path):
        """Compara métricas principales de todos los modelos."""
        metrics_names = ['accuracy', 'precision', 'recall', 'f1-score']

        data = []
        for model_name, results in self.evaluation_results.items():
            class_1_metrics = results['classification_report']['1']
            data.append({
                'Model': model_name,
                'Accuracy': results['classification_report']['accuracy'],
                'Precision': class_1_metrics['precision'],
                'Recall': class_1_metrics['recall'],
                'F1-Score': class_1_metrics['f1-score']
            })

        df_metrics = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(df_metrics))
        width = 0.2

        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i, metric in enumerate(metrics_to_plot):
            ax.bar(
                x + i * width,
                df_metrics[metric],
                width,
                label=metric,
                color=colors[i]
            )

        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(df_metrics['Model'])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrices_comparison(self, save_path: Path):
        """Compara matrices de confusión de todos los modelos."""
        n_models = len(self.evaluation_results)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

        if n_models == 1:
            axes = [axes]

        for ax, (model_name, results) in zip(axes, self.evaluation_results.items()):
            cm_data = results['confusion_matrix']
            cm = np.array([
                [cm_data['tn'], cm_data['fp']],
                [cm_data['fn'], cm_data['tp']]
            ])

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Low (0)', 'High (1)'],
                yticklabels=['Low (0)', 'High (1)'],
                ax=ax,
                cbar=False
            )

            ax.set_title(model_name, fontsize=12, fontweight='bold')
            ax.set_ylabel('Actual' if ax == axes[0] else '', fontsize=10)
            ax.set_xlabel('Predicted', fontsize=10)

        plt.suptitle('Confusion Matrices Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _save_evaluation_results(self, output_path: Path):
        """Guarda resultados de evaluación en JSON."""
        results_file = output_path / 'evaluation_results.json'

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"\nResultados guardados en: {results_file}")

    def get_best_model(self, metric: str = 'f1') -> Tuple[str, Dict]:
        """
        Obtiene el mejor modelo según una métrica.

        Args:
            metric: Métrica a usar ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')

        Returns:
            Tuple con (nombre_modelo, resultados)
        """
        best_score = -1
        best_model_name = None

        for model_name, results in self.evaluation_results.items():
            if metric == 'roc_auc':
                score = results['roc_curve']['auc']
            else:
                class_1_metrics = results['classification_report']['1']
                if metric == 'f1':
                    score = class_1_metrics['f1-score']
                elif metric == 'accuracy':
                    score = results['classification_report']['accuracy']
                elif metric == 'precision':
                    score = class_1_metrics['precision']
                elif metric == 'recall':
                    score = class_1_metrics['recall']
                else:
                    score = class_1_metrics['f1-score']

            if score > best_score:
                best_score = score
                best_model_name = model_name

        return best_model_name, self.evaluation_results[best_model_name]

    def print_summary(self):
        """Imprime resumen de la evaluación."""
        print("\n" + "=" * 80)
        print("RESUMEN DE EVALUACION DE MODELOS")
        print("=" * 80)

        for model_name, results in self.evaluation_results.items():
            print(f"\n{model_name.upper()}")
            print("-" * 60)

            # Métricas principales
            class_report = results['classification_report']
            class_1 = class_report['1']

            print(f"Accuracy:  {class_report['accuracy']:.4f}")
            print(f"Precision: {class_1['precision']:.4f}")
            print(f"Recall:    {class_1['recall']:.4f}")
            print(f"F1-Score:  {class_1['f1-score']:.4f}")
            print(f"ROC AUC:   {results['roc_curve']['auc']:.4f}")

            # Matriz de confusión
            cm = results['confusion_matrix']
            print(f"\nConfusion Matrix:")
            print(f"  TN: {cm['tn']}  FP: {cm['fp']}")
            print(f"  FN: {cm['fn']}  TP: {cm['tp']}")

            # Análisis de errores
            errors = results['error_analysis']
            print(f"\nError Analysis:")
            print(f"  Total Errors: {errors['total_errors']} ({errors['error_rate']:.2%})")
            print(f"  False Positives: {errors['false_positives']['count']}")
            print(f"  False Negatives: {errors['false_negatives']['count']}")

        # Mejor modelo
        best_model, _ = self.get_best_model('f1')
        print("\n" + "=" * 80)
        print(f"MEJOR MODELO (por F1-Score): {best_model}")
        print("=" * 80)


def evaluate_models_pipeline(
        train_path: str = None,
        test_path: str = None,
        models_dir: str = None,
        save_plots: bool = True
) -> ModelEvaluator:
    """
    Pipeline principal de evaluación de modelos.

    Args:
        train_path: Ruta del archivo train_data.csv
        test_path: Ruta del archivo test_data.csv
        models_dir: Directorio con modelos guardados
        save_plots: Si guardar gráficos

    Returns:
        Evaluador con resultados
    """
    import joblib

    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE DE EVALUACION DE MODELOS")
    logger.info("=" * 80)

    # Cargar datos
    train_path = train_path or TRAIN_DATA_PATH
    test_path = test_path or TEST_DATA_PATH

    logger.info(f"Cargando datos de entrenamiento: {train_path}")
    train_df = pd.read_csv(train_path)

    logger.info(f"Cargando datos de test: {test_path}")
    test_df = pd.read_csv(test_path)

    # Separar features y target
    target_col = 'risk_target'
    exclude_cols = [target_col, 'proveedor_id', 'fecha', 'score_cumplimiento']

    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Cargar modelos
    models_path = Path(models_dir or MODELS_DIR)
    models_dict = {}

    model_files = {
        'logistic_regression': 'logistic_regression_model.pkl',
        'random_forest': 'random_forest_model.pkl',
        'xgboost': 'xgboost_model.pkl'
    }

    for model_name, filename in model_files.items():
        model_file = models_path / filename
        if model_file.exists():
            logger.info(f"Cargando modelo: {model_name}")
            models_dict[model_name] = joblib.load(model_file)
        else:
            logger.warning(f"Modelo no encontrado: {model_file}")

    if not models_dict:
        raise ValueError("No se encontraron modelos para evaluar")

    # Crear evaluador
    evaluator = ModelEvaluator(
        models_dict=models_dict,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_cols
    )

    # Evaluar
    evaluator.evaluate_all_models(save_plots=save_plots)

    # Imprimir resumen
    evaluator.print_summary()

    return evaluator


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Ejecutar evaluación
    evaluator = evaluate_models_pipeline(save_plots=True)