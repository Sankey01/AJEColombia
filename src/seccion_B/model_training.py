"""
Model Training Module

Módulo para entrenamiento de modelos de Machine Learning.
Implementa Logistic Regression, Random Forest y XGBoost con optimización
de hiperparámetros y cross-validation temporal.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, cohen_kappa_score,
    classification_report, confusion_matrix
)
import joblib
import logging
from pathlib import Path
import json
from datetime import datetime

# Import condicional de XGBoost
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from .config import (
    MODEL_CONFIG,
    LOGISTIC_CONFIG,
    RANDOM_FOREST_CONFIG,
    XGBOOST_CONFIG,
    MODELS_DIR,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    get_model_params
)


class ModelTrainer:
    """
    Clase principal para entrenamiento de modelos de ML.

    Entrena múltiples modelos con optimización de hiperparámetros
    y evaluación con métricas estándar.
    """

    def __init__(self, config=None):
        """
        Inicializa el entrenador de modelos.

        Args:
            config: Objeto de configuración (default: MODEL_CONFIG)
        """
        self.config = config if config else MODEL_CONFIG

        self.models: Dict[str, Any] = {}
        self.best_models: Dict[str, Any] = {}
        self.training_results: Dict[str, Dict] = {}

        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def train_all_models(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            optimize_hyperparameters: bool = True
    ) -> Dict[str, Any]:
        """
        Entrena todos los modelos configurados.

        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_test: Features de test
            y_test: Target de test
            optimize_hyperparameters: Si optimizar hiperparámetros con GridSearch

        Returns:
            Diccionario con resultados de entrenamiento
        """
        self.logger.info("Iniciando entrenamiento de modelos")
        self.logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        results = {}

        for model_name in self.config.models_to_train:
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Entrenando: {model_name}")
            self.logger.info('=' * 60)

            try:
                if model_name == 'logistic_regression':
                    result = self.train_logistic_regression(
                        X_train, y_train, X_test, y_test,
                        optimize_hyperparameters
                    )
                elif model_name == 'random_forest':
                    result = self.train_random_forest(
                        X_train, y_train, X_test, y_test,
                        optimize_hyperparameters
                    )
                elif model_name == 'xgboost':
                    result = self.train_xgboost(
                        X_train, y_train, X_test, y_test,
                        optimize_hyperparameters
                    )
                else:
                    self.logger.warning(f"Modelo '{model_name}' no reconocido")
                    continue

                results[model_name] = result
                self.training_results[model_name] = result

                self.logger.info(f"✓ {model_name} entrenado exitosamente")

            except Exception as e:
                self.logger.error(f"Error entrenando {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Entrenamiento completado para todos los modelos")
        self.logger.info("=" * 60)

        return results

    def train_logistic_regression(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            optimize: bool = True
    ) -> Dict[str, Any]:
        """
        Entrena modelo de Logistic Regression.

        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_test: Features de test
            y_test: Target de test
            optimize: Si optimizar hiperparámetros

        Returns:
            Diccionario con resultados
        """
        self.logger.info("Configurando Logistic Regression")

        if optimize:
            self.logger.info("Optimizando hiperparámetros con GridSearchCV")

            # Crear cross-validator
            cv = self._get_cross_validator(X_train, y_train)

            # Grid Search
            grid_search = GridSearchCV(
                LogisticRegression(random_state=self.config.random_state),
                param_grid=LOGISTIC_CONFIG.param_grid,
                cv=cv,
                scoring='f1',
                n_jobs=self.config.n_jobs,
                verbose=1
            )

            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_

            self.logger.info(f"Mejores parámetros: {best_params}")
            self.logger.info(f"CV Score (F1): {cv_score:.4f}")
        else:
            self.logger.info("Usando parámetros por defecto")
            best_params = LOGISTIC_CONFIG.default_params
            best_model = LogisticRegression(**best_params)
            best_model.fit(X_train, y_train)
            cv_score = None

        # Guardar modelo
        self.best_models['logistic_regression'] = best_model

        # Evaluar
        metrics = self._evaluate_model(best_model, X_train, y_train, X_test, y_test)

        return {
            'model': best_model,
            'best_params': best_params,
            'cv_score': cv_score,
            'metrics': metrics
        }

    def train_random_forest(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            optimize: bool = True
    ) -> Dict[str, Any]:
        """
        Entrena modelo de Random Forest.

        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_test: Features de test
            y_test: Target de test
            optimize: Si optimizar hiperparámetros

        Returns:
            Diccionario con resultados
        """
        self.logger.info("Configurando Random Forest")

        if optimize:
            self.logger.info("Optimizando hiperparámetros con GridSearchCV")

            # Crear cross-validator
            cv = self._get_cross_validator(X_train, y_train)

            # Grid Search
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=self.config.random_state),
                param_grid=RANDOM_FOREST_CONFIG.param_grid,
                cv=cv,
                scoring='f1',
                n_jobs=self.config.n_jobs,
                verbose=1
            )

            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_

            self.logger.info(f"Mejores parámetros: {best_params}")
            self.logger.info(f"CV Score (F1): {cv_score:.4f}")
        else:
            self.logger.info("Usando parámetros por defecto")
            best_params = RANDOM_FOREST_CONFIG.default_params
            best_model = RandomForestClassifier(**best_params)
            best_model.fit(X_train, y_train)
            cv_score = None

        # Guardar modelo
        self.best_models['random_forest'] = best_model

        # Evaluar
        metrics = self._evaluate_model(best_model, X_train, y_train, X_test, y_test)

        # Importancia de features
        feature_importance = self._get_feature_importance(
            best_model, X_train.columns
        )

        return {
            'model': best_model,
            'best_params': best_params,
            'cv_score': cv_score,
            'metrics': metrics,
            'feature_importance': feature_importance
        }

    def train_xgboost(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            optimize: bool = True
    ) -> Dict[str, Any]:
        """
        Entrena modelo de XGBoost.

        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_test: Features de test
            y_test: Target de test
            optimize: Si optimizar hiperparámetros

        Returns:
            Diccionario con resultados
        """
        if not XGBOOST_AVAILABLE:
            self.logger.error("XGBoost no está instalado")
            return {'error': 'XGBoost no disponible'}

        self.logger.info("Configurando XGBoost")

        if optimize:
            self.logger.info("Optimizando hiperparámetros con GridSearchCV")

            # Crear cross-validator
            cv = self._get_cross_validator(X_train, y_train)

            # Grid Search
            grid_search = GridSearchCV(
                xgb.XGBClassifier(
                    random_state=self.config.random_state,
                    use_label_encoder=False
                ),
                param_grid=XGBOOST_CONFIG.param_grid,
                cv=cv,
                scoring='f1',
                n_jobs=self.config.n_jobs,
                verbose=1
            )

            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_

            self.logger.info(f"Mejores parámetros: {best_params}")
            self.logger.info(f"CV Score (F1): {cv_score:.4f}")
        else:
            self.logger.info("Usando parámetros por defecto")
            best_params = XGBOOST_CONFIG.default_params
            best_model = xgb.XGBClassifier(**best_params)
            best_model.fit(X_train, y_train)
            cv_score = None

        # Guardar modelo
        self.best_models['xgboost'] = best_model

        # Evaluar
        metrics = self._evaluate_model(best_model, X_train, y_train, X_test, y_test)

        # Importancia de features
        feature_importance = self._get_feature_importance(
            best_model, X_train.columns
        )

        return {
            'model': best_model,
            'best_params': best_params,
            'cv_score': cv_score,
            'metrics': metrics,
            'feature_importance': feature_importance
        }

    def _get_cross_validator(
            self,
            X: pd.DataFrame,
            y: pd.Series
    ):
        """
        Crea cross-validator según configuración.

        Args:
            X: Features
            y: Target

        Returns:
            Cross-validator
        """
        if self.config.cv_strategy == 'temporal':
            self.logger.info(f"Usando TimeSeriesSplit con {self.config.cv_folds} folds")
            return TimeSeriesSplit(n_splits=self.config.cv_folds)
        else:
            self.logger.info(f"Usando StratifiedKFold con {self.config.cv_folds} folds")
            return StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state
            )

    def _evaluate_model(
            self,
            model: Any,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame,
            y_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Evalúa modelo en train y test.

        Args:
            model: Modelo entrenado
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_test: Features de test
            y_test: Target de test

        Returns:
            Diccionario con métricas en train y test
        """
        self.logger.info("Evaluando modelo")

        # Predicciones
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Probabilidades para métricas que las requieren
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Calcular métricas
        metrics = {
            'train': self._compute_metrics(y_train, y_train_pred, y_train_proba),
            'test': self._compute_metrics(y_test, y_test_pred, y_test_proba)
        }

        # Log de métricas de test
        self.logger.info("Métricas en TEST:")
        for metric_name, value in metrics['test'].items():
            self.logger.info(f"  {metric_name}: {value:.4f}")

        return metrics

    def _compute_metrics(
            self,
            y_true: pd.Series,
            y_pred: np.ndarray,
            y_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcula métricas de evaluación.

        Args:
            y_true: Valores reales
            y_pred: Predicciones
            y_proba: Probabilidades

        Returns:
            Diccionario con métricas
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'average_precision': average_precision_score(y_true, y_proba),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred)
        }

    def _get_feature_importance(
            self,
            model: Any,
            feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Extrae importancia de features del modelo.

        Args:
            model: Modelo entrenado
            feature_names: Lista de nombres de features

        Returns:
            DataFrame con importancia de features ordenado
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_

            df_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })

            df_importance = df_importance.sort_values(
                'importance',
                ascending=False
            ).reset_index(drop=True)

            return df_importance
        else:
            return pd.DataFrame()

    def get_best_model(self, metric: str = 'f1') -> Tuple[str, Any]:
        """
        Retorna el mejor modelo según métrica especificada.

        Args:
            metric: Métrica para comparar modelos (default: 'f1')

        Returns:
            Tuple con (nombre_modelo, modelo)
        """
        best_name = None
        best_score = -np.inf
        best_model = None

        for model_name, results in self.training_results.items():
            if 'metrics' in results:
                score = results['metrics']['test'].get(metric, -np.inf)
                if score > best_score:
                    best_score = score
                    best_name = model_name
                    best_model = results['model']

        self.logger.info(f"Mejor modelo según {metric}: {best_name} (score: {best_score:.4f})")

        return best_name, best_model

    def save_models(self, output_dir: Optional[str] = None) -> None:
        """
        Guarda todos los modelos entrenados.

        Args:
            output_dir: Directorio de salida (default: MODELS_DIR)
        """
        if output_dir is None:
            output_dir = MODELS_DIR

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Guardando modelos en: {output_dir}")

        for model_name, model in self.best_models.items():
            # Guardar modelo
            model_file = output_path / f'{model_name}_model.pkl'
            joblib.dump(model, model_file)
            self.logger.info(f"  ✓ {model_name} guardado en: {model_file}")

        # Guardar resultados de entrenamiento
        results_file = output_path / 'training_results.json'

        # Convertir resultados a formato serializable
        serializable_results = {}
        for model_name, results in self.training_results.items():
            serializable_results[model_name] = {
                'best_params': results.get('best_params', {}),
                'cv_score': float(results['cv_score']) if results.get('cv_score') is not None else None,
                'metrics': results.get('metrics', {})
            }

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"  ✓ Resultados guardados en: {results_file}")

    def load_model(self, model_name: str, model_dir: Optional[str] = None) -> Any:
        """
        Carga un modelo guardado.

        Args:
            model_name: Nombre del modelo
            model_dir: Directorio donde están los modelos

        Returns:
            Modelo cargado
        """
        if model_dir is None:
            model_dir = MODELS_DIR

        model_file = Path(model_dir) / f'{model_name}_model.pkl'

        if not model_file.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_file}")

        model = joblib.load(model_file)
        self.logger.info(f"Modelo cargado: {model_file}")

        return model

    def print_summary(self) -> None:
        """
        Imprime resumen de resultados de entrenamiento.
        """
        print("\n" + "=" * 80)
        print("RESUMEN DE ENTRENAMIENTO DE MODELOS")
        print("=" * 80)

        for model_name, results in self.training_results.items():
            print(f"\n{'-' * 60}")
            print(f"{model_name.upper()}")
            print('-' * 60)

            if 'error' in results:
                print(f"  ERROR: {results['error']}")
                continue

            print(f"\nMejores parámetros:")
            for param, value in results.get('best_params', {}).items():
                print(f"  {param}: {value}")

            if results.get('cv_score') is not None:
                print(f"\nCV Score (F1): {results['cv_score']:.4f}")

            print(f"\nMétricas en TRAIN:")
            for metric, value in results['metrics']['train'].items():
                print(f"  {metric}: {value:.4f}")

            print(f"\nMétricas en TEST:")
            for metric, value in results['metrics']['test'].items():
                print(f"  {metric}: {value:.4f}")

        print("\n" + "=" * 80)


def train_models_pipeline(
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        optimize_hyperparameters: bool = False,
        save_models: bool = True
) -> Tuple[ModelTrainer, Dict[str, Any]]:
    """
    Pipeline completo de entrenamiento de modelos.

    Args:
        train_path: Ruta del dataset de entrenamiento
        test_path: Ruta del dataset de test
        optimize_hyperparameters: Si optimizar hiperparámetros
        save_models: Si guardar los modelos entrenados

    Returns:
        Tuple con (trainer, results)
    """
    logger = logging.getLogger(__name__)
    logger.info("Iniciando pipeline de entrenamiento")

    # Cargar datos
    if train_path is None:
        train_path = TRAIN_DATA_PATH
    if test_path is None:
        test_path = TEST_DATA_PATH

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Datos cargados - Train: {len(train_df)}, Test: {len(test_df)}")

    # Separar features y target
    exclude_columns = ['risk_target', 'score_cumplimiento', 'fecha', 'proveedor_id']

    feature_columns = [col for col in train_df.columns if col not in exclude_columns]

    X_train = train_df[feature_columns]
    y_train = train_df['risk_target']

    X_test = test_df[feature_columns]
    y_test = test_df['risk_target']

    logger.info(f"Features: {len(feature_columns)}")
    logger.info(f"Features list: {feature_columns}")

    # Entrenar modelos
    trainer = ModelTrainer()
    results = trainer.train_all_models(
        X_train, y_train,
        X_test, y_test,
        optimize_hyperparameters=optimize_hyperparameters
    )

    # Guardar modelos si se solicita
    if save_models:
        trainer.save_models()

    # Imprimir resumen
    trainer.print_summary()

    return trainer, results