"""
Configuration Module

Configuraciones globales para el sistema de modelado predictivo de riesgo de mora.
Incluye umbrales, rutas, hiperparámetros y configuraciones de preprocesamiento.
"""

from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field

# ============================================================
# RUTAS DE ARCHIVOS
# ============================================================

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'resource'
OUTPUT_DIR = BASE_DIR / 'output'
MODELS_DIR = OUTPUT_DIR / 'seccion_B'
REPORTS_DIR = OUTPUT_DIR / 'reports'

# Archivos de entrada
RAW_DATA_PATH = DATA_DIR / 'historicos_pagos.csv'
CLEANED_DATA_PATH = OUTPUT_DIR / 'historicos_pagos_cleaned.csv'

# Archivos de salida
PREPROCESSED_DATA_PATH = OUTPUT_DIR / 'historicos_pagos_preprocessed.csv'
TRAIN_DATA_PATH = OUTPUT_DIR / 'train_data.csv'
TEST_DATA_PATH = OUTPUT_DIR / 'test_data.csv'

# Crear directorios si no existen
for directory in [OUTPUT_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURACIÓN DE LIMPIEZA DE DATOS
# ============================================================

@dataclass
class DataCleaningConfig:
    """Configuración para limpieza de datos."""

    outlier_threshold: float = 1.5  # Umbral para detección de outliers (IQR method)
    remove_outliers: bool = False  # Si eliminar outliers o solo marcarlos

    # Categorías válidas por columna
    valid_categories: Dict[str, List[str]] = field(default_factory=lambda: {
        'categoria_gasto': ['Tecnología', 'Materiales', 'Logística', 'Servicios'],
        'método_pago': ['Efectivo', 'Cheque', 'Transferencia'],
        'indicadores_economicos': ['Creciente', 'Estable', 'Decreciente'],
        'estacionalidad': ['Alta', 'Baja']
    })


# ============================================================
# CONFIGURACIÓN DE PREPROCESAMIENTO
# ============================================================

@dataclass
class PreprocessingConfig:
    """Configuración para preprocesamiento y feature engineering."""

    # Variable objetivo
    target_column: str = 'score_cumplimiento'
    risk_threshold: float = 0.5  # Umbral configurable: < 0.5 = Alto Riesgo

    # Columnas para encoding
    nominal_columns: List[str] = field(default_factory=lambda: [
        'categoria_gasto',
        'método_pago'
    ])

    ordinal_columns: Dict[str, List[str]] = field(default_factory=lambda: {
        'indicadores_economicos': ['Decreciente', 'Estable', 'Creciente'],
        'estacionalidad': ['Baja', 'Alta']
    })

    # Columnas numéricas para normalización
    numerical_columns: List[str] = field(default_factory=lambda: [
        'monto',
        'tiempo_procesamiento'
    ])

    # Features temporales a crear
    temporal_features: List[str] = field(default_factory=lambda: [
        'year',
        'month',
        'quarter',
        'day_of_week',
        'day_of_month',
        'week_of_year',
        'is_month_start',
        'is_month_end',
        'is_quarter_start',
        'is_quarter_end'
    ])

    # Método de normalización
    scaler_method: str = 'standard'  # 'standard', 'minmax', 'robust'

    # Configuración de split temporal
    test_size: float = 0.2
    temporal_split: bool = True  # Split temporal vs aleatorio
    random_state: int = 42


# ============================================================
# CONFIGURACIÓN DE MODELOS
# ============================================================

@dataclass
class ModelConfig:
    """Configuración de hiperparámetros para modelos de ML."""

    random_state: int = 42
    n_jobs: int = -1  # Usar todos los cores disponibles

    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = 'temporal'  # 'temporal' o 'stratified'

    # Modelos a entrenar
    models_to_train: List[str] = field(default_factory=lambda: [
        'logistic_regression',
        'random_forest',
        'xgboost'
    ])


@dataclass
class LogisticRegressionConfig:
    """Hiperparámetros para Logistic Regression."""

    param_grid: Dict[str, List[Any]] = field(default_factory=lambda: {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000],
        'class_weight': [None, 'balanced']
    })

    default_params: Dict[str, Any] = field(default_factory=lambda: {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'liblinear',
        'max_iter': 1000,
        'random_state': 42,
        'class_weight': 'balanced'
    })


@dataclass
class RandomForestConfig:
    """Hiperparámetros para Random Forest."""

    param_grid: Dict[str, List[Any]] = field(default_factory=lambda: {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    })

    default_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced'
    })


@dataclass
class XGBoostConfig:
    """Hiperparámetros para XGBoost."""

    param_grid: Dict[str, List[Any]] = field(default_factory=lambda: {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.3],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'scale_pos_weight': [1, 2, 3]
    })

    default_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'scale_pos_weight': 2,
        'eval_metric': 'logloss'
    })


# ============================================================
# CONFIGURACIÓN DE EVALUACIÓN
# ============================================================

@dataclass
class EvaluationConfig:
    """Configuración para evaluación de modelos."""

    # Métricas principales
    primary_metric: str = 'f1'  # Métrica para selección de mejor modelo

    metrics_to_compute: List[str] = field(default_factory=lambda: [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'roc_auc',
        'average_precision',
        'cohen_kappa'
    ])

    # Umbrales de clasificación
    classification_thresholds: List[float] = field(default_factory=lambda: [
        0.3, 0.4, 0.5, 0.6, 0.7
    ])

    # Configuración de gráficos
    plot_confusion_matrix: bool = True
    plot_roc_curve: bool = True
    plot_precision_recall_curve: bool = True
    plot_feature_importance: bool = True
    top_n_features: int = 20  # Top N features más importantes

    # Formato de reportes
    export_format: List[str] = field(default_factory=lambda: ['json', 'csv', 'html'])


# ============================================================
# CONFIGURACIÓN DE SERIES TEMPORALES
# ============================================================

@dataclass
class TemporalAnalysisConfig:
    """Configuración para análisis de series temporales."""

    # Frecuencia de agregación
    aggregation_frequency: str = 'M'  # 'D': diario, 'W': semanal, 'M': mensual, 'Q': trimestral

    # Ventanas de tiempo para análisis rolling
    rolling_windows: List[int] = field(default_factory=lambda: [
        7,  # 1 semana
        30,  # 1 mes
        90,  # 1 trimestre
        180  # 6 meses
    ])

    # Configuración de forecasting
    forecast_horizon: int = 30  # Días hacia adelante
    forecast_method: str = 'arima'  # 'arima', 'prophet', 'exponential_smoothing'

    # Detección de estacionalidad
    seasonal_periods: List[int] = field(default_factory=lambda: [
        7,  # Semanal
        30,  # Mensual
        90,  # Trimestral
        365  # Anual
    ])

    # Test de estacionariedad
    stationarity_test: str = 'adf'  # 'adf' (Augmented Dickey-Fuller) o 'kpss'
    significance_level: float = 0.05


# ============================================================
# CONFIGURACIÓN DE RISK SCORING
# ============================================================

@dataclass
class RiskScoringConfig:
    """Configuración para cálculo de score de riesgo por proveedor."""

    # Pesos para componentes del score
    weights: Dict[str, float] = field(default_factory=lambda: {
        'score_cumplimiento_avg': 0.40,  # Promedio histórico de cumplimiento
        'payment_delay_frequency': 0.25,  # Frecuencia de retrasos
        'amount_volatility': 0.15,  # Volatilidad en montos
        'recent_performance': 0.20  # Desempeño reciente (últimos 3 meses)
    })

    # Ventana temporal para análisis reciente
    recent_window_days: int = 90  # Últimos 3 meses

    # Clasificación de riesgo
    risk_levels: Dict[str, tuple] = field(default_factory=lambda: {
        'Bajo': (0.7, 1.0),  # Score >= 0.7
        'Medio': (0.4, 0.7),  # 0.4 <= Score < 0.7
        'Alto': (0.0, 0.4)  # Score < 0.4
    })

    # Umbrales de alerta
    alert_threshold_high_risk: float = 0.3
    alert_threshold_medium_risk: float = 0.5

    # Ranking
    top_n_providers: int = 20  # Top N seccion_C de mayor/menor riesgo


# ============================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================

@dataclass
class LoggingConfig:
    """Configuración de logging."""

    log_level: str = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file: str = str(OUTPUT_DIR / 'model_pipeline.log')
    console_output: bool = True


# ============================================================
# INSTANCIAS GLOBALES DE CONFIGURACIÓN
# ============================================================

# Configuración de limpieza
CLEANING_CONFIG = DataCleaningConfig()

# Configuración de preprocesamiento
PREPROCESSING_CONFIG = PreprocessingConfig()

# Configuración general de modelos
MODEL_CONFIG = ModelConfig()

# Configuración específica por modelo
LOGISTIC_CONFIG = LogisticRegressionConfig()
RANDOM_FOREST_CONFIG = RandomForestConfig()
XGBOOST_CONFIG = XGBoostConfig()

# Configuración de evaluación
EVALUATION_CONFIG = EvaluationConfig()

# Configuración de análisis temporal
TEMPORAL_CONFIG = TemporalAnalysisConfig()

# Configuración de risk scoring
RISK_SCORING_CONFIG = RiskScoringConfig()

# Configuración de logging
LOGGING_CONFIG = LoggingConfig()


# ============================================================
# FUNCIONES HELPER
# ============================================================

def get_model_params(model_name: str, use_default: bool = True) -> Dict[str, Any]:
    """
    Obtiene parámetros de un modelo específico.

    Args:
        model_name: Nombre del modelo ('logistic_regression', 'random_forest', 'xgboost')
        use_default: Si usar parámetros por defecto o grid de búsqueda

    Returns:
        Diccionario de parámetros
    """
    config_map = {
        'logistic_regression': LOGISTIC_CONFIG,
        'random_forest': RANDOM_FOREST_CONFIG,
        'xgboost': XGBOOST_CONFIG
    }

    if model_name not in config_map:
        raise ValueError(f"Modelo '{model_name}' no reconocido")

    config = config_map[model_name]
    return config.default_params if use_default else config.param_grid


def update_config(config_name: str, **kwargs) -> None:
    """
    Actualiza valores de configuración dinámicamente.

    Args:
        config_name: Nombre de la configuración a actualizar
        **kwargs: Pares clave-valor a actualizar
    """
    config_map = {
        'cleaning': CLEANING_CONFIG,
        'preprocessing': PREPROCESSING_CONFIG,
        'model': MODEL_CONFIG,
        'evaluation': EVALUATION_CONFIG,
        'temporal': TEMPORAL_CONFIG,
        'risk_scoring': RISK_SCORING_CONFIG,
        'logging': LOGGING_CONFIG
    }

    if config_name not in config_map:
        raise ValueError(f"Configuración '{config_name}' no reconocida")

    config = config_map[config_name]

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Atributo '{key}' no existe en configuración '{config_name}'")


def print_all_configs() -> None:
    """Imprime todas las configuraciones actuales."""

    configs = {
        'Data Cleaning': CLEANING_CONFIG,
        'Preprocessing': PREPROCESSING_CONFIG,
        'Model': MODEL_CONFIG,
        'Logistic Regression': LOGISTIC_CONFIG,
        'Random Forest': RANDOM_FOREST_CONFIG,
        'XGBoost': XGBOOST_CONFIG,
        'Evaluation': EVALUATION_CONFIG,
        'Temporal Analysis': TEMPORAL_CONFIG,
        'Risk Scoring': RISK_SCORING_CONFIG,
        'Logging': LOGGING_CONFIG
    }

    print("\n" + "=" * 80)
    print("CONFIGURACIONES DEL SISTEMA")
    print("=" * 80)

    for name, config in configs.items():
        print(f"\n{'=' * 40}")
        print(f"{name} Configuration")
        print('=' * 40)

        for field_name, field_value in config.__dict__.items():
            if isinstance(field_value, dict) and len(str(field_value)) > 100:
                print(f"{field_name}: <dict with {len(field_value)} items>")
            elif isinstance(field_value, list) and len(field_value) > 10:
                print(f"{field_name}: <list with {len(field_value)} items>")
            else:
                print(f"{field_name}: {field_value}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Test de configuraciones
    print_all_configs()