"""
Pipeline Orquestador Principal - Modelo Predictivo de Riesgo de Mora

Script que ejecuta el flujo completo:
1. Limpieza de datos
2. Preprocesamiento y feature engineering
3. Entrenamiento de modelos
4. Generación de reportes

Uso:
    python main_pipeline.py [--optimize] [--input PATH] [--output PATH]
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import logging

# Agregar src al path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.data_cleaning import load_and_clean_data
from models.data_preprocessing import preprocess_and_split_data
from models.model_training import train_models_pipeline
from models.config import (
    RAW_DATA_PATH,
    CLEANED_DATA_PATH,
    OUTPUT_DIR,
    MODELS_DIR,
    REPORTS_DIR
)


class PipelineOrchestrator:
    """
    Orquestador principal del pipeline de modelado.

    Ejecuta secuencialmente todos los pasos del proceso.
    """

    def __init__(
        self,
        input_path: str = None,
        output_dir: str = None,
        optimize_hyperparameters: bool = False
    ):
        """
        Inicializa el orquestador.

        Args:
            input_path: Ruta del archivo CSV crudo
            output_dir: Directorio de salida
            optimize_hyperparameters: Si optimizar hiperparámetros (más lento)
        """
        self.input_path = input_path or RAW_DATA_PATH
        self.output_dir = Path(output_dir or OUTPUT_DIR)
        self.optimize_hyperparameters = optimize_hyperparameters

        # Configurar logging
        self._setup_logging()

        self.logger.info("="*80)
        self.logger.info("PIPELINE ORQUESTADOR - MODELO PREDICTIVO DE RIESGO DE MORA")
        self.logger.info("="*80)
        self.logger.info(f"Input: {self.input_path}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info(f"Optimización HP: {self.optimize_hyperparameters}")
        self.logger.info("="*80)

    def _setup_logging(self):
        """Configura logging del pipeline."""
        log_file = self.output_dir / 'pipeline_execution.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)

    def run_full_pipeline(self):
        """
        Ejecuta el pipeline completo.

        Returns:
            Dict con resultados de cada etapa
        """
        start_time = datetime.now()

        results = {
            'start_time': start_time.isoformat(),
            'status': 'running'
        }

        try:
            # PASO 1: Limpieza de datos
            self.logger.info("\n" + "="*60)
            self.logger.info("PASO 1/3: LIMPIEZA DE DATOS")
            self.logger.info("="*60)

            df_cleaned, cleaning_report = self._step_1_clean_data()
            results['cleaning'] = {
                'status': 'success',
                'records': len(df_cleaned),
                'report': cleaning_report
            }

            # PASO 2: Preprocesamiento
            self.logger.info("\n" + "="*60)
            self.logger.info("PASO 2/3: PREPROCESAMIENTO Y FEATURE ENGINEERING")
            self.logger.info("="*60)

            train_df, test_df, preprocessor = self._step_2_preprocess_data()
            results['preprocessing'] = {
                'status': 'success',
                'train_size': len(train_df),
                'test_size': len(test_df),
                'features_count': len(preprocessor.get_feature_names())
            }

            # PASO 3: Entrenamiento de modelos
            self.logger.info("\n" + "="*60)
            self.logger.info("PASO 3/3: ENTRENAMIENTO DE MODELOS")
            self.logger.info("="*60)

            trainer, training_results = self._step_3_train_models()
            results['training'] = {
                'status': 'success',
                'models_trained': list(training_results.keys()),
                'results': training_results
            }

            # Finalización
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            results['end_time'] = end_time.isoformat()
            results['duration_seconds'] = duration
            results['status'] = 'success'

            self._print_final_summary(results, duration)

            # Guardar resultados
            self._save_pipeline_results(results)

            return results

        except Exception as e:
            self.logger.error(f"\n❌ ERROR EN PIPELINE: {str(e)}")
            results['status'] = 'error'
            results['error'] = str(e)

            import traceback
            self.logger.error(traceback.format_exc())

            raise

    def _step_1_clean_data(self):
        """
        Paso 1: Limpieza de datos.

        Returns:
            Tuple con (DataFrame limpio, reporte)
        """
        self.logger.info(f"Cargando datos desde: {self.input_path}")

        df_cleaned, report = load_and_clean_data(
            input_path=str(self.input_path),
            output_path=str(CLEANED_DATA_PATH),
            outlier_threshold=1.5
        )

        self.logger.info(f"✓ Datos limpios guardados en: {CLEANED_DATA_PATH}")

        return df_cleaned, report

    def _step_2_preprocess_data(self):
        """
        Paso 2: Preprocesamiento y feature engineering.

        Returns:
            Tuple con (train_df, test_df, preprocessor)
        """
        self.logger.info("Ejecutando preprocesamiento...")

        train_df, test_df, preprocessor = preprocess_and_split_data(
            input_path=str(CLEANED_DATA_PATH),
            export_files=True
        )

        self.logger.info(f"✓ Train data: {len(train_df)} registros")
        self.logger.info(f"✓ Test data: {len(test_df)} registros")
        self.logger.info(f"✓ Features: {len(preprocessor.get_feature_names())}")

        return train_df, test_df, preprocessor

    def _step_3_train_models(self):
        """
        Paso 3: Entrenamiento de modelos.

        Returns:
            Tuple con (trainer, resultados)
        """
        self.logger.info("Entrenando modelos...")

        if self.optimize_hyperparameters:
            self.logger.warning(
                "⚠️  Optimización de hiperparámetros activada - "
                "esto puede tomar 30-60 minutos"
            )
        else:
            self.logger.info(
                "ℹ️  Usando parámetros por defecto (más rápido)"
            )

        trainer, results = train_models_pipeline(
            optimize_hyperparameters=self.optimize_hyperparameters,
            save_models=True
        )

        self.logger.info(f"✓ Modelos entrenados: {list(results.keys())}")
        self.logger.info(f"✓ Modelos guardados en: {MODELS_DIR}")

        return trainer, results

    def _print_final_summary(self, results: dict, duration: float):
        """
        Imprime resumen final del pipeline.

        Args:
            results: Resultados del pipeline
            duration: Duración en segundos
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("🎉 PIPELINE COMPLETADO EXITOSAMENTE")
        self.logger.info("="*80)

        self.logger.info(f"\n⏱️  Duración total: {duration:.2f} segundos ({duration/60:.2f} minutos)")

        # Resumen de limpieza
        if 'cleaning' in results:
            cleaning = results['cleaning']
            self.logger.info(f"\n📊 LIMPIEZA:")
            self.logger.info(f"  Registros procesados: {cleaning['records']:,}")
            if 'report' in cleaning:
                report = cleaning['report']
                if 'total_outliers' in report:
                    self.logger.info(f"  Outliers detectados: {report['total_outliers']}")

        # Resumen de preprocesamiento
        if 'preprocessing' in results:
            prep = results['preprocessing']
            self.logger.info(f"\n🔧 PREPROCESAMIENTO:")
            self.logger.info(f"  Train: {prep['train_size']:,} registros")
            self.logger.info(f"  Test: {prep['test_size']:,} registros")
            self.logger.info(f"  Features: {prep['features_count']}")

        # Resumen de modelos
        if 'training' in results:
            training = results['training']
            self.logger.info(f"\n🤖 MODELOS ENTRENADOS:")

            for model_name, model_results in training['results'].items():
                if 'metrics' in model_results:
                    test_f1 = model_results['metrics']['test']['f1']
                    test_acc = model_results['metrics']['test']['accuracy']
                    self.logger.info(
                        f"  {model_name}: "
                        f"Accuracy={test_acc:.2%}, F1={test_f1:.2%}"
                    )

        self.logger.info(f"\n📁 Archivos generados en: {self.output_dir}")
        self.logger.info("="*80)

    def _save_pipeline_results(self, results: dict):
        """
        Guarda resultados del pipeline en JSON.

        Args:
            results: Diccionario con resultados
        """
        import json

        output_file = REPORTS_DIR / f'pipeline_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        # Convertir a formato serializable
        serializable_results = self._make_serializable(results)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"✓ Resultados guardados en: {output_file}")

    def _make_serializable(self, obj):
        """Convierte objetos a formato JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj


def main():
    """Función principal con argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Pipeline completo de modelo predictivo de riesgo de mora'
    )

    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Ruta del archivo CSV de entrada (default: resource/historicos_pagos.csv)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Directorio de salida (default: output/)'
    )

    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Optimizar hiperparámetros con GridSearchCV (más lento, 30-60 min)'
    )

    args = parser.parse_args()

    # Crear y ejecutar orquestador
    orchestrator = PipelineOrchestrator(
        input_path=args.input,
        output_dir=args.output,
        optimize_hyperparameters=args.optimize
    )

    results = orchestrator.run_full_pipeline()

    # Retornar código de salida
    return 0 if results['status'] == 'success' else 1


if __name__ == "__main__":
    sys.exit(main())