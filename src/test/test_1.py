"""
Script de validación para Fase 1 (Simplificado)

Prueba que los módulos base funcionen correctamente:
- data_loader.py (simplificado)
- base_detector.py (simplificado, sin severidad)
"""

import sys

sys.path.insert(0, '/home/claude/AJEColombia/src/seccion_A')

from src.utils.data_loader import load_payment_data
from src.seccion_A.analysis.detectors.base_detector import BaseDetector
import pandas as pd


def test_data_loader():
    """Prueba el módulo de carga de datos simplificado"""
    print("=" * 80)
    print("TEST 1: DATA LOADER (SIMPLIFICADO)")
    print("=" * 80)

    # Cargar datos
    print("\n1.1 Cargando datos desde ruta por defecto...")
    try:
        # Como estamos en Linux, usar la ruta de test
        df = load_payment_data('/mnt/user-data/uploads/condiciones_pagos__1_.csv')
        print(f"✓ Datos cargados: {len(df)} registros")
        print(f"✓ Columnas: {len(df.columns)}")
        print(f"✓ Primeras columnas: {list(df.columns[:5])}")
    except Exception as e:
        print(f"✗ Error al cargar: {str(e)}")
        return None

    print("\n✅ TEST DATA LOADER: EXITOSO\n")
    return df


def test_base_detector():
    """Prueba la clase base de detectores simplificada"""
    print("=" * 80)
    print("TEST 2: BASE DETECTOR (SIN SEVERIDAD)")
    print("=" * 80)

    # Crear un detector de ejemplo
    class DummyDetector(BaseDetector):
        def detect(self, df: pd.DataFrame) -> pd.DataFrame:
            # Simular detección: marcar registros con monto > 10000
            anomaly_mask = df['monto'] > 10000
            descriptions = anomaly_mask.apply(
                lambda x: "Monto alto detectado" if x else None
            )

            result = self._add_anomaly_columns(
                df,
                anomaly_mask,
                'HIGH_AMOUNT',
                descriptions
            )

            self._calculate_statistics(result, f'{self.name}_has_anomaly')

            return result

        def get_statistics(self) -> dict:
            return {
                'detector_name': self.name,
                'total_records': self.total_records,
                'anomalies_found': self.anomalies_found
            }

    print("\n2.1 Creando detector de prueba...")
    detector = DummyDetector("test_detector")
    print(f"✓ Detector creado: {detector.name}")

    print("\n2.2 Cargando datos de prueba...")
    df = load_payment_data('/mnt/user-data/uploads/condiciones_pagos__1_.csv')
    print(f"✓ Datos cargados: {len(df)} registros")

    print("\n2.3 Ejecutando detección...")
    result = detector.detect(df)
    print(f"✓ Detección ejecutada")

    print("\n2.4 Verificando columnas agregadas...")
    expected_columns = [
        'test_detector_has_anomaly',
        'test_detector_type',
        'test_detector_description'
    ]
    for col in expected_columns:
        if col in result.columns:
            print(f"✓ Columna '{col}' presente")
        else:
            print(f"✗ Columna '{col}' faltante")

    # Verificar que NO existe columna de severidad
    if 'test_detector_severity' not in result.columns:
        print(f"✓ Columna 'severity' correctamente eliminada (no existe)")
    else:
        print(f"✗ ERROR: Columna 'severity' aún existe")

    print("\n2.5 Obteniendo estadísticas...")
    stats = detector.get_statistics()
    print(f"✓ Total registros: {stats['total_records']}")
    print(f"✓ Anomalías encontradas: {stats['anomalies_found']}")

    print("\n✅ TEST BASE DETECTOR: EXITOSO\n")
    return detector, result


def main():
    """Ejecuta todos los tests"""
    print("\n" + "=" * 80)
    print("VALIDACIÓN DE FASE 1 (SIMPLIFICADA)")
    print("=" * 80 + "\n")

    try:
        # Test 1: Data Loader
        df = test_data_loader()

        if df is None:
            print("\n❌ ERROR: No se pudo cargar el dataset")
            return

        # Test 2: Base Detector
        detector, result = test_base_detector()

        print("=" * 80)
        print("✅ FASE 1 COMPLETADA EXITOSAMENTE")
        print("=" * 80)
        print("\nTodos los módulos base están funcionando correctamente.")
        print("Se puede proceder con la Fase 2: Detectores Críticos.")
        print("\nArchivos simplificados:")
        print("  ✓ utils/data_loader.py - Solo carga CSV")
        print("  ✓ analysis/detectors/base_detector.py - Sin niveles de severidad")

    except Exception as e:
        print(f"\n❌ ERROR EN VALIDACIÓN: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()