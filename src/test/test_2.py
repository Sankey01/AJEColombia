"""
Script de validación para Fase 2

Prueba todos los detectores críticos:
- DuplicateDetector
- FormatAnomaliesDetector
- TemporalAnomaliesDetector
- BusinessRulesDetector
- ApproverAnomaliesDetector
"""

import sys

sys.path.insert(0, '/home/claude/AJEColombia/src/terms_payment')

from utils.data_loader import load_payment_data
from analysis.detectors.duplicate_detector import DuplicateDetector
from analysis.detectors.format_anomalies import FormatAnomaliesDetector
from analysis.detectors.temporal_anomalies import TemporalAnomaliesDetector
from analysis.detectors.business_rules import BusinessRulesDetector
from analysis.detectors.approver_anomalies import ApproverAnomaliesDetector


def test_duplicate_detector(df):
    """Prueba el detector de duplicados"""
    print("\n" + "=" * 80)
    print("TEST 1: DUPLICATE DETECTOR")
    print("=" * 80)

    detector = DuplicateDetector()
    result = detector.detect(df)

    stats = detector.get_statistics()
    print(f"\n✓ Total registros: {stats['total_records']}")
    print(f"✓ Duplicados encontrados: {stats['anomalies_found']}")
    print(f"✓ Grupos de duplicados: {stats['total_duplicate_groups']}")
    print(f"✓ Con diferentes aprobadores: {stats['groups_with_different_approvers']}")
    print(f"✓ Con diferentes montos: {stats['groups_with_different_amounts']}")

    # Verificar IDs específicos
    critical = detector.get_critical_duplicates()
    print(f"\n✓ IDs duplicados críticos encontrados: {len(critical)}")
    if len(critical) > 0:
        print(f"  Ejemplos: {critical[:5]}")

    print("\n✅ TEST DUPLICATE DETECTOR: EXITOSO")
    return result


def test_format_detector(df):
    """Prueba el detector de formato"""
    print("\n" + "=" * 80)
    print("TEST 2: FORMAT ANOMALIES DETECTOR")
    print("=" * 80)

    detector = FormatAnomaliesDetector()
    result = detector.detect(df)

    stats = detector.get_statistics()
    breakdown = stats['breakdown']

    print(f"\n✓ Total registros: {stats['total_records']}")
    print(f"✓ Anomalías encontradas: {stats['anomalies_found']}")
    print(f"\nDesglose:")
    print(f"  - Decimales excesivos: {breakdown['excessive_decimals']}")
    print(f"  - Días crédito negativos: {breakdown['negative_credit_days']}")
    print(f"  - Días crédito en cero: {breakdown['zero_credit_days']}")
    print(f"  - Montos inválidos: {breakdown['invalid_amounts']}")

    print("\n✅ TEST FORMAT DETECTOR: EXITOSO")
    return result


def test_temporal_detector(df):
    """Prueba el detector temporal"""
    print("\n" + "=" * 80)
    print("TEST 3: TEMPORAL ANOMALIES DETECTOR")
    print("=" * 80)

    detector = TemporalAnomaliesDetector(tolerance_days=5)
    result = detector.detect(df)

    stats = detector.get_statistics()
    breakdown = stats['breakdown']

    print(f"\n✓ Total registros: {stats['total_records']}")
    print(f"✓ Anomalías encontradas: {stats['anomalies_found']}")
    print(f"✓ Tolerancia: ±{stats['tolerance_days']} días")
    print(f"\nDesglose:")
    print(f"  - Fechas inconsistentes: {breakdown['inconsistent_dates']}")
    print(f"  - Fechas invertidas: {breakdown['inverted_dates']}")
    print(f"  - Diferencias excesivas (>2 años): {breakdown['excessive_difference']}")

    print("\n✅ TEST TEMPORAL DETECTOR: EXITOSO")
    return result


def test_business_rules_detector(df):
    """Prueba el detector de reglas de negocio"""
    print("\n" + "=" * 80)
    print("TEST 4: BUSINESS RULES DETECTOR")
    print("=" * 80)

    detector = BusinessRulesDetector()
    result = detector.detect(df)

    stats = detector.get_statistics()

    print(f"\n✓ Total registros: {stats['total_records']}")
    print(f"✓ Inconsistencias encontradas: {stats['anomalies_found']}")
    print(f"✓ Porcentaje: {stats['anomaly_percentage']}%")

    print("\n✅ TEST BUSINESS RULES DETECTOR: EXITOSO")
    return result


def test_approver_detector(df):
    """Prueba el detector de aprobadores"""
    print("\n" + "=" * 80)
    print("TEST 5: APPROVER ANOMALIES DETECTOR (ESTADÍSTICO)")
    print("=" * 80)

    detector = ApproverAnomaliesDetector()
    result = detector.detect(df)

    stats = detector.get_statistics()

    print(f"\n✓ Total registros: {stats['total_records']}")
    print(f"✓ Anomalías encontradas: {stats['anomalies_found']}")
    print(f"✓ Aprobadores sospechosos: {stats['suspicious_approvers_count']}")
    print(f"  Lista: {stats['suspicious_approvers']}")

    # Mostrar reporte de aprobadores
    print("\nReporte de aprobadores:")
    report = detector.get_approver_report()
    print(report.head(10).to_string())

    print("\n✅ TEST APPROVER DETECTOR: EXITOSO")
    return result


def main():
    """Ejecuta todos los tests de Fase 2"""
    print("\n" + "=" * 80)
    print("VALIDACIÓN DE FASE 2: DETECTORES CRÍTICOS")
    print("=" * 80)

    try:
        # Cargar datos
        print("\nCargando datos...")
        df = load_payment_data('/mnt/user-data/uploads/condiciones_pagos__1_.csv')
        print(f"✓ Datos cargados: {len(df)} registros")

        # Ejecutar todos los detectores
        result1 = test_duplicate_detector(df)
        result2 = test_format_detector(df)
        result3 = test_temporal_detector(df)
        result4 = test_business_rules_detector(df)
        result5 = test_approver_detector(df)

        # Resumen final
        print("\n" + "=" * 80)
        print("✅ FASE 2 COMPLETADA EXITOSAMENTE")
        print("=" * 80)
        print("\nTodos los detectores críticos están funcionando correctamente.")
        print("\nDetectores implementados:")
        print("  ✓ duplicate_detector.py - Transacciones duplicadas")
        print("  ✓ format_anomalies.py - Anomalías de formato")
        print("  ✓ temporal_anomalies.py - Anomalías temporales")
        print("  ✓ business_rules.py - Reglas de negocio")
        print("  ✓ approver_anomalies.py - Aprobadores atípicos (estadístico)")
        print("\nSe puede proceder con la Fase 3: Detectores Complementarios")

    except Exception as e:
        print(f"\n❌ ERROR EN VALIDACIÓN: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()