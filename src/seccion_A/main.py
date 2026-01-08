"""
Script de Test Final - Integración Completa

Prueba el orquestador principal que integra todos los detectores.
Este es el script final que valida todo el sistema de detección de anomalías.
"""

from src.utils.data_loader import load_payment_data
from src.seccion_A.analysis.anomaly_detector import AnomalyDetectorOrchestrator


def main():
    """Test completo del sistema de detección de anomalías"""
    print("\n" + "="*80)
    print("TEST FINAL: SISTEMA COMPLETO DE DETECCIÓN DE ANOMALÍAS")
    print("="*80)

    try:
        # Paso 1: Cargar datos (usa ruta por defecto de Windows)
        print("\n[1/4] Cargando datos...")
        df = load_payment_data()  # Usa ruta por defecto
        print(f"✓ Datos cargados: {len(df)} registros")

        # Paso 2: Inicializar orquestador
        print("\n[2/4] Inicializando orquestador...")
        orchestrator = AnomalyDetectorOrchestrator()
        print(f"✓ Orquestador inicializado con {len(orchestrator.detectors)} detectores")

        # Paso 3: Ejecutar detección
        print("\n[3/4] Ejecutando detección de anomalías...")
        result = orchestrator.detect_all(df)
        print(f"✓ Detección completada")
        print(f"✓ DataFrame consolidado: {len(result)} registros, {len(result.columns)} columnas")

        # Paso 4: Generar resumen
        print("\n[4/4] Generando resumen...")
        summary = orchestrator.get_summary()
        orchestrator.print_summary()

        # Mostrar top 10 registros con más anomalías
        print("\n" + "="*80)
        print("TOP 10 REGISTROS CON MÁS ANOMALÍAS")
        print("="*80)
        top10 = orchestrator.get_top_anomalies(10)
        print(top10[['ID_transaccion', 'proveedor_id', 'monto', 'aprobador', 'total_anomalies']].to_string())

        # Exportar resultados (ruta de Windows)
        print("\n" + "="*80)
        print("EXPORTANDO RESULTADOS")
        print("="*80)
        output_dir = r'/output/anomaly_detection_results'
        orchestrator.export_results(
            output_dir=output_dir,
            export_csv=True,
            export_json=True,
            export_summary=True
        )

        # Resumen final
        print("\n" + "="*80)
        print("✅ TEST FINAL COMPLETADO EXITOSAMENTE")
        print("="*80)
        print("\n📊 Resumen ejecutivo:")
        print(f"  Total registros: {summary['total_records']:,}")
        print(f"  Con anomalías: {summary['records_with_anomalies']:,} ({summary['anomaly_percentage']}%)")
        print(f"  Limpios: {summary['records_clean']:,} ({summary['clean_percentage']}%)")
        print(f"\n📁 Archivos generados en: {output_dir}")
        print("  ✓ anomalies_full.csv - DataFrame completo con todas las columnas")
        print("  ✓ anomalies_only.csv - Solo registros con anomalías")
        print("  ✓ statistics.json - Estadísticas detalladas en JSON")
        print("  ✓ summary.txt - Resumen ejecutivo en texto")
        print("\n🎯 Sistema listo para producción")

    except Exception as e:
        print(f"\n❌ ERROR EN TEST FINAL: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()