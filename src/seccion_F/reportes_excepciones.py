"""
Módulo de Reportes de Excepciones - Sección F
Generación de reportes, métricas y entregables finales
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def generar_dashboard_resumen(df: pd.DataFrame) -> Dict:
    """
    Genera un dashboard resumen del estado de excepciones.
    
    Args:
        df: DataFrame con casos clasificados
        
    Returns:
        Diccionario con métricas del dashboard
    """
    dashboard = {
        'fecha_generacion': datetime.now().isoformat(),
        'metricas_generales': {
            'total_casos': len(df),
            'monto_total': df['monto_involucrado'].sum(),
            'monto_promedio': df['monto_involucrado'].mean(),
            'monto_maximo': df['monto_involucrado'].max(),
        },
        'por_estado': df['estado'].value_counts().to_dict(),
        'por_complejidad': df['nivel_complejidad'].value_counts().to_dict(),
        'por_tipo_caso': df['tipo_caso'].value_counts().to_dict(),
        'por_responsable': df['responsable'].value_counts().to_dict(),
    }
    
    if 'nivel_urgencia' in df.columns:
        dashboard['por_urgencia'] = df['nivel_urgencia'].value_counts().to_dict()
    
    if 'segmento_riesgo' in df.columns:
        dashboard['por_segmento_riesgo'] = df['segmento_riesgo'].value_counts().to_dict()
    
    return dashboard


def analizar_tendencias_tipo_caso(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza tendencias por tipo de caso.
    
    Args:
        df: DataFrame con casos
        
    Returns:
        DataFrame con análisis de tendencias
    """
    tendencias = []
    
    for tipo in df['tipo_caso'].unique():
        casos_tipo = df[df['tipo_caso'] == tipo]
        
        tendencia = {
            'tipo_caso': tipo,
            'total_casos': len(casos_tipo),
            'porcentaje': len(casos_tipo) / len(df) * 100,
            'monto_total': casos_tipo['monto_involucrado'].sum(),
            'monto_promedio': casos_tipo['monto_involucrado'].mean(),
            'casos_abiertos': len(casos_tipo[casos_tipo['estado'] == 'Abierto']),
            'casos_alta_complejidad': len(casos_tipo[casos_tipo['nivel_complejidad'] == 'Alto']),
            'tasa_resolucion': len(casos_tipo[casos_tipo['estado'] == 'Cerrado']) / len(casos_tipo) * 100,
        }
        tendencias.append(tendencia)
    
    df_tendencias = pd.DataFrame(tendencias)
    df_tendencias = df_tendencias.sort_values('total_casos', ascending=False)
    
    return df_tendencias


def generar_reporte_responsables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera reporte detallado de carga de trabajo por responsable.
    
    Args:
        df: DataFrame con casos
        
    Returns:
        DataFrame con reporte de responsables
    """
    reporte = []
    
    for responsable in df['responsable'].unique():
        casos_resp = df[df['responsable'] == responsable]
        
        info = {
            'responsable': responsable,
            'total_casos': len(casos_resp),
            'casos_abiertos': len(casos_resp[casos_resp['estado'] == 'Abierto']),
            'casos_en_proceso': len(casos_resp[casos_resp['estado'] == 'En Proceso']),
            'casos_cerrados': len(casos_resp[casos_resp['estado'] == 'Cerrado']),
            'tasa_resolucion': len(casos_resp[casos_resp['estado'] == 'Cerrado']) / len(casos_resp) * 100,
            'monto_gestionado': casos_resp['monto_involucrado'].sum(),
            'complejidad_alta': len(casos_resp[casos_resp['nivel_complejidad'] == 'Alto']),
            'complejidad_media': len(casos_resp[casos_resp['nivel_complejidad'] == 'Medio']),
            'complejidad_baja': len(casos_resp[casos_resp['nivel_complejidad'] == 'Bajo']),
        }
        
        # Calcular score de eficiencia
        info['score_eficiencia'] = (
            info['tasa_resolucion'] * 0.5 +
            (1 - info['complejidad_alta'] / max(info['total_casos'], 1)) * 30 +
            (info['casos_cerrados'] / max(info['total_casos'], 1)) * 20
        )
        
        reporte.append(info)
    
    df_reporte = pd.DataFrame(reporte)
    df_reporte = df_reporte.sort_values('score_eficiencia', ascending=False)
    
    return df_reporte


def exportar_entregables(df_clasificado: pd.DataFrame, 
                          df_flujo: pd.DataFrame = None,
                          prefijo: str = 'seccion_f'):
    """
    Exporta todos los entregables de la Sección F.
    
    Args:
        df_clasificado: DataFrame con casos clasificados
        df_flujo: DataFrame con flujo de trabajo (opcional)
        prefijo: Prefijo para nombres de archivo
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("EXPORTANDO ENTREGABLES SECCIÓN F")
    print("=" * 60)
    
    # 1. Exportar casos clasificados
    ruta_casos = os.path.join(OUTPUT_DIR, f'{prefijo}_casos_clasificados.csv')
    cols_casos = ['tipo_caso', 'nivel_complejidad', 'monto_involucrado', 'estado', 
                  'responsable', 'fecha_incidencia', 'score_prioridad', 'nivel_urgencia', 
                  'segmento_riesgo']
    cols_disponibles = [c for c in cols_casos if c in df_clasificado.columns]
    df_clasificado[cols_disponibles].to_csv(ruta_casos, index=False)
    print(f"[OK] Casos clasificados: {ruta_casos}")
    
    # 2. Exportar flujo de trabajo si está disponible
    if df_flujo is not None:
        ruta_flujo = os.path.join(OUTPUT_DIR, f'{prefijo}_flujo_trabajo.csv')
        df_flujo.to_csv(ruta_flujo, index=False)
        print(f"[OK] Flujo de trabajo: {ruta_flujo}")
    
    # 3. Exportar tendencias por tipo de caso
    df_tendencias = analizar_tendencias_tipo_caso(df_clasificado)
    ruta_tendencias = os.path.join(OUTPUT_DIR, f'{prefijo}_tendencias_tipo.csv')
    df_tendencias.to_csv(ruta_tendencias, index=False)
    print(f"[OK] Tendencias por tipo: {ruta_tendencias}")
    
    # 4. Exportar reporte de responsables
    df_responsables = generar_reporte_responsables(df_clasificado)
    ruta_responsables = os.path.join(OUTPUT_DIR, f'{prefijo}_reporte_responsables.csv')
    df_responsables.to_csv(ruta_responsables, index=False)
    print(f"[OK] Reporte responsables: {ruta_responsables}")
    
    return {
        'casos_clasificados': ruta_casos,
        'tendencias': ruta_tendencias,
        'responsables': ruta_responsables,
    }


def ejecutar_analisis_completo() -> Dict:
    """
    Ejecuta el análisis completo de la Sección F.
    
    Returns:
        Diccionario con todos los resultados
    """
    from .gestor_excepciones import ejecutar_clasificacion
    from .workflow_excepciones import (
        generar_flujo_trabajo, 
        generar_matriz_escalacion,
        exportar_matriz_escalacion,
        proponer_mejoras_proceso,
        obtener_metricas_workflow
    )
    from .alertas_complejidad import (
        generar_reporte_alertas,
        exportar_alertas
    )
    
    print("=" * 60)
    print("SECCIÓN F: ANÁLISIS COMPLETO DE GESTIÓN DE EXCEPCIONES")
    print("=" * 60)
    
    # 1. Clasificación de casos
    print("\n[1/4] Clasificando casos especiales...")
    df_clasificado, stats_clasificacion = ejecutar_clasificacion()
    
    # 2. Flujo de trabajo
    print("[2/4] Generando flujo de trabajo...")
    df_flujo = generar_flujo_trabajo(df_clasificado)
    metricas_workflow = obtener_metricas_workflow(df_flujo)
    
    # 3. Sistema de alertas
    print("[3/4] Generando alertas...")
    reporte_alertas = generar_reporte_alertas(df_clasificado)
    
    # 4. Exportar entregables
    print("[4/4] Exportando entregables...")
    rutas_exportadas = exportar_entregables(df_clasificado, df_flujo)
    
    # Exportar matriz de escalación
    matriz_escalacion = exportar_matriz_escalacion()
    
    # Exportar alertas
    exportar_alertas(df_clasificado)
    
    # Dashboard resumen
    dashboard = generar_dashboard_resumen(df_clasificado)
    
    resultados = {
        'df_clasificado': df_clasificado,
        'df_flujo': df_flujo,
        'matriz_escalacion': matriz_escalacion,
        'dashboard': dashboard,
        'stats_clasificacion': stats_clasificacion,
        'metricas_workflow': metricas_workflow,
        'reporte_alertas': reporte_alertas,
        'mejoras_propuestas': proponer_mejoras_proceso(),
        'rutas_exportadas': rutas_exportadas,
    }
    
    return resultados


def imprimir_resumen_ejecutivo(resultados: Dict):
    """
    Imprime un resumen ejecutivo del análisis.
    
    Args:
        resultados: Diccionario con resultados del análisis
    """
    print("\n" + "=" * 60)
    print("RESUMEN EJECUTIVO - SECCIÓN F")
    print("=" * 60)
    
    dashboard = resultados['dashboard']
    stats = resultados['stats_clasificacion']
    metricas = resultados['metricas_workflow']
    alertas = resultados['reporte_alertas']
    
    print(f"\n[METRICAS GENERALES]")
    print(f"   Total de casos: {dashboard['metricas_generales']['total_casos']}")
    print(f"   Monto total: ${dashboard['metricas_generales']['monto_total']:,.2f}")
    print(f"   Monto promedio: ${dashboard['metricas_generales']['monto_promedio']:,.2f}")
    
    print(f"\n[DISTRIBUCION POR ESTADO]")
    for estado, count in dashboard['por_estado'].items():
        porcentaje = count / dashboard['metricas_generales']['total_casos'] * 100
        print(f"   {estado}: {count} ({porcentaje:.1f}%)")
    
    print(f"\n[NIVELES DE URGENCIA]")
    if 'por_urgencia' in dashboard:
        for urgencia, count in dashboard['por_urgencia'].items():
            print(f"   {urgencia}: {count}")
    
    print(f"\n[ALERTAS]")
    print(f"   Casos criticos: {alertas['resumen']['total_casos_criticos']}")
    print(f"   Alertas activas: {alertas['resumen']['total_alertas']}")
    print(f"   Anomalias detectadas: {alertas['resumen']['total_anomalias']}")
    
    print(f"\n[CUMPLIMIENTO SLA]")
    print(f"   Casos vencidos: {metricas['casos_vencidos']}")
    print(f"   Por vencer: {metricas['casos_por_vencer']}")
    print(f"   En tiempo: {metricas['casos_en_tiempo']}")
    
    print(f"\n[TOP 3 MEJORAS PROPUESTAS]")
    for mejora in resultados['mejoras_propuestas'][:3]:
        print(f"   [{mejora['prioridad']}] {mejora['propuesta']}")
    
    print("\n" + "=" * 60)
    print("Archivos generados en: " + OUTPUT_DIR)
    print("=" * 60)


if __name__ == '__main__':
    resultados = ejecutar_analisis_completo()
    imprimir_resumen_ejecutivo(resultados)
