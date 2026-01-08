"""
Módulo de Alertas por Complejidad - Sección F
Sistema de alertas automáticas y monitoreo
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def detectar_casos_criticos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta casos críticos que requieren atención inmediata.
    
    Criterios:
    - Alta complejidad + Alto monto + Estado Abierto
    - Cualquier caso con score de prioridad >= 80
    
    Args:
        df: DataFrame con casos clasificados
        
    Returns:
        DataFrame con casos críticos
    """
    # Calcular percentil 75 del monto
    umbral_monto = df['monto_involucrado'].quantile(0.75)
    
    # Filtrar casos críticos
    condicion_critica = (
        ((df['nivel_complejidad'] == 'Alto') & 
         (df['monto_involucrado'] >= umbral_monto) & 
         (df['estado'] == 'Abierto')) |
        (df.get('score_prioridad', pd.Series([0]*len(df))) >= 80)
    )
    
    casos_criticos = df[condicion_critica].copy()
    casos_criticos['razon_critico'] = 'Alta complejidad + Alto monto + Abierto'
    
    return casos_criticos


def detectar_casos_sla_vencido(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica casos con SLA vencido o próximo a vencer.
    
    Args:
        df: DataFrame con información de SLA
        
    Returns:
        DataFrame con casos en riesgo de SLA
    """
    if 'estado_sla' not in df.columns:
        return pd.DataFrame()
    
    casos_riesgo = df[df['estado_sla'].isin(['Vencido', 'Por Vencer'])].copy()
    
    return casos_riesgo


def calcular_carga_responsable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la carga de trabajo por responsable.
    
    Args:
        df: DataFrame con casos
        
    Returns:
        DataFrame con métricas por responsable
    """
    metricas = []
    
    for responsable in df['responsable'].unique():
        casos_resp = df[df['responsable'] == responsable]
        
        metrica = {
            'responsable': responsable,
            'total_casos': len(casos_resp),
            'casos_abiertos': len(casos_resp[casos_resp['estado'] == 'Abierto']),
            'casos_en_proceso': len(casos_resp[casos_resp['estado'] == 'En Proceso']),
            'casos_cerrados': len(casos_resp[casos_resp['estado'] == 'Cerrado']),
            'alta_complejidad': len(casos_resp[casos_resp['nivel_complejidad'] == 'Alto']),
            'monto_total': casos_resp['monto_involucrado'].sum(),
            'promedio_antiguedad': casos_resp['dias_antiguedad'].mean() if 'dias_antiguedad' in casos_resp.columns else 0,
        }
        
        # Calcular índice de carga (normalizado)
        metrica['indice_carga'] = (
            metrica['casos_abiertos'] * 3 + 
            metrica['casos_en_proceso'] * 2 + 
            metrica['alta_complejidad'] * 2
        )
        
        metricas.append(metrica)
    
    df_metricas = pd.DataFrame(metricas)
    df_metricas = df_metricas.sort_values('indice_carga', ascending=False)
    
    return df_metricas


def generar_alertas_responsable(df_carga: pd.DataFrame) -> List[Dict]:
    """
    Genera alertas basadas en la carga de trabajo de responsables.
    
    Args:
        df_carga: DataFrame con métricas por responsable
        
    Returns:
        Lista de alertas
    """
    alertas = []
    
    # Umbral de sobrecarga
    umbral_casos = df_carga['total_casos'].mean() * 1.5
    umbral_abiertos = df_carga['casos_abiertos'].mean() * 1.5
    
    for _, row in df_carga.iterrows():
        if row['total_casos'] > umbral_casos:
            alertas.append({
                'tipo': 'SOBRECARGA',
                'severidad': 'Alta',
                'responsable': row['responsable'],
                'mensaje': f"{row['responsable']} tiene {row['total_casos']} casos asignados (umbral: {umbral_casos:.0f})",
                'accion_sugerida': 'Redistribuir casos a otros responsables',
            })
        
        if row['casos_abiertos'] > umbral_abiertos:
            alertas.append({
                'tipo': 'ACUMULACIÓN',
                'severidad': 'Media',
                'responsable': row['responsable'],
                'mensaje': f"{row['responsable']} tiene {row['casos_abiertos']} casos abiertos sin procesar",
                'accion_sugerida': 'Priorizar cierre de casos pendientes',
            })
        
        if row['alta_complejidad'] > 5:
            alertas.append({
                'tipo': 'CONCENTRACIÓN_RIESGO',
                'severidad': 'Alta',
                'responsable': row['responsable'],
                'mensaje': f"{row['responsable']} tiene {row['alta_complejidad']} casos de alta complejidad",
                'accion_sugerida': 'Escalar casos complejos o asignar apoyo',
            })
    
    return alertas


def detectar_patrones_anomalos(df: pd.DataFrame) -> List[Dict]:
    """
    Detecta patrones anómalos en la distribución de casos.
    
    Args:
        df: DataFrame con casos
        
    Returns:
        Lista de anomalías detectadas
    """
    anomalias = []
    
    # Concentración por tipo de caso
    distribucion_tipo = df['tipo_caso'].value_counts()
    promedio_tipo = distribucion_tipo.mean()
    
    for tipo, count in distribucion_tipo.items():
        if count > promedio_tipo * 2:
            anomalias.append({
                'tipo': 'CONCENTRACIÓN_TIPO',
                'descripcion': f"Alto volumen de casos '{tipo}': {count} casos",
                'valor_normal': f"{promedio_tipo:.0f} casos promedio",
                'recomendacion': 'Revisar proceso fuente de este tipo de caso',
            })
    
    # Acumulación de casos abiertos
    ratio_abiertos = len(df[df['estado'] == 'Abierto']) / len(df)
    if ratio_abiertos > 0.4:
        anomalias.append({
            'tipo': 'ACUMULACIÓN_ABIERTOS',
            'descripcion': f"El {ratio_abiertos*100:.1f}% de casos están abiertos",
            'valor_normal': 'Menos del 40% abiertos',
            'recomendacion': 'Aumentar capacidad de procesamiento',
        })
    
    # Montos inusualmente altos
    umbral_monto = df['monto_involucrado'].quantile(0.95)
    casos_alto_monto = len(df[df['monto_involucrado'] >= umbral_monto])
    if casos_alto_monto > len(df) * 0.1:
        anomalias.append({
            'tipo': 'CONCENTRACIÓN_MONTO',
            'descripcion': f"{casos_alto_monto} casos con monto >= ${umbral_monto:,.2f}",
            'valor_normal': 'Máximo 5% de casos con montos muy altos',
            'recomendacion': 'Revisar umbrales de escalación',
        })
    
    return anomalias


def generar_reporte_alertas(df: pd.DataFrame) -> Dict:
    """
    Genera un reporte completo de alertas.
    
    Args:
        df: DataFrame con casos clasificados
        
    Returns:
        Diccionario con todas las alertas
    """
    # Calcular métricas de carga
    df_carga = calcular_carga_responsable(df)
    
    reporte = {
        'timestamp': datetime.now().isoformat(),
        'casos_criticos': detectar_casos_criticos(df).to_dict('records'),
        'alertas_responsables': generar_alertas_responsable(df_carga),
        'anomalias': detectar_patrones_anomalos(df),
        'metricas_responsables': df_carga.to_dict('records'),
        'resumen': {
            'total_casos_criticos': len(detectar_casos_criticos(df)),
            'total_alertas': len(generar_alertas_responsable(df_carga)),
            'total_anomalias': len(detectar_patrones_anomalos(df)),
        }
    }
    
    return reporte


def exportar_alertas(df: pd.DataFrame, prefijo: str = 'seccion_f'):
    """
    Exporta alertas a archivos CSV.
    
    Args:
        df: DataFrame con casos
        prefijo: Prefijo para nombres de archivo
    """
    # Exportar casos críticos
    casos_criticos = detectar_casos_criticos(df)
    if len(casos_criticos) > 0:
        ruta_criticos = os.path.join(OUTPUT_DIR, f'{prefijo}_casos_criticos.csv')
        cols_exportar = ['tipo_caso', 'nivel_complejidad', 'monto_involucrado', 
                        'estado', 'responsable', 'fecha_incidencia']
        cols_disponibles = [c for c in cols_exportar if c in casos_criticos.columns]
        casos_criticos[cols_disponibles].to_csv(ruta_criticos, index=False)
        print(f"Casos críticos exportados a: {ruta_criticos}")
    
    # Exportar métricas de responsables
    df_carga = calcular_carga_responsable(df)
    ruta_carga = os.path.join(OUTPUT_DIR, f'{prefijo}_metricas_responsables.csv')
    df_carga.to_csv(ruta_carga, index=False)
    print(f"Métricas de responsables exportadas a: {ruta_carga}")


if __name__ == '__main__':
    from .gestor_excepciones import ejecutar_clasificacion
    
    print("=" * 60)
    print("SISTEMA DE ALERTAS POR COMPLEJIDAD")
    print("=" * 60)
    
    # Cargar casos clasificados
    df_clasificado, _ = ejecutar_clasificacion()
    
    # Generar reporte de alertas
    reporte = generar_reporte_alertas(df_clasificado)
    
    print(f"\n--- Resumen de Alertas ---")
    print(f"Casos críticos: {reporte['resumen']['total_casos_criticos']}")
    print(f"Alertas de responsables: {reporte['resumen']['total_alertas']}")
    print(f"Anomalías detectadas: {reporte['resumen']['total_anomalias']}")
    
    print("\n--- Alertas de Responsables ---")
    for alerta in reporte['alertas_responsables']:
        print(f"  [{alerta['severidad']}] {alerta['tipo']}: {alerta['mensaje']}")
    
    print("\n--- Anomalías Detectadas ---")
    for anomalia in reporte['anomalias']:
        print(f"  - {anomalia['tipo']}: {anomalia['descripcion']}")
    
    print("\n--- Carga por Responsable ---")
    df_carga = calcular_carga_responsable(df_clasificado)
    print(df_carga[['responsable', 'total_casos', 'casos_abiertos', 'indice_carga']].to_string(index=False))
