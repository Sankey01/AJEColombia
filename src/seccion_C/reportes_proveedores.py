"""
Módulo de Reportes para Análisis de Proveedores
Genera reportes formateados y visualizaciones básicas
"""

import pandas as pd
from typing import Dict
from analisis_proveedores import (
    ejecutar_analisis_completo,
    exportar_resultados
)


def generar_reporte_segmentacion(df: pd.DataFrame) -> str:
    """
    Genera reporte de segmentación de seccion_C.
    
    Args:
        df: DataFrame con datos procesados
        
    Returns:
        String con reporte formateado
    """
    reporte = []
    reporte.append("=" * 60)
    reporte.append("REPORTE DE SEGMENTACIÓN DE PROVEEDORES")
    reporte.append("=" * 60)
    
    # Distribución por segmento
    reporte.append("\n1. DISTRIBUCIÓN POR SEGMENTO DE RIESGO")
    reporte.append("-" * 40)
    
    for segmento in ['Premium', 'Estándar', 'Supervisión', 'Alto_Riesgo']:
        subset = df[df['segmento_riesgo'] == segmento]
        count = len(subset)
        pct = count / len(df) * 100
        reporte.append(f"  {segmento:15s}: {count:3d} seccion_C ({pct:.1f}%)")
    
    # Distribución cruzada: segmento x tipo empresa
    reporte.append("\n2. MATRIZ SEGMENTO vs TIPO EMPRESA")
    reporte.append("-" * 40)
    
    cross = pd.crosstab(
        df['segmento_riesgo'], 
        df['tipo_empresa'],
        margins=True
    )
    reporte.append(cross.to_string())
    
    # Top seccion_C por segmento
    reporte.append("\n3. TOP 3 PROVEEDORES POR SEGMENTO")
    reporte.append("-" * 40)
    
    for segmento in ['Premium', 'Estándar', 'Supervisión', 'Alto_Riesgo']:
        subset = df[df['segmento_riesgo'] == segmento].nlargest(3, 'score_performance')
        reporte.append(f"\n  {segmento}:")
        for _, row in subset.iterrows():
            reporte.append(
                f"    - {row['nombre']} ({row['tipo_empresa']}, {row['país']}) "
                f"Score: {row['score_performance']:.1f}"
            )
    
    return '\n'.join(reporte)


def generar_reporte_riesgo_categoria(df: pd.DataFrame) -> str:
    """
    Genera reporte de análisis de riesgo por categoría.
    
    Args:
        df: DataFrame con datos procesados
        
    Returns:
        String con reporte formateado
    """
    reporte = []
    reporte.append("=" * 60)
    reporte.append("ANÁLISIS DE RIESGO POR CATEGORÍA")
    reporte.append("=" * 60)
    
    # Análisis por tipo de empresa
    reporte.append("\n1. RIESGO POR TIPO DE EMPRESA")
    reporte.append("-" * 40)
    
    stats = df.groupby('tipo_empresa').agg({
        'score_riesgo': ['mean', 'std', 'min', 'max'],
        'score_performance': 'mean'
    }).round(2)
    
    stats.columns = ['Riesgo_Medio', 'Riesgo_Std', 'Riesgo_Min', 'Riesgo_Max', 'Performance_Medio']
    reporte.append(stats.to_string())
    
    # Análisis por país
    reporte.append("\n\n2. RIESGO POR PAÍS")
    reporte.append("-" * 40)
    
    stats_pais = df.groupby('país').agg({
        'score_riesgo': 'mean',
        'score_performance': 'mean',
        'ID_proveedor': 'count'
    }).round(2)
    
    stats_pais.columns = ['Riesgo_Medio', 'Performance_Medio', 'Num_Proveedores']
    stats_pais = stats_pais.sort_values('Riesgo_Medio', ascending=False)
    reporte.append(stats_pais.to_string())
    
    # Análisis por categoría de riesgo original
    reporte.append("\n\n3. VALIDACIÓN CATEGORÍA ORIGINAL vs SCORE CALCULADO")
    reporte.append("-" * 40)
    
    validacion = df.groupby('categoría_riesgo').agg({
        'score_riesgo': ['mean', 'min', 'max'],
        'ID_proveedor': 'count'
    }).round(2)
    
    validacion.columns = ['Score_Medio', 'Score_Min', 'Score_Max', 'Cantidad']
    reporte.append(validacion.to_string())
    
    # Proveedores mal clasificados (alto riesgo calculado pero bajo original)
    reporte.append("\n\n4. PROVEEDORES CON DISCREPANCIA DE CLASIFICACIÓN")
    reporte.append("-" * 40)
    
    # Proveedores de "Alto" riesgo original pero buen performance
    bueno_en_alto = df[
        (df['categoría_riesgo'] == 'Alto') & 
        (df['nivel_performance'].isin(['Bueno', 'Excelente']))
    ]
    reporte.append(f"\n  Alto riesgo original pero buen performance: {len(bueno_en_alto)}")
    
    # Proveedores de "Bajo" riesgo original pero mal performance
    malo_en_bajo = df[
        (df['categoría_riesgo'] == 'Bajo') & 
        (df['nivel_performance'].isin(['Crítico', 'Regular']))
    ]
    reporte.append(f"  Bajo riesgo original pero mal performance: {len(malo_en_bajo)}")
    
    return '\n'.join(reporte)


def generar_reporte_credito(recomendaciones: pd.DataFrame) -> str:
    """
    Genera reporte de recomendaciones de límites de crédito.
    
    Args:
        recomendaciones: DataFrame con recomendaciones de crédito
        
    Returns:
        String con reporte formateado
    """
    reporte = []
    reporte.append("=" * 60)
    reporte.append("RECOMENDACIONES DE LÍMITES DE CRÉDITO")
    reporte.append("=" * 60)
    
    # Resumen de acciones
    reporte.append("\n1. RESUMEN DE ACCIONES RECOMENDADAS")
    reporte.append("-" * 40)
    
    acciones = recomendaciones['accion_credito'].value_counts()
    for accion, count in acciones.items():
        pct = count / len(recomendaciones) * 100
        reporte.append(f"  {accion:10s}: {count:3d} seccion_C ({pct:.1f}%)")
    
    # Proveedores con mayor variación positiva
    reporte.append("\n2. TOP 10 - AUMENTAR LÍMITE (Mayor oportunidad)")
    reporte.append("-" * 40)
    
    top_aumentar = recomendaciones[
        recomendaciones['accion_credito'] == 'Aumentar'
    ].nlargest(10, 'variacion_credito')
    
    for _, row in top_aumentar.iterrows():
        reporte.append(
            f"  {row['nombre']:15s} | {row['tipo_empresa']:8s} | "
            f"Actual: ${row['límites_crédito']:,.0f} -> "
            f"Recomendado: ${row['limite_recomendado']:,.0f} ({row['variacion_credito']:+.1f}%)"
        )
    
    # Proveedores con mayor variación negativa
    reporte.append("\n3. TOP 10 - REDUCIR LÍMITE (Mayor riesgo)")
    reporte.append("-" * 40)
    
    top_reducir = recomendaciones[
        recomendaciones['accion_credito'] == 'Reducir'
    ].nsmallest(10, 'variacion_credito')
    
    for _, row in top_reducir.iterrows():
        reporte.append(
            f"  {row['nombre']:15s} | {row['tipo_empresa']:8s} | "
            f"Actual: ${row['límites_crédito']:,.0f} -> "
            f"Recomendado: ${row['limite_recomendado']:,.0f} ({row['variacion_credito']:+.1f}%)"
        )
    
    # Resumen financiero
    reporte.append("\n4. IMPACTO FINANCIERO ESTIMADO")
    reporte.append("-" * 40)
    
    credito_actual = recomendaciones['límites_crédito'].sum()
    credito_recomendado = recomendaciones['limite_recomendado'].sum()
    diferencia = credito_recomendado - credito_actual
    
    reporte.append(f"  Crédito total actual:      ${credito_actual:,.0f}")
    reporte.append(f"  Crédito total recomendado: ${credito_recomendado:,.0f}")
    reporte.append(f"  Diferencia:                ${diferencia:+,.0f}")
    
    return '\n'.join(reporte)


def generar_reporte_completo() -> str:
    """
    Genera el reporte completo de análisis de seccion_C.
    
    Returns:
        String con reporte completo
    """
    # Ejecutar análisis
    resultados = ejecutar_analisis_completo()
    
    # Generar reportes
    reporte_segmentacion = generar_reporte_segmentacion(resultados['datos_procesados'])
    reporte_riesgo = generar_reporte_riesgo_categoria(resultados['datos_procesados'])
    reporte_credito = generar_reporte_credito(resultados['recomendaciones_credito'])
    
    # Combinar
    reporte_final = '\n\n'.join([
        reporte_segmentacion,
        reporte_riesgo,
        reporte_credito
    ])
    
    # Exportar CSVs
    exportar_resultados(resultados)
    
    return reporte_final


if __name__ == '__main__':
    reporte = generar_reporte_completo()
    print(reporte)
