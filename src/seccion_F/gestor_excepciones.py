"""
Módulo Gestor de Excepciones - Sección F
Clasificación automática de casos especiales
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, Tuple

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESOURCE_DIR = os.path.join(BASE_DIR, 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def cargar_excepciones(archivo: str = 'excepciones_casos_especiales.csv') -> pd.DataFrame:
    """
    Carga el archivo CSV de excepciones y realiza limpieza básica.
    
    Args:
        archivo: Nombre del archivo CSV
        
    Returns:
        DataFrame con los datos limpios
    """
    ruta = os.path.join(RESOURCE_DIR, archivo)
    df = pd.read_csv(ruta)
    
    # Convertir fecha a datetime
    df['fecha_incidencia'] = pd.to_datetime(df['fecha_incidencia'])
    
    # Limpiar espacios en columnas de texto
    columnas_texto = ['tipo_caso', 'nivel_complejidad', 'estado', 'responsable']
    for col in columnas_texto:
        if col in df.columns:
            df[col] = df[col].str.strip()
    
    # Calcular antigüedad del caso en días
    df['dias_antiguedad'] = (datetime.now() - df['fecha_incidencia']).dt.days
    
    return df


def calcular_score_prioridad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula un score de prioridad compuesto para cada caso.
    Score más alto = mayor prioridad de atención.
    
    Componentes:
    - Nivel de complejidad (40%)
    - Monto involucrado normalizado (30%)
    - Antigüedad del caso (20%)
    - Estado del caso (10%)
    
    Args:
        df: DataFrame con datos de excepciones
        
    Returns:
        DataFrame con columna score_prioridad agregada
    """
    df = df.copy()
    
    # Score por complejidad (1-3)
    mapa_complejidad = {'Bajo': 1, 'Medio': 2, 'Alto': 3}
    df['score_complejidad'] = df['nivel_complejidad'].map(mapa_complejidad)
    
    # Score por monto (normalizado 0-3)
    if df['monto_involucrado'].max() > 0:
        df['score_monto'] = (df['monto_involucrado'] / df['monto_involucrado'].max()) * 3
    else:
        df['score_monto'] = 0
    
    # Score por antigüedad (normalizado 0-3, más antiguo = más prioridad)
    if df['dias_antiguedad'].max() > 0:
        df['score_antiguedad'] = (df['dias_antiguedad'] / df['dias_antiguedad'].max()) * 3
    else:
        df['score_antiguedad'] = 0
    
    # Score por estado (Abierto=3, En Proceso=2, Cerrado=1)
    mapa_estado = {'Abierto': 3, 'En Proceso': 2, 'Cerrado': 1}
    df['score_estado'] = df['estado'].map(mapa_estado)
    
    # Score compuesto ponderado
    df['score_prioridad'] = (
        df['score_complejidad'] * 0.40 +
        df['score_monto'] * 0.30 +
        df['score_antiguedad'] * 0.20 +
        df['score_estado'] * 0.10
    )
    
    # Normalizar a escala 0-100
    df['score_prioridad'] = (df['score_prioridad'] / 3) * 100
    
    return df


def clasificar_urgencia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera etiquetas de urgencia automáticas basadas en el score de prioridad.
    
    Categorías:
    - Crítico: score >= 75
    - Alto: score >= 55
    - Medio: score >= 35
    - Bajo: score < 35
    
    Args:
        df: DataFrame con score_prioridad
        
    Returns:
        DataFrame con columna nivel_urgencia agregada
    """
    df = df.copy()
    
    condiciones = [
        (df['score_prioridad'] >= 75),
        (df['score_prioridad'] >= 55),
        (df['score_prioridad'] >= 35),
    ]
    etiquetas = ['Crítico', 'Alto', 'Medio']
    
    df['nivel_urgencia'] = np.select(condiciones, etiquetas, default='Bajo')
    
    return df


def segmentar_casos_riesgo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segmenta casos según categorías de riesgo combinando múltiples factores.
    
    Segmentos:
    - Riesgo Crítico: Alta complejidad + Alto monto + Estado Abierto
    - Riesgo Alto: Alta complejidad O Alto monto
    - Riesgo Medio: Complejidad/monto medio
    - Riesgo Bajo: Baja complejidad y bajo monto
    
    Args:
        df: DataFrame con datos de excepciones
        
    Returns:
        DataFrame con columna segmento_riesgo agregada
    """
    df = df.copy()
    
    # Definir umbrales
    umbral_monto_alto = df['monto_involucrado'].quantile(0.75)
    umbral_monto_medio = df['monto_involucrado'].quantile(0.50)
    
    def asignar_segmento(row):
        es_alta_complejidad = row['nivel_complejidad'] == 'Alto'
        es_monto_alto = row['monto_involucrado'] >= umbral_monto_alto
        es_abierto = row['estado'] == 'Abierto'
        es_monto_medio = row['monto_involucrado'] >= umbral_monto_medio
        es_complejidad_media = row['nivel_complejidad'] == 'Medio'
        
        if es_alta_complejidad and es_monto_alto and es_abierto:
            return 'Riesgo Crítico'
        elif es_alta_complejidad or es_monto_alto:
            return 'Riesgo Alto'
        elif es_complejidad_media or es_monto_medio:
            return 'Riesgo Medio'
        else:
            return 'Riesgo Bajo'
    
    df['segmento_riesgo'] = df.apply(asignar_segmento, axis=1)
    
    return df


def obtener_estadisticas_clasificacion(df: pd.DataFrame) -> Dict:
    """
    Genera estadísticas resumidas de la clasificación de casos.
    
    Args:
        df: DataFrame con clasificaciones aplicadas
        
    Returns:
        Diccionario con estadísticas
    """
    stats = {
        'total_casos': len(df),
        'por_tipo_caso': df['tipo_caso'].value_counts().to_dict(),
        'por_complejidad': df['nivel_complejidad'].value_counts().to_dict(),
        'por_estado': df['estado'].value_counts().to_dict(),
        'por_urgencia': df['nivel_urgencia'].value_counts().to_dict() if 'nivel_urgencia' in df.columns else {},
        'por_segmento': df['segmento_riesgo'].value_counts().to_dict() if 'segmento_riesgo' in df.columns else {},
        'monto_total': df['monto_involucrado'].sum(),
        'monto_promedio': df['monto_involucrado'].mean(),
        'casos_criticos': len(df[df.get('nivel_urgencia', pd.Series([''])*len(df)) == 'Crítico']) if 'nivel_urgencia' in df.columns else 0,
    }
    
    return stats


def ejecutar_clasificacion() -> Tuple[pd.DataFrame, Dict]:
    """
    Ejecuta el proceso completo de clasificación de excepciones.
    
    Returns:
        Tupla con (DataFrame clasificado, estadísticas)
    """
    # Cargar datos
    df = cargar_excepciones()
    
    # Aplicar clasificaciones
    df = calcular_score_prioridad(df)
    df = clasificar_urgencia(df)
    df = segmentar_casos_riesgo(df)
    
    # Ordenar por prioridad
    df = df.sort_values('score_prioridad', ascending=False)
    
    # Obtener estadísticas
    stats = obtener_estadisticas_clasificacion(df)
    
    return df, stats


if __name__ == '__main__':
    df_clasificado, estadisticas = ejecutar_clasificacion()
    
    print("=" * 60)
    print("CLASIFICACIÓN AUTOMÁTICA DE CASOS ESPECIALES")
    print("=" * 60)
    
    print(f"\nTotal de casos: {estadisticas['total_casos']}")
    print(f"Monto total involucrado: ${estadisticas['monto_total']:,.2f}")
    print(f"Monto promedio: ${estadisticas['monto_promedio']:,.2f}")
    
    print("\n--- Distribución por Urgencia ---")
    for urgencia, count in estadisticas['por_urgencia'].items():
        print(f"  {urgencia}: {count}")
    
    print("\n--- Distribución por Segmento de Riesgo ---")
    for segmento, count in estadisticas['por_segmento'].items():
        print(f"  {segmento}: {count}")
    
    print("\n--- Top 10 Casos por Prioridad ---")
    cols_mostrar = ['tipo_caso', 'nivel_complejidad', 'monto_involucrado', 'estado', 'score_prioridad', 'nivel_urgencia']
    print(df_clasificado[cols_mostrar].head(10).to_string(index=False))
