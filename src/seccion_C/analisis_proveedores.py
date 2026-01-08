"""
Módulo de Análisis de Proveedores y Riesgo
Sección C - Evaluación de riesgos y segmentación de seccion_C
"""

import pandas as pd
import numpy as np
import ast
import os
from typing import Dict, List, Tuple

# Rutas relativas del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESOURCE_DIR = os.path.join(BASE_DIR, 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def cargar_datos_proveedores(archivo: str = 'proveedores_condiciones.csv') -> pd.DataFrame:
    """
    Carga el archivo CSV de seccion_C y realiza limpieza básica.
    
    Args:
        archivo: Nombre del archivo CSV a cargar
        
    Returns:
        DataFrame con los datos limpios
    """
    ruta = os.path.join(RESOURCE_DIR, archivo)
    df = pd.read_csv(ruta)
    
    # Parsear el campo histórico_performance (es un diccionario en string)
    df['performance_dict'] = df['histórico_performance'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else {}
    )
    
    # Extraer métricas de performance
    df['on_time_delivery'] = df['performance_dict'].apply(
        lambda x: int(x.get('on_time_delivery', '0%').replace('%', ''))
    )
    df['quality_issues'] = df['performance_dict'].apply(
        lambda x: x.get('quality_issues', 0)
    )
    df['customer_complaints'] = df['performance_dict'].apply(
        lambda x: x.get('customer_complaints', 0)
    )
    
    # Parsear certificaciones
    df['certificaciones_list'] = df['certificaciones'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else []
    )
    
    # Contar certificaciones válidas (excluyendo 'None')
    df['num_certificaciones'] = df['certificaciones_list'].apply(
        lambda x: len([c for c in x if c != 'None'])
    )
    
    return df


def calcular_score_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula un score compuesto de performance para cada proveedor.
    Score más alto = mejor performance
    
    Args:
        df: DataFrame con datos de seccion_C
        
    Returns:
        DataFrame con columna score_performance agregada
    """
    df = df.copy()
    
    # Normalizar métricas (0-100)
    df['score_entrega'] = df['on_time_delivery']  # Ya está en %
    df['score_calidad'] = 100 - (df['quality_issues'] * 10)  # Penalizar issues
    df['score_quejas'] = 100 - (df['customer_complaints'] * 10)  # Penalizar quejas
    
    # Score compuesto ponderado
    df['score_performance'] = (
        df['score_entrega'] * 0.5 +
        df['score_calidad'] * 0.3 +
        df['score_quejas'] * 0.2
    )
    
    # Clasificar performance
    df['nivel_performance'] = pd.cut(
        df['score_performance'],
        bins=[0, 60, 75, 90, 100],
        labels=['Crítico', 'Regular', 'Bueno', 'Excelente']
    )
    
    return df


def segmentar_proveedores_riesgo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segmenta seccion_C calculando un score de riesgo compuesto.
    Considera: categoría actual, performance, tipo empresa, certificaciones.
    
    Args:
        df: DataFrame con datos de seccion_C
        
    Returns:
        DataFrame con segmentación de riesgo
    """
    df = df.copy()
    
    # Mapeo de categoría de riesgo a valores numéricos
    riesgo_map = {'Alto': 3, 'Medio': 2, 'Bajo': 1}
    df['riesgo_num'] = df['categoría_riesgo'].map(riesgo_map)
    
    # Mapeo de tipo empresa (empresas pequeñas = mayor riesgo)
    empresa_riesgo = {'Pequeña': 3, 'Mediana': 2, 'Grande': 1}
    df['empresa_riesgo'] = df['tipo_empresa'].map(empresa_riesgo)
    
    # Calcular score de riesgo compuesto
    # Menor score = menor riesgo
    df['score_riesgo'] = (
        df['riesgo_num'] * 30 +
        df['empresa_riesgo'] * 20 +
        (100 - df['score_performance']) * 0.4 +
        (3 - df['num_certificaciones'].clip(0, 3)) * 10
    )
    
    # Segmentar por cuartiles
    df['segmento_riesgo'] = pd.qcut(
        df['score_riesgo'],
        q=4,
        labels=['Premium', 'Estándar', 'Supervisión', 'Alto_Riesgo']
    )
    
    return df


def analisis_correlacion_empresa_performance(df: pd.DataFrame) -> Dict:
    """
    Analiza la correlación entre tipo de empresa y métricas de performance.
    
    Args:
        df: DataFrame con datos de seccion_C
        
    Returns:
        Diccionario con estadísticas por tipo de empresa
    """
    resultados = {}
    
    # Agrupar por tipo de empresa
    grupo = df.groupby('tipo_empresa').agg({
        'on_time_delivery': ['mean', 'std'],
        'quality_issues': ['mean', 'sum'],
        'customer_complaints': ['mean', 'sum'],
        'score_performance': 'mean',
        'ID_proveedor': 'count'
    }).round(2)
    
    grupo.columns = [
        'entrega_puntual_media', 'entrega_puntual_std',
        'issues_calidad_media', 'issues_calidad_total',
        'quejas_media', 'quejas_total',
        'score_performance_media', 'total_proveedores'
    ]
    
    resultados['estadisticas_tipo_empresa'] = grupo.to_dict()
    
    # Ranking por tipo de empresa
    ranking = df.groupby('tipo_empresa')['score_performance'].mean().sort_values(ascending=False)
    resultados['ranking_tipo_empresa'] = ranking.to_dict()
    
    return resultados


def analisis_patrones_geograficos(df: pd.DataFrame) -> Dict:
    """
    Identifica patrones geográficos de riesgo por país.
    
    Args:
        df: DataFrame con datos de seccion_C
        
    Returns:
        Diccionario con análisis por país
    """
    resultados = {}
    
    # Estadísticas por país
    grupo = df.groupby('país').agg({
        'score_riesgo': ['mean', 'std'],
        'score_performance': 'mean',
        'límites_crédito': ['mean', 'sum'],
        'categoría_riesgo': lambda x: (x == 'Alto').sum(),
        'ID_proveedor': 'count'
    }).round(2)
    
    grupo.columns = [
        'riesgo_medio', 'riesgo_std',
        'performance_medio', 
        'credito_promedio', 'credito_total',
        'proveedores_alto_riesgo', 'total_proveedores'
    ]
    
    # Calcular porcentaje de alto riesgo
    grupo['pct_alto_riesgo'] = (
        grupo['proveedores_alto_riesgo'] / grupo['total_proveedores'] * 100
    ).round(1)
    
    resultados['estadisticas_pais'] = grupo.to_dict()
    
    # Ranking de países por riesgo promedio
    ranking_riesgo = df.groupby('país')['score_riesgo'].mean().sort_values(ascending=False)
    resultados['ranking_paises_riesgo'] = ranking_riesgo.to_dict()
    
    return resultados


def evaluar_certificaciones(df: pd.DataFrame) -> Dict:
    """
    Evalúa el impacto de las certificaciones en el performance.
    
    Args:
        df: DataFrame con datos de seccion_C
        
    Returns:
        Diccionario con análisis de certificaciones
    """
    resultados = {}
    
    # Performance por número de certificaciones
    grupo_cert = df.groupby('num_certificaciones').agg({
        'score_performance': 'mean',
        'on_time_delivery': 'mean',
        'score_riesgo': 'mean',
        'ID_proveedor': 'count'
    }).round(2)
    
    grupo_cert.columns = [
        'performance_medio', 'entrega_puntual_media',
        'riesgo_medio', 'total_proveedores'
    ]
    
    resultados['impacto_num_certificaciones'] = grupo_cert.to_dict()
    
    # Análisis por tipo de certificación
    todas_certs = []
    for _, row in df.iterrows():
        for cert in row['certificaciones_list']:
            if cert != 'None':
                todas_certs.append({
                    'certificacion': cert,
                    'performance': row['score_performance'],
                    'riesgo': row['score_riesgo']
                })
    
    df_certs = pd.DataFrame(todas_certs)
    if not df_certs.empty:
        grupo_tipo = df_certs.groupby('certificacion').agg({
            'performance': 'mean',
            'riesgo': 'mean',
            'certificacion': 'count'
        }).rename(columns={'certificacion': 'frecuencia'}).round(2)
        
        resultados['impacto_tipo_certificacion'] = grupo_tipo.to_dict()
    
    return resultados


def generar_matriz_aprobacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera matriz de aprobación por segmento con límites recomendados.
    
    Args:
        df: DataFrame con datos de seccion_C
        
    Returns:
        DataFrame con matriz de aprobación
    """
    # Definir criterios por segmento
    criterios = {
        'Premium': {
            'limite_credito_max': 50000,
            'condiciones_pago': 'Net 90',
            'descuento_maximo': 15,
            'requiere_aprobacion': 'Automática',
            'frecuencia_revision': 'Anual'
        },
        'Estándar': {
            'limite_credito_max': 35000,
            'condiciones_pago': 'Net 60',
            'descuento_maximo': 10,
            'requiere_aprobacion': 'Jefe Compras',
            'frecuencia_revision': 'Semestral'
        },
        'Supervisión': {
            'limite_credito_max': 20000,
            'condiciones_pago': 'Net 30',
            'descuento_maximo': 5,
            'requiere_aprobacion': 'Gerente',
            'frecuencia_revision': 'Trimestral'
        },
        'Alto_Riesgo': {
            'limite_credito_max': 10000,
            'condiciones_pago': 'Anticipado',
            'descuento_maximo': 0,
            'requiere_aprobacion': 'Director',
            'frecuencia_revision': 'Mensual'
        }
    }
    
    matriz = pd.DataFrame(criterios).T
    matriz.index.name = 'segmento'
    
    # Agregar estadísticas reales por segmento
    stats = df.groupby('segmento_riesgo', observed=True).agg({
        'ID_proveedor': 'count',
        'score_performance': 'mean',
        'límites_crédito': 'mean'
    }).round(2)
    
    stats.columns = ['num_proveedores', 'performance_promedio', 'credito_actual_promedio']
    
    matriz = matriz.join(stats)
    
    return matriz


def generar_recomendaciones_credito(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera recomendaciones de límites de crédito por proveedor.
    
    Args:
        df: DataFrame con datos de seccion_C
        
    Returns:
        DataFrame con recomendaciones
    """
    df = df.copy()
    
    # Límite base por segmento
    limite_base = {
        'Premium': 45000,
        'Estándar': 30000,
        'Supervisión': 18000,
        'Alto_Riesgo': 8000
    }
    
    df['limite_base'] = df['segmento_riesgo'].map(limite_base).astype(float)
    
    # Ajustar por performance
    df['factor_performance'] = df['score_performance'] / 80  # 80 como benchmark
    
    # Ajustar por tipo de empresa
    factor_empresa = {'Grande': 1.3, 'Mediana': 1.0, 'Pequeña': 0.8}
    df['factor_empresa'] = df['tipo_empresa'].map(factor_empresa).astype(float)
    
    # Calcular límite recomendado
    df['limite_recomendado'] = (
        df['limite_base'] * df['factor_performance'] * df['factor_empresa']
    ).round(-2)  # Redondear a centenas
    
    # Variación vs actual
    df['variacion_credito'] = (
        (df['limite_recomendado'] - df['límites_crédito']) / df['límites_crédito'] * 100
    ).round(1)
    
    # Recomendación
    df['accion_credito'] = df['variacion_credito'].apply(
        lambda x: 'Aumentar' if x > 10 else ('Reducir' if x < -10 else 'Mantener')
    )
    
    return df[[
        'ID_proveedor', 'nombre', 'tipo_empresa', 'país',
        'segmento_riesgo', 'nivel_performance',
        'límites_crédito', 'limite_recomendado', 
        'variacion_credito', 'accion_credito'
    ]]


def ejecutar_analisis_completo() -> Dict:
    """
    Ejecuta el análisis completo de seccion_C.
    
    Returns:
        Diccionario con todos los resultados del análisis
    """
    print("Iniciando análisis de seccion_C...")
    
    # Cargar datos
    df = cargar_datos_proveedores()
    print(f"  - {len(df)} seccion_C cargados")
    
    # Calcular performance
    df = calcular_score_performance(df)
    print("  - Score de performance calculado")
    
    # Segmentación de riesgo
    df = segmentar_proveedores_riesgo(df)
    print("  - Segmentación de riesgo completada")
    
    # Análisis por tipo de empresa
    correlacion_empresa = analisis_correlacion_empresa_performance(df)
    print("  - Análisis tipo empresa vs performance completado")
    
    # Patrones geográficos
    patrones_geo = analisis_patrones_geograficos(df)
    print("  - Patrones geográficos identificados")
    
    # Evaluación de certificaciones
    impacto_certs = evaluar_certificaciones(df)
    print("  - Impacto de certificaciones evaluado")
    
    # Matriz de aprobación
    matriz_aprobacion = generar_matriz_aprobacion(df)
    print("  - Matriz de aprobación generada")
    
    # Recomendaciones de crédito
    recomendaciones = generar_recomendaciones_credito(df)
    print("  - Recomendaciones de crédito generadas")
    
    # Compilar resultados
    resultados = {
        'datos_procesados': df,
        'correlacion_empresa': correlacion_empresa,
        'patrones_geograficos': patrones_geo,
        'impacto_certificaciones': impacto_certs,
        'matriz_aprobacion': matriz_aprobacion,
        'recomendaciones_credito': recomendaciones
    }
    
    print("\n¡Análisis completado!")
    
    return resultados


def exportar_resultados(resultados: Dict, prefijo: str = 'seccion_c') -> None:
    """
    Exporta los resultados del análisis a archivos CSV.
    
    Args:
        resultados: Diccionario con resultados del análisis
        prefijo: Prefijo para los nombres de archivo
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Exportar DataFrame principal
    df = resultados['datos_procesados']
    cols_export = [
        'ID_proveedor', 'nombre', 'tipo_empresa', 'país', 
        'categoría_riesgo', 'score_performance', 'nivel_performance',
        'score_riesgo', 'segmento_riesgo', 'num_certificaciones'
    ]
    df[cols_export].to_csv(
        os.path.join(OUTPUT_DIR, f'{prefijo}_segmentacion_proveedores.csv'),
        index=False
    )
    
    # Exportar matriz de aprobación
    resultados['matriz_aprobacion'].to_csv(
        os.path.join(OUTPUT_DIR, f'{prefijo}_matriz_aprobacion.csv')
    )
    
    # Exportar recomendaciones de crédito
    resultados['recomendaciones_credito'].to_csv(
        os.path.join(OUTPUT_DIR, f'{prefijo}_recomendaciones_credito.csv'),
        index=False
    )
    
    print(f"\nResultados exportados en: {OUTPUT_DIR}")


# Ejecución como script
if __name__ == '__main__':
    resultados = ejecutar_analisis_completo()
    exportar_resultados(resultados)
    
    # Mostrar resumen
    print("\n" + "="*60)
    print("RESUMEN EJECUTIVO")
    print("="*60)
    
    df = resultados['datos_procesados']
    
    print("\n1. DISTRIBUCIÓN POR SEGMENTO DE RIESGO:")
    print(df['segmento_riesgo'].value_counts().to_string())
    
    print("\n2. DISTRIBUCIÓN POR NIVEL DE PERFORMANCE:")
    print(df['nivel_performance'].value_counts().to_string())
    
    print("\n3. MATRIZ DE APROBACIÓN:")
    print(resultados['matriz_aprobacion'].to_string())
    
    print("\n4. ACCIONES DE CRÉDITO RECOMENDADAS:")
    print(resultados['recomendaciones_credito']['accion_credito'].value_counts().to_string())
