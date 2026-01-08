"""
Módulo Workflow de Excepciones - Sección F
Flujo de trabajo automatizado y matriz de escalación
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List
from datetime import timedelta

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


# Configuración de SLA por tipo de caso y complejidad
SLA_CONFIG = {
    'Proveedor VIP': {'Bajo': 2, 'Medio': 1, 'Alto': 0.5},  # días
    'Pago Urgente': {'Bajo': 1, 'Medio': 0.5, 'Alto': 0.25},
    'Ajuste Contable': {'Bajo': 5, 'Medio': 3, 'Alto': 2},
    'Excepción Regulatoria': {'Bajo': 3, 'Medio': 2, 'Alto': 1},
    'Aprobación Manual': {'Bajo': 3, 'Medio': 2, 'Alto': 1},
    'Caso de Fuerza Mayor': {'Bajo': 2, 'Medio': 1, 'Alto': 0.5},
    'Transacción con Disputa': {'Bajo': 7, 'Medio': 5, 'Alto': 3},
    'Transacción Internacional Compleja': {'Bajo': 5, 'Medio': 3, 'Alto': 2},
}

# Configuración de niveles de aprobación
NIVELES_APROBACION = {
    'Bajo': ['Operativo'],
    'Medio': ['Operativo', 'Supervisor'],
    'Alto': ['Operativo', 'Supervisor', 'Gerencia'],
}

# Umbrales de monto para escalación adicional
UMBRAL_MONTO_SUPERVISOR = 25000
UMBRAL_MONTO_GERENCIA = 40000


def generar_matriz_escalacion() -> pd.DataFrame:
    """
    Genera la matriz de escalación por tipo de caso y nivel de complejidad.
    
    Returns:
        DataFrame con la matriz de escalación
    """
    filas = []
    
    for tipo_caso in SLA_CONFIG.keys():
        for complejidad in ['Bajo', 'Medio', 'Alto']:
            sla_dias = SLA_CONFIG[tipo_caso][complejidad]
            niveles = NIVELES_APROBACION[complejidad]
            
            fila = {
                'tipo_caso': tipo_caso,
                'complejidad': complejidad,
                'sla_dias': sla_dias,
                'sla_horas': sla_dias * 24,
                'nivel_1_operativo': 'Sí',
                'nivel_2_supervisor': 'Sí' if 'Supervisor' in niveles else 'No',
                'nivel_3_gerencia': 'Sí' if 'Gerencia' in niveles else 'No',
                'aprobadores_requeridos': len(niveles),
            }
            filas.append(fila)
    
    return pd.DataFrame(filas)


def asignar_ruta_escalacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna rutas de escalación automáticas a cada caso.
    
    Args:
        df: DataFrame con casos clasificados
        
    Returns:
        DataFrame con rutas de escalación asignadas
    """
    df = df.copy()
    
    def determinar_ruta(row):
        complejidad = row['nivel_complejidad']
        monto = row['monto_involucrado']
        
        # Obtener niveles base por complejidad
        niveles = NIVELES_APROBACION.get(complejidad, ['Operativo']).copy()
        
        # Escalar por monto si aplica
        if monto >= UMBRAL_MONTO_GERENCIA and 'Gerencia' not in niveles:
            niveles.append('Gerencia')
        elif monto >= UMBRAL_MONTO_SUPERVISOR and 'Supervisor' not in niveles:
            niveles.append('Supervisor')
        
        return ' → '.join(niveles)
    
    def determinar_sla(row):
        tipo = row['tipo_caso']
        complejidad = row['nivel_complejidad']
        return SLA_CONFIG.get(tipo, {}).get(complejidad, 3)
    
    df['ruta_escalacion'] = df.apply(determinar_ruta, axis=1)
    df['sla_dias'] = df.apply(determinar_sla, axis=1)
    df['sla_vencimiento'] = df['fecha_incidencia'] + pd.to_timedelta(df['sla_dias'], unit='d')
    
    return df


def calcular_cumplimiento_sla(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el estado de cumplimiento del SLA para cada caso.
    
    Args:
        df: DataFrame con SLA asignado
        
    Returns:
        DataFrame con estado de cumplimiento
    """
    df = df.copy()
    
    from datetime import datetime
    ahora = datetime.now()
    
    def evaluar_sla(row):
        if row['estado'] == 'Cerrado':
            return 'Cumplido'
        elif pd.to_datetime(row['sla_vencimiento']) < ahora:
            return 'Vencido'
        elif (pd.to_datetime(row['sla_vencimiento']) - ahora).days <= 1:
            return 'Por Vencer'
        else:
            return 'En Tiempo'
    
    df['estado_sla'] = df.apply(evaluar_sla, axis=1)
    
    return df


def generar_flujo_trabajo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera el flujo de trabajo completo para cada caso.
    
    Args:
        df: DataFrame con casos
        
    Returns:
        DataFrame con flujo de trabajo detallado
    """
    df = asignar_ruta_escalacion(df)
    df = calcular_cumplimiento_sla(df)
    
    # Seleccionar columnas relevantes para el flujo
    columnas_flujo = [
        'tipo_caso', 'nivel_complejidad', 'monto_involucrado', 'estado',
        'responsable', 'ruta_escalacion', 'sla_dias', 'estado_sla',
        'fecha_incidencia', 'sla_vencimiento'
    ]
    
    return df[columnas_flujo].copy()


def proponer_mejoras_proceso() -> List[Dict]:
    """
    Propone mejoras en los procesos de aprobación basadas en análisis.
    
    Returns:
        Lista de propuestas de mejora
    """
    mejoras = [
        {
            'area': 'Automatización',
            'propuesta': 'Implementar aprobación automática para casos de baja complejidad y monto < $10,000',
            'impacto': 'Alto',
            'esfuerzo': 'Medio',
            'prioridad': 1,
        },
        {
            'area': 'SLA',
            'propuesta': 'Establecer alertas automáticas 24 horas antes del vencimiento del SLA',
            'impacto': 'Alto',
            'esfuerzo': 'Bajo',
            'prioridad': 2,
        },
        {
            'area': 'Escalación',
            'propuesta': 'Crear canal de escalación express para Pagos Urgentes y Proveedor VIP',
            'impacto': 'Medio',
            'esfuerzo': 'Bajo',
            'prioridad': 3,
        },
        {
            'area': 'Delegación',
            'propuesta': 'Permitir a supervisores aprobar casos de alta complejidad hasta $30,000',
            'impacto': 'Medio',
            'esfuerzo': 'Bajo',
            'prioridad': 4,
        },
        {
            'area': 'Monitoreo',
            'propuesta': 'Dashboard en tiempo real para tracking de casos por responsable',
            'impacto': 'Alto',
            'esfuerzo': 'Alto',
            'prioridad': 5,
        },
        {
            'area': 'Clasificación',
            'propuesta': 'Usar ML para pre-clasificación automática de tipo de caso',
            'impacto': 'Alto',
            'esfuerzo': 'Alto',
            'prioridad': 6,
        },
    ]
    
    return mejoras


def obtener_metricas_workflow(df: pd.DataFrame) -> Dict:
    """
    Calcula métricas del workflow de excepciones.
    
    Args:
        df: DataFrame con flujo de trabajo
        
    Returns:
        Diccionario con métricas
    """
    metricas = {
        'total_casos': len(df),
        'casos_vencidos': len(df[df['estado_sla'] == 'Vencido']),
        'casos_por_vencer': len(df[df['estado_sla'] == 'Por Vencer']),
        'casos_en_tiempo': len(df[df['estado_sla'] == 'En Tiempo']),
        'tasa_cumplimiento': len(df[df['estado_sla'] == 'Cumplido']) / len(df) * 100 if len(df) > 0 else 0,
        'por_ruta_escalacion': df['ruta_escalacion'].value_counts().to_dict(),
        'sla_promedio': df['sla_dias'].mean(),
    }
    
    return metricas


def exportar_matriz_escalacion(prefijo: str = 'seccion_f'):
    """
    Exporta la matriz de escalación a CSV.
    
    Args:
        prefijo: Prefijo para el nombre del archivo
    """
    matriz = generar_matriz_escalacion()
    ruta = os.path.join(OUTPUT_DIR, f'{prefijo}_matriz_escalacion.csv')
    matriz.to_csv(ruta, index=False)
    print(f"Matriz de escalación exportada a: {ruta}")
    return matriz


if __name__ == '__main__':
    from .gestor_excepciones import ejecutar_clasificacion
    
    print("=" * 60)
    print("FLUJO DE TRABAJO Y ESCALACIÓN DE EXCEPCIONES")
    print("=" * 60)
    
    # Cargar casos clasificados
    df_clasificado, _ = ejecutar_clasificacion()
    
    # Generar flujo de trabajo
    df_flujo = generar_flujo_trabajo(df_clasificado)
    
    # Métricas
    metricas = obtener_metricas_workflow(df_flujo)
    
    print(f"\n--- Métricas de SLA ---")
    print(f"Casos vencidos: {metricas['casos_vencidos']}")
    print(f"Casos por vencer: {metricas['casos_por_vencer']}")
    print(f"Casos en tiempo: {metricas['casos_en_tiempo']}")
    
    print("\n--- Matriz de Escalación ---")
    matriz = generar_matriz_escalacion()
    print(matriz.to_string(index=False))
    
    print("\n--- Propuestas de Mejora ---")
    mejoras = proponer_mejoras_proceso()
    for m in mejoras:
        print(f"  [{m['prioridad']}] {m['area']}: {m['propuesta']}")
