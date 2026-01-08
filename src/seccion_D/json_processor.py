"""
Módulo de Procesamiento JSON - Sección D
==========================================
Extrae, valida, transforma y analiza datos JSON de transacciones y políticas de pago.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
import os

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESOURCE_DIR = os.path.join(BASE_DIR, 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


class JSONProcessor:
    """
    Procesador de JSON para transacciones y políticas de pago.
    Implementa extracción, validación, transformación y análisis de datos.
    """
    
    def __init__(self, archivo: str = 'transacciones_politicas_pago.json'):
        """
        Inicializa el procesador cargando el archivo JSON.
        
        Args:
            archivo: Nombre del archivo JSON a procesar
        """
        self.archivo = archivo
        self.ruta = os.path.join(RESOURCE_DIR, archivo)
        self.data = None
        self.policies = None
        self.transactions_df = None
        self.validation_report = {}
        
    def cargar_json(self) -> Dict:
        """
        Carga y parsea el archivo JSON.
        
        Returns:
            Diccionario con los datos del JSON
        """
        try:
            with open(self.ruta, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✓ JSON cargado exitosamente: {self.archivo}")
            return self.data
        except json.JSONDecodeError as e:
            raise ValueError(f"Error al parsear JSON: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Archivo no encontrado: {self.ruta}")
    
    def validar_estructura(self) -> Dict:
        """
        Valida la estructura del JSON y genera reporte de validación.
        
        Returns:
            Diccionario con el reporte de validación
        """
        if self.data is None:
            self.cargar_json()
        
        reporte = {
            'estructura_valida': True,
            'errores': [],
            'advertencias': [],
            'estadisticas': {}
        }
        
        # Validar claves principales
        claves_requeridas = ['payment_policies', 'transactions']
        for clave in claves_requeridas:
            if clave not in self.data:
                reporte['estructura_valida'] = False
                reporte['errores'].append(f"Falta clave principal: '{clave}'")
        
        if not reporte['estructura_valida']:
            self.validation_report = reporte
            return reporte
        
        # Validar estructura de políticas
        policies = self.data.get('payment_policies', {}).get('policies', {})
        supplier_types = policies.get('supplier_type', {})
        
        reporte['estadisticas']['tipos_proveedor'] = list(supplier_types.keys())
        reporte['estadisticas']['num_tipos_proveedor'] = len(supplier_types)
        
        for tipo, config in supplier_types.items():
            campos_requeridos = ['authorization_limit', 'approval_flow', 'discounts', 'penalties']
            for campo in campos_requeridos:
                if campo not in config:
                    reporte['advertencias'].append(
                        f"Tipo '{tipo}' no tiene campo '{campo}'"
                    )
        
        # Validar estructura de transacciones
        transactions = self.data.get('transactions', [])
        reporte['estadisticas']['num_transacciones'] = len(transactions)
        
        campos_transaccion = [
            'transaction_id', 'supplier_id', 'supplier_type', 
            'invoice_date', 'due_date', 'amount', 'payment_terms', 
            'status', 'metadata'
        ]
        
        campos_faltantes = set()
        for i, trans in enumerate(transactions):
            for campo in campos_transaccion:
                if campo not in trans:
                    campos_faltantes.add(campo)
        
        if campos_faltantes:
            reporte['advertencias'].append(
                f"Campos faltantes en algunas transacciones: {campos_faltantes}"
            )
        
        # Validar tipos de datos
        tipos_invalidos = []
        for trans in transactions:
            if not isinstance(trans.get('transaction_id'), int):
                tipos_invalidos.append(f"transaction_id inválido: {trans.get('transaction_id')}")
            if not isinstance(trans.get('amount'), (int, float)):
                tipos_invalidos.append(f"amount inválido en trans {trans.get('transaction_id')}")
        
        if tipos_invalidos:
            reporte['advertencias'].extend(tipos_invalidos[:5])  # Solo primeros 5
            if len(tipos_invalidos) > 5:
                reporte['advertencias'].append(f"... y {len(tipos_invalidos) - 5} más")
        
        # Validar metadata
        trans_sin_metadata = sum(1 for t in transactions if 'metadata' not in t)
        if trans_sin_metadata > 0:
            reporte['advertencias'].append(
                f"{trans_sin_metadata} transacciones sin metadata"
            )
        
        self.validation_report = reporte
        print(f"✓ Validación completada. Estructura válida: {reporte['estructura_valida']}")
        return reporte
    
    def extraer_politicas(self) -> pd.DataFrame:
        """
        Extrae las políticas de pago a formato tabular.
        
        Returns:
            DataFrame con las políticas aplanadas
        """
        if self.data is None:
            self.cargar_json()
        
        policies = self.data.get('payment_policies', {}).get('policies', {})
        supplier_types = policies.get('supplier_type', {})
        
        rows = []
        for tipo, config in supplier_types.items():
            row = {
                'supplier_type': tipo,
                'authorization_limit': config.get('authorization_limit'),
                'approval_flow': ', '.join(config.get('approval_flow', [])),
                'discount_early_payment': config.get('discounts', {}).get('early_payment'),
                'discount_bulk_order': config.get('discounts', {}).get('bulk_order'),
                'penalty_late_payment': config.get('penalties', {}).get('late_payment')
            }
            rows.append(row)
        
        self.policies = pd.DataFrame(rows)
        print(f"✓ Políticas extraídas: {len(self.policies)} tipos de proveedor")
        return self.policies
    
    def transformar_transacciones(self) -> pd.DataFrame:
        """
        Transforma las transacciones anidadas a formato tabular plano.
        
        Returns:
            DataFrame con las transacciones aplanadas
        """
        if self.data is None:
            self.cargar_json()
        
        transactions = self.data.get('transactions', [])
        
        rows = []
        for trans in transactions:
            metadata = trans.get('metadata', {})
            related = metadata.get('related_transactions', [])
            
            row = {
                'transaction_id': trans.get('transaction_id'),
                'supplier_id': trans.get('supplier_id'),
                'supplier_type': trans.get('supplier_type'),
                'invoice_date': trans.get('invoice_date'),
                'due_date': trans.get('due_date'),
                'amount': trans.get('amount'),
                'payment_terms': trans.get('payment_terms'),
                'status': trans.get('status'),
                'created_by': metadata.get('created_by'),
                'created_at': metadata.get('created_at'),
                'num_related_transactions': len(related),
                'related_transactions': ','.join(map(str, related)) if related else ''
            }
            rows.append(row)
        
        self.transactions_df = pd.DataFrame(rows)
        
        # Convertir fechas
        self.transactions_df['invoice_date'] = pd.to_datetime(
            self.transactions_df['invoice_date'], errors='coerce'
        )
        self.transactions_df['due_date'] = pd.to_datetime(
            self.transactions_df['due_date'], errors='coerce'
        )
        self.transactions_df['created_at'] = pd.to_datetime(
            self.transactions_df['created_at'], errors='coerce'
        )
        
        # Calcular métricas adicionales
        self.transactions_df['days_until_due'] = (
            self.transactions_df['due_date'] - self.transactions_df['invoice_date']
        ).dt.days
        
        # Extraer días de payment_terms
        self.transactions_df['payment_terms_days'] = self.transactions_df['payment_terms'].str.extract(
            r'(\d+)'
        ).astype(float)
        
        print(f"✓ Transacciones transformadas: {len(self.transactions_df)} registros")
        return self.transactions_df
    
    def validar_integridad(self) -> Dict:
        """
        Valida la integridad de los datos transformados.
        
        Returns:
            Diccionario con validaciones de integridad
        """
        if self.transactions_df is None:
            self.transformar_transacciones()
        
        if self.policies is None:
            self.extraer_politicas()
        
        df = self.transactions_df
        integridad = {
            'duplicados': {},
            'valores_nulos': {},
            'rangos_invalidos': {},
            'referencias_invalidas': {}
        }
        
        # Verificar duplicados en transaction_id
        duplicados = df[df.duplicated(subset=['transaction_id'], keep=False)]
        integridad['duplicados']['transaction_id'] = len(duplicados)
        
        # Valores nulos por columna
        nulos = df.isnull().sum()
        integridad['valores_nulos'] = nulos[nulos > 0].to_dict()
        
        # Rangos inválidos
        montos_negativos = df[df['amount'] < 0]
        integridad['rangos_invalidos']['montos_negativos'] = len(montos_negativos)
        
        fechas_invalidas = df[df['due_date'] < df['invoice_date']]
        integridad['rangos_invalidos']['due_date_antes_invoice'] = len(fechas_invalidas)
        
        # Tipos de proveedor válidos
        tipos_validos = set(self.policies['supplier_type'])
        tipos_transacciones = set(df['supplier_type'].unique())
        tipos_invalidos = tipos_transacciones - tipos_validos
        integridad['referencias_invalidas']['tipos_proveedor_invalidos'] = list(tipos_invalidos)
        
        # Verificar referencias cruzadas (related_transactions existen)
        all_ids = set(df['transaction_id'])
        referencias_rotas = []
        for _, row in df.iterrows():
            if row['related_transactions']:
                related = [int(x) for x in row['related_transactions'].split(',') if x]
                for ref in related:
                    if ref not in all_ids:
                        referencias_rotas.append({
                            'transaction_id': row['transaction_id'],
                            'referencia_rota': ref
                        })
        
        integridad['referencias_invalidas']['referencias_rotas'] = len(referencias_rotas)
        integridad['referencias_invalidas']['detalle_rotas'] = referencias_rotas[:10]
        
        print(f"✓ Validación de integridad completada")
        return integridad
    
    def detectar_inconsistencias_politicas(self) -> pd.DataFrame:
        """
        Detecta inconsistencias entre políticas de pago y transacciones.
        
        Returns:
            DataFrame con las inconsistencias detectadas
        """
        if self.transactions_df is None:
            self.transformar_transacciones()
        
        if self.policies is None:
            self.extraer_politicas()
        
        # Crear mapa de límites por tipo
        limites = self.policies.set_index('supplier_type')['authorization_limit'].to_dict()
        
        df = self.transactions_df.copy()
        inconsistencias = []
        
        for _, trans in df.iterrows():
            supplier_type = trans['supplier_type']
            amount = trans['amount']
            limite = limites.get(supplier_type, float('inf'))
            
            # 1. Monto excede límite de autorización
            if amount > limite:
                inconsistencias.append({
                    'transaction_id': trans['transaction_id'],
                    'tipo_inconsistencia': 'EXCEDE_LIMITE',
                    'descripcion': f"Monto ${amount:.2f} excede límite ${limite} para tipo '{supplier_type}'",
                    'severidad': 'ALTA',
                    'supplier_type': supplier_type,
                    'amount': amount,
                    'limite': limite,
                    'exceso': amount - limite
                })
            
            # 2. Status Overdue con payment_terms inconsistentes
            if trans['status'] == 'Overdue':
                days_until_due = trans['days_until_due']
                if days_until_due and days_until_due < 0:
                    inconsistencias.append({
                        'transaction_id': trans['transaction_id'],
                        'tipo_inconsistencia': 'FECHA_INVALIDA',
                        'descripcion': f"Due date anterior a invoice date ({days_until_due} días)",
                        'severidad': 'MEDIA',
                        'supplier_type': supplier_type,
                        'amount': amount,
                        'limite': None,
                        'exceso': None
                    })
            
            # 3. Verificar coherencia payment_terms vs days_until_due
            terms_days = trans['payment_terms_days']
            actual_days = trans['days_until_due']
            if pd.notna(terms_days) and pd.notna(actual_days):
                # Permitir margen de 30 días de tolerancia
                if abs(actual_days - terms_days) > 365:  # Más de un año de diferencia
                    inconsistencias.append({
                        'transaction_id': trans['transaction_id'],
                        'tipo_inconsistencia': 'TERMINOS_INCOHERENTES',
                        'descripcion': f"Payment terms ({terms_days}d) vs días reales ({actual_days}d)",
                        'severidad': 'BAJA',
                        'supplier_type': supplier_type,
                        'amount': amount,
                        'limite': None,
                        'exceso': None
                    })
        
        df_inconsistencias = pd.DataFrame(inconsistencias)
        print(f"✓ Inconsistencias detectadas: {len(df_inconsistencias)}")
        return df_inconsistencias
    
    def generar_reporte_completo(self) -> Dict:
        """
        Genera un reporte completo del procesamiento.
        
        Returns:
            Diccionario con todos los resultados del análisis
        """
        print("\n" + "="*60)
        print("PROCESAMIENTO JSON - SECCIÓN D")
        print("="*60)
        
        # 1. Cargar y validar estructura
        self.cargar_json()
        validacion_estructura = self.validar_estructura()
        
        # 2. Extraer políticas
        df_policies = self.extraer_politicas()
        
        # 3. Transformar transacciones
        df_transactions = self.transformar_transacciones()
        
        # 4. Validar integridad
        integridad = self.validar_integridad()
        
        # 5. Detectar inconsistencias
        df_inconsistencias = self.detectar_inconsistencias_politicas()
        
        # 6. Estadísticas resumen
        resumen = {
            'total_transacciones': len(df_transactions),
            'monto_total': df_transactions['amount'].sum(),
            'monto_promedio': df_transactions['amount'].mean(),
            'distribucion_status': df_transactions['status'].value_counts().to_dict(),
            'distribucion_tipo_proveedor': df_transactions['supplier_type'].value_counts().to_dict(),
            'distribucion_creador': df_transactions['created_by'].value_counts().to_dict(),
            'transacciones_con_referencias': (df_transactions['num_related_transactions'] > 0).sum(),
            'total_inconsistencias': len(df_inconsistencias),
            'inconsistencias_por_tipo': df_inconsistencias['tipo_inconsistencia'].value_counts().to_dict() if not df_inconsistencias.empty else {}
        }
        
        reporte = {
            'validacion_estructura': validacion_estructura,
            'politicas': df_policies,
            'transacciones': df_transactions,
            'integridad': integridad,
            'inconsistencias': df_inconsistencias,
            'resumen': resumen
        }
        
        print("\n✓ Procesamiento completado exitosamente")
        return reporte
    
    def exportar_resultados(self, prefijo: str = 'seccion_d') -> None:
        """
        Exporta todos los resultados a archivos CSV y JSON.
        
        Args:
            prefijo: Prefijo para los nombres de archivo
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Generar reporte si no existe
        if self.transactions_df is None:
            self.generar_reporte_completo()
        
        # Exportar políticas
        self.policies.to_csv(
            os.path.join(OUTPUT_DIR, f'{prefijo}_politicas_pago.csv'),
            index=False
        )
        
        # Exportar transacciones
        self.transactions_df.to_csv(
            os.path.join(OUTPUT_DIR, f'{prefijo}_transacciones.csv'),
            index=False
        )
        
        # Exportar inconsistencias
        inconsistencias = self.detectar_inconsistencias_politicas()
        if not inconsistencias.empty:
            inconsistencias.to_csv(
                os.path.join(OUTPUT_DIR, f'{prefijo}_inconsistencias.csv'),
                index=False
            )
        
        # Exportar reporte de validación
        with open(os.path.join(OUTPUT_DIR, f'{prefijo}_reporte_validacion.json'), 'w', encoding='utf-8') as f:
            # Convertir integridad a serializable
            integridad = self.validar_integridad()
            reporte = {
                'validacion_estructura': self.validation_report,
                'integridad': integridad,
                'resumen': {
                    'total_transacciones': len(self.transactions_df),
                    'monto_total': float(self.transactions_df['amount'].sum()),
                    'monto_promedio': float(self.transactions_df['amount'].mean()),
                    'distribucion_status': self.transactions_df['status'].value_counts().to_dict(),
                    'total_inconsistencias': len(inconsistencias)
                }
            }
            json.dump(reporte, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n✓ Resultados exportados en: {OUTPUT_DIR}")
        print(f"  - {prefijo}_politicas_pago.csv")
        print(f"  - {prefijo}_transacciones.csv")
        print(f"  - {prefijo}_inconsistencias.csv")
        print(f"  - {prefijo}_reporte_validacion.json")


def ejecutar_procesamiento_json() -> Dict:
    """
    Función principal para ejecutar el procesamiento completo.
    
    Returns:
        Diccionario con todos los resultados
    """
    processor = JSONProcessor()
    reporte = processor.generar_reporte_completo()
    processor.exportar_resultados()
    return reporte


# Ejecución como script
if __name__ == '__main__':
    resultados = ejecutar_procesamiento_json()
    
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)
    
    resumen = resultados['resumen']
    print(f"\n📊 Total transacciones: {resumen['total_transacciones']}")
    print(f"💰 Monto total: ${resumen['monto_total']:,.2f}")
    print(f"📈 Monto promedio: ${resumen['monto_promedio']:,.2f}")
    
    print("\n📋 Distribución por Status:")
    for status, count in resumen['distribucion_status'].items():
        print(f"   - {status}: {count}")
    
    print("\n🏢 Distribución por Tipo Proveedor:")
    for tipo, count in resumen['distribucion_tipo_proveedor'].items():
        print(f"   - {tipo}: {count}")
    
    print(f"\n⚠️  Total inconsistencias detectadas: {resumen['total_inconsistencias']}")
    if resumen['inconsistencias_por_tipo']:
        for tipo, count in resumen['inconsistencias_por_tipo'].items():
            print(f"   - {tipo}: {count}")
