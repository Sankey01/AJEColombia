
# Documentación de Datasets para Evaluación de Auditores

Este repositorio contiene varios datasets generados para la evaluación de auditores con un enfoque en analítica de datos e inteligencia artificial. Los datos son ficticios pero altamente realistas, diseñados para incluir anomalías intencionales y patrones de riesgo.

## Descripción de los Datasets

### 1. Condiciones de Pagos
- **Archivo**: `condiciones_pagos.csv`
- **Estructura de Datos**:
  - `ID_transaccion`: Identificador único de la transacción.
  - `proveedor_id`: Identificador del proveedor.
  - `fecha_factura`: Fecha de emisión de la factura.
  - `fecha_vencimiento`: Fecha de vencimiento del pago.
  - `monto`: Monto de la transacción.
  - `condiciones_pago`: Términos de pago (30, 60, 90 días).
  - `estado_pago`: Estado del pago (Pagado, Pendiente, Retrasado).
  - `dias_credito`: Días de crédito otorgados.
  - `descuento_pronto_pago`: Descuento por pronto pago.
  - `penalizacion_mora`: Penalización por mora.
  - `metodo_pago`: Método de pago utilizado.
  - `aprobador`: Persona que aprobó el pago.

- **Anomalías Intencionales**:
  - Pagos duplicados.
  - Montos irregulares.
  - Plazos inconsistentes.
  - Aprobadores no autorizados.

- **Casos de Uso**:
  - Análisis de patrones de pago.
  - Detección de anomalías en transacciones.
  - Evaluación de políticas de crédito.

### 2. Transacciones y Políticas de Pago
- **Archivo**: `transacciones_politicas_pago.json`
- **Estructura de Datos**:
  - `payment_policies`: Políticas de pago por tipo de proveedor.
  - `transactions`: Lista de transacciones con detalles como fecha, monto, y términos de pago.

- **Anomalías Intencionales**:
  - Transacciones con términos de pago inconsistentes.

- **Casos de Uso**:
  - Evaluación de políticas de pago.
  - Análisis de flujo de aprobación.

### 3. Proveedores y Condiciones Contractuales
- **Archivo**: `proveedores_condiciones.csv`
- **Estructura de Datos**:
  - `ID_proveedor`: Identificador único del proveedor.
  - `nombre`: Nombre del proveedor.
  - `tipo_empresa`: Tamaño de la empresa (Pequeña, Mediana, Grande).
  - `país`: País de origen.
  - `categoría_riesgo`: Nivel de riesgo asociado.
  - `condiciones_negociadas`: Términos de pago negociados.
  - `descuentos_especiales`: Descuentos aplicables.
  - `límites_crédito`: Límites de crédito otorgados.
  - `histórico_performance`: Historial de desempeño.
  - `contactos_emergencia`: Contacto de emergencia.
  - `certificaciones`: Certificaciones obtenidas.

- **Anomalías Intencionales**:
  - Proveedores con alto riesgo y problemas de calidad.

- **Casos de Uso**:
  - Evaluación de riesgo de proveedores.
  - Análisis de condiciones contractuales.

### 4. Históricos de Pagos
- **Archivo**: `historicos_pagos.csv`
- **Estructura de Datos**:
  - `fecha`: Fecha de la transacción.
  - `proveedor_id`: Identificador del proveedor.
  - `categoria_gasto`: Categoría del gasto.
  - `monto`: Monto de la transacción.
  - `método_pago`: Método de pago utilizado.
  - `tiempo_procesamiento`: Tiempo de procesamiento del pago.
  - `score_cumplimiento`: Puntaje de cumplimiento.
  - `indicadores_economicos`: Indicadores económicos asociados.
  - `estacionalidad`: Nivel de estacionalidad.

- **Anomalías Intencionales**:
  - Eventos atípicos en montos de transacciones.

- **Casos de Uso**:
  - Análisis de tendencias históricas.
  - Predicción de flujos de caja.

### 5. Configuración para Herramientas de IA
- **Archivo**: `configuracion_ia.json`
- **Estructura de Datos**:
  - Configuraciones para modelos predictivos y prescriptivos.
  - Parámetros para algoritmos de machine learning.
  - Umbrales de anomalías y reglas de negocio.

- **Casos de Uso**:
  - Configuración de modelos de IA.
  - Optimización de procesos de negocio.

### 6. Excepciones y Casos Especiales
- **Archivo**: `excepciones_casos_especiales.csv`
- **Estructura de Datos**:
  - `tipo_caso`: Tipo de caso especial.
  - `nivel_complejidad`: Nivel de complejidad del caso.
  - `fecha_incidencia`: Fecha de la incidencia.
  - `descripcion`: Descripción del caso.
  - `monto_involucrado`: Monto involucrado en el caso.
  - `estado`: Estado actual del caso.
  - `responsable`: Persona responsable del caso.
  - `comentarios`: Comentarios adicionales.

- **Anomalías Intencionales**:
  - Casos con montos inusualmente altos.

- **Casos de Uso**:
  - Análisis de excepciones y casos especiales.
  - Evaluación de procesos de aprobación manual.

## Guía de Evaluación para Candidatos

Los candidatos deben ser capaces de:
- Identificar y analizar anomalías en los datasets.
- Desarrollar modelos predictivos y prescriptivos basados en los datos.
- Proponer mejoras en las políticas de pago y condiciones contractuales.
- Evaluar el riesgo asociado a proveedores y transacciones.
- Configurar y optimizar herramientas de IA para análisis avanzado.

Este conjunto de datos está diseñado para evaluar competencias analíticas avanzadas y la capacidad de aplicar técnicas de inteligencia artificial en contextos de auditoría y análisis de riesgos.
