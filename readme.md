# AJE Colombia - Plataforma de Análisis de Datos y Gestión de Riesgos

Esta plataforma integral ha sido diseñada para optimizar los procesos de gestión financiera, detección de anomalías y evaluación de riesgos de proveedores en AJE Colombia. El sistema combina análisis estadístico avanzado, modelos de Machine Learning y automatización de flujos de trabajo para mejorar la toma de decisiones.

## Módulos del Sistema

El proyecto está estructurado en secciones modulares, cada una enfocada en un aspecto crítico del negocio:

### Sección A: Detección de Anomalías en Pagos
**Directorio:** `src/seccion_A`
Sistema orientado a la identificación y análisis de irregularidades en los procesos de pago.
*   **Dashboard Interactivo:** Visualización de métricas clave y alertas (`dashboard_anomalias.py`).
*   **Análisis Profundo:** Detección de patrones sospechosos, pagos duplicados y desviaciones en condiciones de crédito.

### Sección B: Evaluación de Riesgo y Scoring
**Directorio:** `src/seccion_B`
Motor de inteligencia artificial para la calificación de riesgo de proveedores.
*   **Modelos ML:** Entrenamiento y evaluación de modelos predictivos (`model_training.py`, `model_evaluation.py`).
*   **Scoring de Riesgo:** Cálculo automatizado de puntuaciones de riesgo (`risk_scoring.py`).
*   **Visualización:** Herramientas para el análisis visual del riesgo (`risk_viz.py`).
*   **Limpieza de Datos:** Pipelines de preprocesamiento robustos.

### Sección C: Análisis y Gestión de Proveedores
**Directorio:** `src/seccion_C`
Herramientas para la evaluación estratégica del desempeño de proveedores.
*   **Análisis de Desempeño:** Correlación entre tipos de empresa, ubicación geográfica y cumplimiento (`analisis_proveedores.py`).
*   **Recomendaciones de Crédito:** Generación automática de sugerencias de líneas de crédito.
*   **Dashboard de Proveedores:** Tablero de control para el seguimiento de métricas de proveedores (`dashboard_proveedores.py`).

### Sección D: Procesamiento de Datos Transaccionales
**Directorio:** `src/seccion_D`
Motor de ingestión y validación de datos.
*   **Procesador JSON:** Transformación, validación y estandarización de logs transaccionales y archivos de configuración (`json_processor.py`).

### Sección E: Integración con IA (Gemini)
**Directorio:** `src/seccion_E`
Módulo de inteligencia artificial generativa.
*   **Gemini Client:** Integración con la API de Google Gemini para análisis de texto y generación de insights (`gemini.py`).
*   **Gestión de Prompts:** Biblioteca de prompts optimizados para consultas de negocio.

### Sección F: Gestión de Excepciones y Workflows
**Directorio:** `src/seccion_F`
Sistema automatizado para el manejo de casos especiales.
*   **Gestor de Excepciones:** Clasificación y enrutamiento de casos que requieren atención manual (`gestor_excepciones.py`).
*   **Workflows de Aprobación:** Automatización de flujos de trabajo y matrices de escalamiento (`workflow_excepciones.py`).
*   **Alertas de Complejidad:** Sistema de notificación basado en la complejidad del caso (`alertas_complejidad.py`).

## Estructura del Proyecto

```
AJEColombia/
├── data/               # Conjuntos de datos (CSV, JSON)
├── output/             # Reportes generados, gráficos y modelos exportados
├── src/                # Código fuente
│   ├── seccion_A/      # Anomalías
│   ├── seccion_B/      # Riesgo (ML)
│   ├── seccion_C/      # Proveedores
│   ├── seccion_D/      # JSON Processor
│   ├── seccion_E/      # IA / Gemini
│   ├── seccion_F/      # Excepciones
│   └── utils/          # Utilidades generales
├── docs/               # Documentación adicional
├── readme.md           # Este archivo
└── requirements.txt    # Dependencias del proyecto
```

## Requisitos

*   Python 3.8+
*   Librerías listadas en `requirements.txt` (pandas, plotly, scikit-learn, etc.)

---
Desarrollado para AJE Colombia.
