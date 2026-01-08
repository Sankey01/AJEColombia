"""
Dashboard Interactivo de Anomalías en Condiciones de Pago
==========================================================
Dashboard construido con Dash y Plotly para visualización
interactiva de anomalías detectadas usando el AnomalyDetectorOrchestrator existente.
"""

import dash
from dash import dcc, html, dash_table, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import sys

# Agregar el directorio src al path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from seccion_A.analysis.anomaly_detector import AnomalyDetectorOrchestrator

# ============================================================
# CONFIGURACIÓN E INICIALIZACIÓN
# ============================================================

# Cargar datos
df_original = pd.read_csv(r"C:\Users\Kenny\PycharmProjects\AJEColombia\resource\condiciones_pagos.csv")

# Inicializar el orquestador existente
orchestrator = AnomalyDetectorOrchestrator()
df_consolidated = orchestrator.detect_all(df_original)
summary = orchestrator.get_summary()

# Colores del tema
COLORS = {
    'primary': '#1a73e8',
    'success': '#34a853',
    'warning': '#fbbc04',
    'danger': '#ea4335',
    'dark': '#202124',
    'light': '#f8f9fa',
    'background': '#1e1e2e',
    'card': '#2d2d3f',
    'text': '#ffffff',
    'muted': '#a0a0a0'
}

DETECTOR_COLORS = {
    'duplicate': '#ea4335',
    'format': '#ff6b35',
    'temporal': '#fbbc04',
    'business_rules': '#34a853',
    'approver': '#1a73e8',
    'amount': '#9c27b0',
    'discount_penalty': '#00bcd4',
    'cross_consistency': '#ff9800'
}

# Traducciones de nombres de detectores a español
DETECTOR_NAMES_ES = {
    'duplicate': 'Duplicados',
    'format': 'Formato',
    'temporal': 'Temporales',
    'business_rules': 'Reglas de Negocio',
    'approver': 'Aprobador',
    'amount': 'Montos',
    'discount_penalty': 'Descuento/Penalización',
    'cross_consistency': 'Consistencia Cruzada'
}

# Obtener total de duplicados
duplicate_stats = summary.get('detector_statistics', {}).get('duplicate', {})
TOTAL_DUPLICADOS = duplicate_stats.get('anomalies_found', 0)

# ============================================================
# CREAR APLICACIÓN DASH
# ============================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}],
    title='Dashboard de Anomalías - Condiciones de Pago'
)

server = app.server

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def create_kpi_card(title, value, subtitle, icon, color):
    """Crea una tarjeta KPI estilizada."""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f"bi bi-{icon}", style={'fontSize': '2.5rem', 'color': color}),
            ], style={'position': 'absolute', 'right': '20px', 'top': '20px', 'opacity': '0.3'}),
            html.H6(title, className="text-muted mb-2", style={'fontSize': '0.85rem'}),
            html.H2(value, className="mb-1", style={'fontWeight': 'bold', 'color': color}),
            html.P(subtitle, className="text-muted mb-0", style={'fontSize': '0.8rem'}),
        ], style={'position': 'relative', 'padding': '1.5rem'})
    ], style={
        'backgroundColor': COLORS['card'],
        'border': 'none',
        'borderRadius': '12px',
        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.3)',
        'height': '100%'
    })


def get_detector_name_es(detector_key: str) -> str:
    """Obtiene el nombre del detector en español."""
    return DETECTOR_NAMES_ES.get(detector_key, detector_key.replace('_', ' ').title())


def create_detector_distribution_chart():
    """Gráfico de distribución de anomalías por detector."""
    detector_stats = summary.get('detector_statistics', {})
    
    data = []
    for detector_name, stats in detector_stats.items():
        data.append({
            'Detector': get_detector_name_es(detector_name),
            'DetectorKey': detector_name,
            'Anomalías': stats.get('anomalies_found', 0),
            'Porcentaje': stats.get('anomaly_percentage', 0)
        })
    
    df_chart = pd.DataFrame(data)
    
    if len(df_chart) == 0:
        return go.Figure()
    
    df_chart = df_chart.sort_values('Anomalías', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df_chart['Anomalías'],
        y=df_chart['Detector'],
        orientation='h',
        marker_color=[DETECTOR_COLORS.get(d, COLORS['primary']) 
                      for d in df_chart['DetectorKey']],
        text=df_chart['Anomalías'],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Anomalías por Tipo de Detector',
        xaxis_title='Cantidad de Anomalías',
        yaxis_title='',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        title_font_size=16,
        margin=dict(l=180, r=30, t=60, b=50),
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_anomaly_distribution_pie():
    """Gráfico de pastel de distribución de anomalías."""
    anomaly_dist = summary.get('anomaly_distribution', {})
    
    data = []
    for num_anomalies, count in anomaly_dist.items():
        if int(num_anomalies) > 0:
            data.append({
                'Anomalías': f'{num_anomalies} anomalía(s)',
                'Registros': count
            })
    
    if not data:
        return go.Figure()
    
    df_pie = pd.DataFrame(data)
    
    fig = px.pie(
        df_pie,
        values='Registros',
        names='Anomalías',
        title='Distribución de Registros por Cantidad de Anomalías',
        color_discrete_sequence=px.colors.sequential.Reds_r,
        hole=0.4
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        title_font_size=16,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2),
        margin=dict(l=20, r=20, t=60, b=60)
    )
    
    return fig


def create_amount_histogram():
    """Histograma de montos con anomalías resaltadas."""
    fig = go.Figure()
    
    # Histograma de todos los montos
    fig.add_trace(go.Histogram(
        x=df_consolidated['monto'],
        name='Todas las transacciones',
        marker_color=COLORS['primary'],
        opacity=0.7,
        nbinsx=50
    ))
    
    # Overlay de registros con anomalías
    anomalies = df_consolidated[df_consolidated['has_any_anomaly'] == True]
    if len(anomalies) > 0:
        fig.add_trace(go.Histogram(
            x=anomalies['monto'],
            name='Con anomalías',
            marker_color=COLORS['danger'],
            opacity=0.8,
            nbinsx=50
        ))
    
    fig.update_layout(
        title='Distribución de Montos',
        xaxis_title='Monto ($)',
        yaxis_title='Frecuencia',
        barmode='overlay',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        title_font_size=16,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=30, t=80, b=50)
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_approver_analysis():
    """Análisis de aprobadores - Monto total excluyendo duplicados."""
    # Identificar la columna de duplicados si existe
    duplicate_col = None
    for col in df_consolidated.columns:
        if 'duplicate' in col.lower() and 'has_anomaly' in col.lower():
            duplicate_col = col
            break
    
    # Filtrar registros NO duplicados para calcular montos
    if duplicate_col and duplicate_col in df_consolidated.columns:
        df_no_duplicates = df_consolidated[df_consolidated[duplicate_col] == False]
    else:
        df_no_duplicates = df_consolidated
    
    # Agrupar por aprobador en datos SIN duplicados
    approver_data = df_no_duplicates.groupby('aprobador').agg({
        'ID_transaccion': 'count',
        'monto': 'sum',
        'has_any_anomaly': 'sum'
    }).reset_index()
    approver_data.columns = ['Aprobador', 'Transacciones', 'Monto Total', 'Con Anomalías']
    approver_data = approver_data.sort_values('Monto Total', ascending=True)
    
    # Crear gráfico de barras horizontal con montos
    fig = go.Figure()
    
    # Barra de monto total
    fig.add_trace(go.Bar(
        x=approver_data['Monto Total'],
        y=approver_data['Aprobador'],
        orientation='h',
        name='Monto Total',
        marker_color=COLORS['primary'],
        text=approver_data.apply(
            lambda row: f"${row['Monto Total']:,.0f} ({row['Transacciones']} trans.)", axis=1
        ),
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Monto: $%{x:,.2f}<br>Transacciones: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Monto Total por Aprobador (Sin Duplicados)',
        xaxis_title='Monto Total ($)',
        yaxis_title='Aprobador',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        title_font_size=16,
        margin=dict(l=120, r=30, t=60, b=50),
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_timeline_chart():
    """Gráfico de línea temporal de anomalías."""
    df_timeline = df_consolidated.copy()
    df_timeline['fecha_factura'] = pd.to_datetime(df_timeline['fecha_factura'])
    df_timeline['mes'] = df_timeline['fecha_factura'].dt.to_period('M').astype(str)
    
    # Solo registros con anomalías
    anomalies = df_timeline[df_timeline['has_any_anomaly'] == True]
    
    timeline_data = anomalies.groupby('mes').agg({
        'ID_transaccion': 'count',
        'total_anomalies': 'sum'
    }).reset_index()
    timeline_data.columns = ['Mes', 'Registros', 'Total Anomalías']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timeline_data['Mes'],
        y=timeline_data['Registros'],
        mode='lines+markers',
        name='Registros con anomalías',
        line=dict(color=COLORS['danger'], width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Evolución Temporal de Registros con Anomalías',
        xaxis_title='Mes',
        yaxis_title='Cantidad de Registros',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        title_font_size=16,
        margin=dict(l=50, r=30, t=60, b=80),
        xaxis={'tickangle': -45}
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_payment_status_chart():
    """Gráfico de estado de pagos."""
    status_data = df_consolidated.groupby('estado_pago').agg({
        'ID_transaccion': 'count',
        'has_any_anomaly': 'sum'
    }).reset_index()
    status_data.columns = ['Estado', 'Total', 'Con Anomalías']
    status_data['Sin Anomalías'] = status_data['Total'] - status_data['Con Anomalías']
    
    status_colors = {
        'Pagado': '#34a853',
        'Pendiente': '#fbbc04',
        'Retrasado': '#ea4335'
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Sin Anomalías',
        x=status_data['Estado'],
        y=status_data['Sin Anomalías'],
        marker_color=[status_colors.get(s, COLORS['primary']) for s in status_data['Estado']],
        opacity=0.6
    ))
    
    fig.add_trace(go.Bar(
        name='Con Anomalías',
        x=status_data['Estado'],
        y=status_data['Con Anomalías'],
        marker_color=COLORS['danger']
    ))
    
    fig.update_layout(
        title='Estado de Pagos vs Anomalías',
        xaxis_title='Estado',
        yaxis_title='Cantidad',
        barmode='stack',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        title_font_size=16,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=30, t=80, b=50)
    )
    
    return fig


def create_detector_breakdown_chart():
    """Desglose detallado por detector."""
    detector_stats = summary.get('detector_statistics', {})
    
    breakdown_data = []
    for detector_name, stats in detector_stats.items():
        breakdown = stats.get('breakdown', {})
        for sub_type, count in breakdown.items():
            breakdown_data.append({
                'Detector': detector_name.replace('_', ' ').title(),
                'Tipo': sub_type,
                'Cantidad': count
            })
    
    if not breakdown_data:
        # Si no hay breakdown, mostrar solo totales
        for detector_name, stats in detector_stats.items():
            breakdown_data.append({
                'Detector': detector_name.replace('_', ' ').title(),
                'Tipo': 'Total',
                'Cantidad': stats.get('anomalies_found', 0)
            })
    
    df_breakdown = pd.DataFrame(breakdown_data)
    
    if len(df_breakdown) == 0:
        return go.Figure()
    
    fig = px.sunburst(
        df_breakdown,
        path=['Detector', 'Tipo'],
        values='Cantidad',
        title='Desglose de Anomalías por Detector y Tipo',
        color='Detector',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        title_font_size=16,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_top_anomalies_table():
    """Obtiene los registros con más anomalías."""
    top_anomalies = orchestrator.get_top_anomalies(20)
    
    # Seleccionar columnas relevantes
    display_cols = ['ID_transaccion', 'proveedor_id', 'monto', 'estado_pago', 
                    'aprobador', 'total_anomalies']
    
    available_cols = [col for col in display_cols if col in top_anomalies.columns]
    
    return top_anomalies[available_cols].to_dict('records')


def create_risk_matrix():
    """Genera matriz de riesgo."""
    detector_stats = summary.get('detector_statistics', {})
    
    risk_config = {
        'duplicate': {'impacto': 3, 'probabilidad': 2, 'descripcion': 'Transacciones duplicadas'},
        'format': {'impacto': 1, 'probabilidad': 3, 'descripcion': 'Errores de formato'},
        'temporal': {'impacto': 2, 'probabilidad': 2, 'descripcion': 'Inconsistencias temporales'},
        'business_rules': {'impacto': 3, 'probabilidad': 2, 'descripcion': 'Violación reglas de negocio'},
        'approver': {'impacto': 4, 'probabilidad': 2, 'descripcion': 'Anomalías de aprobador'},
        'amount': {'impacto': 4, 'probabilidad': 1, 'descripcion': 'Montos atípicos'},
        'discount_penalty': {'impacto': 2, 'probabilidad': 2, 'descripcion': 'Descuentos/penalizaciones'},
        'cross_consistency': {'impacto': 3, 'probabilidad': 3, 'descripcion': 'Inconsistencias cruzadas'}
    }
    
    risk_data = []
    for detector_name, stats in detector_stats.items():
        config = risk_config.get(detector_name, {'impacto': 2, 'probabilidad': 2, 'descripcion': detector_name})
        cantidad = stats.get('anomalies_found', 0)
        
        risk_data.append({
            'Tipo': get_detector_name_es(detector_name),
            'Cantidad': cantidad,
            'Impacto': config['impacto'],
            'Probabilidad': config['probabilidad'],
            'Riesgo': config['impacto'] * config['probabilidad'],
            'Descripción': config['descripcion']
        })
    
    return pd.DataFrame(risk_data)


def create_risk_matrix_heatmap():
    """Mapa de calor de matriz de riesgo."""
    risk_df = create_risk_matrix()
    
    fig = px.scatter(
        risk_df,
        x='Probabilidad',
        y='Impacto',
        size='Cantidad',
        color='Riesgo',
        color_continuous_scale='RdYlGn_r',
        hover_name='Tipo',
        hover_data=['Cantidad', 'Descripción'],
        title='Matriz de Riesgo por Tipo de Anomalía',
        size_max=50
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        title_font_size=16,
        xaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3, 4],
            ticktext=['Baja', 'Media', 'Alta', 'Muy Alta'],
            title='Probabilidad',
            range=[0.5, 4.5]
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3, 4],
            ticktext=['Bajo', 'Medio', 'Alto', 'Muy Alto'],
            title='Impacto',
            range=[0.5, 4.5]
        ),
        margin=dict(l=50, r=30, t=60, b=50)
    )
    
    # Agregar cuadrantes de riesgo
    fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=2.5, y1=2.5,
                  line=dict(color="rgba(52, 168, 83, 0.3)", width=2),
                  fillcolor="rgba(52, 168, 83, 0.1)")
    fig.add_shape(type="rect", x0=2.5, y0=2.5, x1=4.5, y1=4.5,
                  line=dict(color="rgba(234, 67, 53, 0.3)", width=2),
                  fillcolor="rgba(234, 67, 53, 0.1)")
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.2)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.2)')
    
    return fig


# ============================================================
# LAYOUT DEL DASHBOARD
# ============================================================

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1([
                    html.I(className="bi bi-shield-exclamation me-3"),
                    "Dashboard de Anomalías"
                ], className="mb-0", style={'fontWeight': 'bold'}),
                html.P("Análisis de Condiciones de Pago - AnomalyDetectorOrchestrator", 
                       className="text-muted mb-0")
            ], style={'padding': '1.5rem 0'})
        ], width=8),
        dbc.Col([
            html.Div([
                html.Span("Última actualización: ", className="text-muted"),
                html.Span(pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"), 
                         style={'color': COLORS['success']})
            ], className="text-end pt-4")
        ], width=4)
    ], className="mb-4"),
    
    # KPIs Row
    dbc.Row([
        dbc.Col([
            create_kpi_card(
                "TOTAL TRANSACCIONES",
                f"{summary.get('total_records', 0):,}",
                "En el dataset",
                "database",
                COLORS['primary']
            )
        ], lg=3, md=6, className="mb-3"),
        dbc.Col([
            create_kpi_card(
                "CON ANOMALÍAS",
                f"{summary.get('records_with_anomalies', 0):,}",
                f"{summary.get('anomaly_percentage', 0):.1f}% del total",
                "exclamation-triangle",
                COLORS['danger']
            )
        ], lg=3, md=6, className="mb-3"),
        dbc.Col([
            create_kpi_card(
                "REGISTROS LIMPIOS",
                f"{summary.get('records_clean', 0):,}",
                f"{summary.get('clean_percentage', 0):.1f}% del total",
                "check-circle",
                COLORS['success']
            )
        ], lg=3, md=6, className="mb-3"),
        dbc.Col([
            create_kpi_card(
                "DETECTORES ACTIVOS",
                f"{len(orchestrator.detectors)}",
                "Tipos de anomalías",
                "gear",
                COLORS['warning']
            )
        ], lg=3, md=6, className="mb-3"),
    ], className="mb-4"),
    
    # Charts Row 1
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_detector_distribution_chart(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], lg=6, className="mb-3"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_anomaly_distribution_pie(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], lg=6, className="mb-3"),
    ]),
    
    # Charts Row 2
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_amount_histogram(),
                        config={'displayModeBar': False},
                        style={'height': '350px'}
                    )
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], lg=6, className="mb-3"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_approver_analysis(),
                        config={'displayModeBar': False},
                        style={'height': '350px'}
                    )
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], lg=6, className="mb-3"),
    ]),
    
    # Charts Row 3
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_timeline_chart(),
                        config={'displayModeBar': False},
                        style={'height': '350px'}
                    )
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], lg=6, className="mb-3"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_risk_matrix_heatmap(),
                        config={'displayModeBar': False},
                        style={'height': '350px'}
                    )
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], lg=6, className="mb-3"),
    ]),
    
    # Charts Row 4  
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_payment_status_chart(),
                        config={'displayModeBar': False},
                        style={'height': '300px'}
                    )
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], lg=5, className="mb-3"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_detector_breakdown_chart(),
                        config={'displayModeBar': False},
                        style={'height': '300px'}
                    )
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], lg=7, className="mb-3"),
    ]),
    
    # Matriz de Riesgo Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        html.I(className="bi bi-table me-2"),
                        "Matriz de Riesgo Detallada"
                    ], className="mb-0")
                ], style={'backgroundColor': COLORS['card'], 'border': 'none'}),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='risk-table',
                        columns=[
                            {'name': col, 'id': col} for col in create_risk_matrix().columns
                        ],
                        data=create_risk_matrix().to_dict('records'),
                        style_header={
                            'backgroundColor': COLORS['dark'],
                            'color': COLORS['text'],
                            'fontWeight': 'bold',
                            'textAlign': 'left',
                            'padding': '12px'
                        },
                        style_cell={
                            'backgroundColor': COLORS['card'],
                            'color': COLORS['text'],
                            'textAlign': 'left',
                            'padding': '12px',
                            'border': f'1px solid {COLORS["dark"]}'
                        },
                        style_data_conditional=[
                            {
                                'if': {'filter_query': '{Riesgo} >= 9'},
                                'backgroundColor': 'rgba(234, 67, 53, 0.3)',
                            },
                            {
                                'if': {'filter_query': '{Riesgo} >= 6 && {Riesgo} < 9'},
                                'backgroundColor': 'rgba(255, 107, 53, 0.3)',
                            },
                            {
                                'if': {'filter_query': '{Riesgo} < 6'},
                                'backgroundColor': 'rgba(251, 188, 4, 0.2)',
                            },
                        ],
                        sort_action='native',
                        page_size=10
                    )
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], className="mb-4"),
    ]),
    
    # Tabla de Top Anomalías
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        html.I(className="bi bi-list-ol me-2"),
                        "Top 20 Registros con Más Anomalías"
                    ], className="mb-0")
                ], style={'backgroundColor': COLORS['card'], 'border': 'none'}),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='top-anomalies-table',
                        columns=[
                            {'name': 'ID Transacción', 'id': 'ID_transaccion'},
                            {'name': 'Proveedor', 'id': 'proveedor_id'},
                            {'name': 'Monto', 'id': 'monto', 'type': 'numeric', 
                             'format': {'specifier': '$,.2f'}},
                            {'name': 'Estado', 'id': 'estado_pago'},
                            {'name': 'Aprobador', 'id': 'aprobador'},
                            {'name': 'Total Anomalías', 'id': 'total_anomalies'},
                        ],
                        data=create_top_anomalies_table(),
                        style_header={
                            'backgroundColor': COLORS['dark'],
                            'color': COLORS['text'],
                            'fontWeight': 'bold',
                            'textAlign': 'left',
                            'padding': '12px'
                        },
                        style_cell={
                            'backgroundColor': COLORS['card'],
                            'color': COLORS['text'],
                            'textAlign': 'left',
                            'padding': '10px',
                            'border': f'1px solid {COLORS["dark"]}'
                        },
                        style_data_conditional=[
                            {
                                'if': {'filter_query': '{total_anomalies} >= 5'},
                                'backgroundColor': 'rgba(234, 67, 53, 0.4)',
                            },
                            {
                                'if': {'filter_query': '{total_anomalies} >= 3 && {total_anomalies} < 5'},
                                'backgroundColor': 'rgba(255, 107, 53, 0.3)',
                            },
                        ],
                        sort_action='native',
                        page_size=10
                    )
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], className="mb-4"),
    ]),
    
    # Estadísticas por Detector
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        html.I(className="bi bi-bar-chart me-2"),
                        "Estadísticas Detalladas por Detector"
                    ], className="mb-0")
                ], style={'backgroundColor': COLORS['card'], 'border': 'none'}),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6(detector_name.replace('_', ' ').title(), 
                                           className="text-primary mb-2"),
                                    html.P([
                                        html.Strong(f"{stats.get('anomalies_found', 0):,}"),
                                        html.Span(f" anomalías ({stats.get('anomaly_percentage', 0):.1f}%)", 
                                                 className="text-muted")
                                    ], className="mb-1"),
                                    html.Small(
                                        f"Desglose: {', '.join([f'{k}: {v}' for k, v in stats.get('breakdown', {}).items()][:3])}" 
                                        if stats.get('breakdown') else "Sin desglose disponible",
                                        className="text-muted"
                                    )
                                ])
                            ], style={'backgroundColor': COLORS['dark'], 'border': 'none', 
                                     'borderLeft': f'4px solid {DETECTOR_COLORS.get(detector_name, COLORS["primary"])}'})
                        ], lg=3, md=6, className="mb-3")
                        for detector_name, stats in summary.get('detector_statistics', {}).items()
                    ])
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], className="mb-4"),
    ]),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(style={'borderColor': COLORS['muted']}),
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                f"Dashboard generado usando AnomalyDetectorOrchestrator. ",
                f"Total de {summary.get('total_records', 0):,} transacciones analizadas con ",
                f"{len(orchestrator.detectors)} detectores especializados."
            ], className="text-muted text-center small")
        ])
    ])
    
], fluid=True, style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'padding': '2rem'})


# ============================================================
# EJECUTAR SERVIDOR
# ============================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DASHBOARD DE ANOMALÍAS - CONDICIONES DE PAGO")
    print("Usando: AnomalyDetectorOrchestrator")
    print("=" * 60)
    print(f"Total transacciones: {summary.get('total_records', 0):,}")
    print(f"Con anomalías: {summary.get('records_with_anomalies', 0):,}")
    print(f"Porcentaje: {summary.get('anomaly_percentage', 0):.2f}%")
    print("=" * 60)
    print("\nIniciando servidor en http://127.0.0.1:8050")
    print("Presiona Ctrl+C para detener\n")
    
    app.run(debug=True, host='127.0.0.1', port=8050)
