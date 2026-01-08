"""
Dashboard Interactivo de Análisis de Proveedores
=================================================
Dashboard construido con Dash y Plotly para visualización
interactiva del análisis de riesgo y segmentación de seccion_C.
"""

import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import sys

# Agregar el directorio src al path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analisis_proveedores import ejecutar_analisis_completo

# ============================================================
# CONFIGURACIÓN E INICIALIZACIÓN
# ============================================================

# Ejecutar análisis de seccion_C
resultados = ejecutar_analisis_completo()
df = resultados['datos_procesados']
matriz_aprobacion = resultados['matriz_aprobacion']
recomendaciones = resultados['recomendaciones_credito']
correlacion_empresa = resultados['correlacion_empresa']
patrones_geo = resultados['patrones_geograficos']
impacto_certs = resultados['impacto_certificaciones']

# Colores del tema (mismo estilo que dashboard_anomalias)
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

SEGMENT_COLORS = {
    'Premium': '#34a853',
    'Estándar': '#1a73e8',
    'Supervisión': '#fbbc04',
    'Alto_Riesgo': '#ea4335'
}

PERFORMANCE_COLORS = {
    'Excelente': '#34a853',
    'Bueno': '#1a73e8',
    'Regular': '#fbbc04',
    'Crítico': '#ea4335'
}

EMPRESA_COLORS = {
    'Grande': '#1a73e8',
    'Mediana': '#34a853',
    'Pequeña': '#fbbc04'
}

# ============================================================
# CREAR APLICACIÓN DASH
# ============================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}],
    title='Dashboard de Proveedores - Análisis de Riesgo'
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


def create_segmento_distribution_chart():
    """Gráfico de distribución por segmento de riesgo."""
    segment_counts = df['segmento_riesgo'].value_counts()
    
    fig = px.bar(
        x=segment_counts.index.tolist(),
        y=segment_counts.values,
        color=segment_counts.index.tolist(),
        color_discrete_map=SEGMENT_COLORS,
        labels={'x': 'Segmento', 'y': 'Cantidad'},
        title='Distribución por Segmento de Riesgo'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        title_font_size=16,
        showlegend=False,
        margin=dict(l=50, r=30, t=60, b=50)
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_performance_pie():
    """Gráfico de pastel de niveles de performance."""
    performance_counts = df['nivel_performance'].value_counts()
    
    fig = px.pie(
        values=performance_counts.values,
        names=performance_counts.index,
        title='Distribución por Nivel de Performance',
        color=performance_counts.index,
        color_discrete_map=PERFORMANCE_COLORS,
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


def create_empresa_performance_chart():
    """Gráfico de performance por tipo de empresa."""
    stats = df.groupby('tipo_empresa').agg({
        'score_performance': 'mean',
        'ID_proveedor': 'count'
    }).round(2).reset_index()
    stats.columns = ['Tipo Empresa', 'Performance Promedio', 'Cantidad']
    stats = stats.sort_values('Performance Promedio', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=stats['Performance Promedio'],
        y=stats['Tipo Empresa'],
        orientation='h',
        marker_color=[EMPRESA_COLORS.get(e, COLORS['primary']) for e in stats['Tipo Empresa']],
        text=stats.apply(lambda r: f"{r['Performance Promedio']:.1f} ({r['Cantidad']} prov.)", axis=1),
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Performance Promedio por Tipo de Empresa',
        xaxis_title='Score de Performance',
        yaxis_title='',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        title_font_size=16,
        margin=dict(l=100, r=30, t=60, b=50),
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_geographic_risk_chart():
    """Gráfico de riesgo por país."""
    stats = df.groupby('país').agg({
        'score_riesgo': 'mean',
        'score_performance': 'mean',
        'ID_proveedor': 'count'
    }).round(2).reset_index()
    stats.columns = ['País', 'Riesgo Promedio', 'Performance Promedio', 'Cantidad']
    stats = stats.sort_values('Riesgo Promedio', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=stats['Riesgo Promedio'],
        y=stats['País'],
        orientation='h',
        marker_color=stats['Riesgo Promedio'],
        marker_colorscale='RdYlGn_r',
        text=stats.apply(lambda r: f"{r['Riesgo Promedio']:.1f} ({r['Cantidad']} prov.)", axis=1),
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Score de Riesgo Promedio por País',
        xaxis_title='Score de Riesgo',
        yaxis_title='',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        title_font_size=16,
        margin=dict(l=100, r=30, t=60, b=50),
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_certificaciones_impact_chart():
    """Gráfico de impacto de certificaciones en performance."""
    stats = df.groupby('num_certificaciones').agg({
        'score_performance': 'mean',
        'score_riesgo': 'mean',
        'ID_proveedor': 'count'
    }).round(2).reset_index()
    stats.columns = ['Num Certificaciones', 'Performance', 'Riesgo', 'Cantidad']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=stats['Num Certificaciones'],
            y=stats['Performance'],
            name='Performance',
            marker_color=COLORS['success'],
            text=stats['Performance'].round(1),
            textposition='auto'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=stats['Num Certificaciones'],
            y=stats['Riesgo'],
            name='Riesgo',
            mode='lines+markers',
            line=dict(color=COLORS['danger'], width=3),
            marker=dict(size=10)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title='Impacto del Número de Certificaciones',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        title_font_size=16,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(title_text='Número de Certificaciones', gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(title_text='Score Performance', secondary_y=False, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(title_text='Score Riesgo', secondary_y=True, gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_credit_scatter():
    """Gráfico de dispersión: Crédito actual vs Recomendado."""
    fig = px.scatter(
        recomendaciones,
        x='límites_crédito',
        y='limite_recomendado',
        color='accion_credito',
        color_discrete_map={
            'Aumentar': COLORS['success'],
            'Mantener': COLORS['primary'],
            'Reducir': COLORS['danger']
        },
        hover_name='nombre',
        hover_data=['tipo_empresa', 'país', 'segmento_riesgo'],
        title='Límite de Crédito: Actual vs Recomendado',
        size_max=15
    )
    
    # Línea diagonal de referencia
    max_val = max(recomendaciones['límites_crédito'].max(), recomendaciones['limite_recomendado'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Sin cambio',
        line=dict(dash='dash', color=COLORS['muted'])
    ))
    
    fig.update_layout(
        xaxis_title='Límite Actual ($)',
        yaxis_title='Límite Recomendado ($)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        title_font_size=16,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=50, r=30, t=80, b=50)
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_segmento_empresa_heatmap():
    """Mapa de calor: Segmento vs Tipo Empresa."""
    cross = pd.crosstab(df['segmento_riesgo'], df['tipo_empresa'])
    
    fig = px.imshow(
        cross,
        labels=dict(x='Tipo Empresa', y='Segmento Riesgo', color='Cantidad'),
        title='Matriz: Segmento de Riesgo vs Tipo de Empresa',
        color_continuous_scale='Blues',
        text_auto=True
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        title_font_size=16,
        margin=dict(l=50, r=30, t=60, b=50)
    )
    
    return fig


def create_risk_performance_scatter():
    """Scatter plot de Riesgo vs Performance."""
    fig = px.scatter(
        df,
        x='score_performance',
        y='score_riesgo',
        color='segmento_riesgo',
        color_discrete_map=SEGMENT_COLORS,
        symbol='tipo_empresa',
        hover_name='nombre',
        hover_data=['país', 'num_certificaciones'],
        title='Mapa de Riesgo vs Performance',
        size_max=12
    )
    
    fig.update_layout(
        xaxis_title='Score de Performance',
        yaxis_title='Score de Riesgo',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        title_font_size=16,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=50, r=30, t=100, b=50)
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_acciones_credito_pie():
    """Gráfico de pastel de acciones de crédito recomendadas."""
    acciones = recomendaciones['accion_credito'].value_counts()
    
    fig = px.pie(
        values=acciones.values,
        names=acciones.index,
        title='Distribución de Acciones de Crédito Recomendadas',
        color=acciones.index,
        color_discrete_map={
            'Aumentar': COLORS['success'],
            'Mantener': COLORS['primary'],
            'Reducir': COLORS['danger']
        },
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


# ============================================================
# CALCULAR KPIs
# ============================================================

total_proveedores = len(df)
prov_alto_riesgo = len(df[df['segmento_riesgo'] == 'Alto_Riesgo'])
prov_premium = len(df[df['segmento_riesgo'] == 'Premium'])
performance_promedio = df['score_performance'].mean()
credito_total = df['límites_crédito'].sum()

# ============================================================
# LAYOUT DEL DASHBOARD
# ============================================================

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1([
                    html.I(className="bi bi-truck me-3"),
                    "Dashboard de Proveedores"
                ], className="mb-0", style={'fontWeight': 'bold'}),
                html.P("Análisis de Riesgo y Segmentación - Sección C", 
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
                "TOTAL PROVEEDORES",
                f"{total_proveedores:,}",
                "En el dataset",
                "building",
                COLORS['primary']
            )
        ], lg=3, md=6, className="mb-3"),
        dbc.Col([
            create_kpi_card(
                "PREMIUM",
                f"{prov_premium:,}",
                f"{prov_premium/total_proveedores*100:.1f}% del total",
                "award",
                COLORS['success']
            )
        ], lg=3, md=6, className="mb-3"),
        dbc.Col([
            create_kpi_card(
                "ALTO RIESGO",
                f"{prov_alto_riesgo:,}",
                f"{prov_alto_riesgo/total_proveedores*100:.1f}% del total",
                "exclamation-triangle",
                COLORS['danger']
            )
        ], lg=3, md=6, className="mb-3"),
        dbc.Col([
            create_kpi_card(
                "PERFORMANCE PROMEDIO",
                f"{performance_promedio:.1f}",
                "Score de 0 a 100",
                "speedometer2",
                COLORS['warning']
            )
        ], lg=3, md=6, className="mb-3"),
    ], className="mb-4"),
    
    # Charts Row 1: Segmentación y Performance
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_segmento_distribution_chart(),
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
                        figure=create_performance_pie(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], lg=6, className="mb-3"),
    ]),
    
    # Charts Row 2: Tipo Empresa y Geografía
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_empresa_performance_chart(),
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
                        figure=create_geographic_risk_chart(),
                        config={'displayModeBar': False},
                        style={'height': '350px'}
                    )
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], lg=6, className="mb-3"),
    ]),
    
    # Charts Row 3: Certificaciones y Mapa Riesgo/Performance
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_certificaciones_impact_chart(),
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
                        figure=create_risk_performance_scatter(),
                        config={'displayModeBar': False},
                        style={'height': '350px'}
                    )
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], lg=6, className="mb-3"),
    ]),
    
    # Charts Row 4: Crédito
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_credit_scatter(),
                        config={'displayModeBar': False},
                        style={'height': '350px'}
                    )
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], lg=7, className="mb-3"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_acciones_credito_pie(),
                        config={'displayModeBar': False},
                        style={'height': '350px'}
                    )
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], lg=5, className="mb-3"),
    ]),
    
    # Heatmap Segmento vs Empresa
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_segmento_empresa_heatmap(),
                        config={'displayModeBar': False},
                        style={'height': '350px'}
                    )
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], lg=12, className="mb-3"),
    ]),
    
    # Matriz de Aprobación Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        html.I(className="bi bi-clipboard-check me-2"),
                        "Matriz de Aprobación por Segmento"
                    ], className="mb-0")
                ], style={'backgroundColor': COLORS['card'], 'border': 'none'}),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='matriz-table',
                        columns=[
                            {'name': 'Segmento', 'id': 'segmento'},
                            {'name': 'Límite Crédito Máx', 'id': 'limite_credito_max', 'type': 'numeric', 
                             'format': {'specifier': '$,.0f'}},
                            {'name': 'Condiciones Pago', 'id': 'condiciones_pago'},
                            {'name': 'Descuento Máx %', 'id': 'descuento_maximo'},
                            {'name': 'Requiere Aprobación', 'id': 'requiere_aprobacion'},
                            {'name': 'Frecuencia Revisión', 'id': 'frecuencia_revision'},
                            {'name': 'Num Proveedores', 'id': 'num_proveedores'},
                            {'name': 'Performance Prom.', 'id': 'performance_promedio'},
                        ],
                        data=matriz_aprobacion.reset_index().rename(columns={'index': 'segmento'}).to_dict('records'),
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
                                'if': {'filter_query': '{segmento} = "Premium"'},
                                'backgroundColor': 'rgba(52, 168, 83, 0.2)',
                            },
                            {
                                'if': {'filter_query': '{segmento} = "Alto_Riesgo"'},
                                'backgroundColor': 'rgba(234, 67, 53, 0.2)',
                            },
                        ],
                        sort_action='native',
                        page_size=10
                    )
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], className="mb-4"),
    ]),
    
    # Top Proveedores para Aumentar/Reducir Crédito
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        html.I(className="bi bi-arrow-up-circle me-2", style={'color': COLORS['success']}),
                        "Top 10 - Aumentar Límite de Crédito"
                    ], className="mb-0")
                ], style={'backgroundColor': COLORS['card'], 'border': 'none'}),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='aumentar-table',
                        columns=[
                            {'name': 'Proveedor', 'id': 'nombre'},
                            {'name': 'Tipo Empresa', 'id': 'tipo_empresa'},
                            {'name': 'Segmento', 'id': 'segmento_riesgo'},
                            {'name': 'Límite Actual', 'id': 'límites_crédito', 'type': 'numeric', 
                             'format': {'specifier': '$,.0f'}},
                            {'name': 'Límite Recomendado', 'id': 'limite_recomendado', 'type': 'numeric', 
                             'format': {'specifier': '$,.0f'}},
                            {'name': 'Variación %', 'id': 'variacion_credito'},
                        ],
                        data=recomendaciones[recomendaciones['accion_credito'] == 'Aumentar'].nlargest(10, 'variacion_credito').to_dict('records'),
                        style_header={
                            'backgroundColor': COLORS['dark'],
                            'color': COLORS['text'],
                            'fontWeight': 'bold',
                            'textAlign': 'left',
                            'padding': '10px'
                        },
                        style_cell={
                            'backgroundColor': COLORS['card'],
                            'color': COLORS['text'],
                            'textAlign': 'left',
                            'padding': '10px',
                            'border': f'1px solid {COLORS["dark"]}'
                        },
                        page_size=10
                    )
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], lg=6, className="mb-4"),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        html.I(className="bi bi-arrow-down-circle me-2", style={'color': COLORS['danger']}),
                        "Top 10 - Reducir Límite de Crédito"
                    ], className="mb-0")
                ], style={'backgroundColor': COLORS['card'], 'border': 'none'}),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='reducir-table',
                        columns=[
                            {'name': 'Proveedor', 'id': 'nombre'},
                            {'name': 'Tipo Empresa', 'id': 'tipo_empresa'},
                            {'name': 'Segmento', 'id': 'segmento_riesgo'},
                            {'name': 'Límite Actual', 'id': 'límites_crédito', 'type': 'numeric', 
                             'format': {'specifier': '$,.0f'}},
                            {'name': 'Límite Recomendado', 'id': 'limite_recomendado', 'type': 'numeric', 
                             'format': {'specifier': '$,.0f'}},
                            {'name': 'Variación %', 'id': 'variacion_credito'},
                        ],
                        data=recomendaciones[recomendaciones['accion_credito'] == 'Reducir'].nsmallest(10, 'variacion_credito').to_dict('records'),
                        style_header={
                            'backgroundColor': COLORS['dark'],
                            'color': COLORS['text'],
                            'fontWeight': 'bold',
                            'textAlign': 'left',
                            'padding': '10px'
                        },
                        style_cell={
                            'backgroundColor': COLORS['card'],
                            'color': COLORS['text'],
                            'textAlign': 'left',
                            'padding': '10px',
                            'border': f'1px solid {COLORS["dark"]}'
                        },
                        page_size=10
                    )
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': 'none', 'borderRadius': '12px'})
        ], lg=6, className="mb-4"),
    ]),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(style={'borderColor': COLORS['muted']}),
            html.P([
                "Dashboard de Análisis de Proveedores | ",
                f"Crédito total en cartera: ${credito_total:,.0f}"
            ], className="text-muted text-center")
        ])
    ])
    
], fluid=True, style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'padding': '20px'})


# ============================================================
# EJECUTAR APLICACIÓN
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("DASHBOARD DE PROVEEDORES - ANÁLISIS DE RIESGO")
    print("="*60)
    print(f"\nProveedores analizados: {total_proveedores}")
    print(f"Proveedores Premium: {prov_premium}")
    print(f"Proveedores Alto Riesgo: {prov_alto_riesgo}")
    print(f"Performance promedio: {performance_promedio:.1f}")
    print(f"\nIniciando servidor en http://127.0.0.1:8051")
    print("="*60 + "\n")
    
    app.run(debug=True, port=8051)
