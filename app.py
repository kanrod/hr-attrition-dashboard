"""
Dashboard Final — IBM HR Analytics Employee Attrition
Programación para Ciencia de Datos II
Juan Camilo Rodríguez Fontecha
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, confusion_matrix,
                             classification_report)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. CARGA Y PREPARACIÓN DE DATOS
# ─────────────────────────────────────────────
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
df['Attrition_bin'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df['OverTime_bin']  = df['OverTime'].map({'Yes': 1, 'No': 0})

# ─── Contraste de hipótesis ───────────────────
g_yes = df[df['Attrition'] == 'Yes']['MonthlyIncome']
g_no  = df[df['Attrition'] == 'No']['MonthlyIncome']
stat_lev, p_lev = stats.levene(g_yes, g_no)
equal = p_lev > 0.05
t_stat, p_val = stats.ttest_ind(g_yes, g_no,
                                equal_var=equal,
                                alternative='two-sided')
prueba_usada = 'T de Student' if equal else 'T de Welch'

# ─── Regresión lineal ─────────────────────────
features_lin = ['TotalWorkingYears', 'Age', 'JobLevel']
X_lin = df[features_lin]; y_lin = df['MonthlyIncome']
X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(
    X_lin, y_lin, test_size=0.20, random_state=42)
model_lin = LinearRegression().fit(X_tr_l, y_tr_l)
y_pred_lin = model_lin.predict(X_te_l)
mse  = mean_squared_error(y_te_l, y_pred_lin)
rmse = np.sqrt(mse)
r2   = r2_score(y_te_l, y_pred_lin)

# ─── Regresión logística ──────────────────────
features_log = ['MonthlyIncome', 'YearsAtCompany', 'OverTime_bin']
X_log = df[features_log]; y_log = df['Attrition_bin']
X_tr_g, X_te_g, y_tr_g, y_te_g = train_test_split(
    X_log, y_log, test_size=0.20, random_state=42)
model_log = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
model_log.fit(X_tr_g, y_tr_g)
y_proba_g = model_log.predict_proba(X_te_g)

UMBRAL = 0.30
y_pred_g = (y_proba_g[:, 1] >= UMBRAL).astype(int)
acc_g  = accuracy_score(y_te_g, y_pred_g)
cm_g   = confusion_matrix(y_te_g, y_pred_g)
VP, FN = cm_g[1,1], cm_g[1,0]
FP, VN = cm_g[0,1], cm_g[0,0]
sens = VP / (VP + FN)
espec = VN / (VN + FP)
prec  = VP / (VP + FP) if (VP + FP) > 0 else 0

# ─────────────────────────────────────────────
# PALETA DE COLORES
# ─────────────────────────────────────────────
C_BLUE   = '#0F3460'
C_ACCENT = '#E94560'
C_LIGHT  = '#16213E'
C_GRAY   = '#A8DADC'
C_WHITE  = '#F5F5F5'
C_CARD   = '#1A1A2E'

# ─────────────────────────────────────────────
# FIGURAS PRE-CALCULADAS
# ─────────────────────────────────────────────

def make_attrition_pie():
    vals = df['Attrition'].value_counts()
    fig = go.Figure(go.Pie(
        labels=['Permanece', 'Renuncia'],
        values=[vals['No'], vals['Yes']],
        marker_colors=[C_BLUE, C_ACCENT],
        hole=0.45,
        textinfo='label+percent',
        textfont_size=13
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=C_WHITE,
        margin=dict(t=30, b=10, l=10, r=10),
        showlegend=False,
        height=260
    )
    return fig

def make_income_boxplot():
    fig = go.Figure()
    for grp, col, name in [('No', C_BLUE, 'Permanece'), ('Yes', C_ACCENT, 'Renuncia')]:
        fig.add_trace(go.Box(
            y=df[df['Attrition']==grp]['MonthlyIncome'],
            name=name, marker_color=col,
            boxmean=True, width=0.45
        ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=C_WHITE,
        yaxis=dict(title='Ingreso Mensual (USD)', gridcolor='#2a2a4a'),
        xaxis=dict(gridcolor='#2a2a4a'),
        margin=dict(t=20, b=30, l=50, r=20),
        height=300, showlegend=True
    )
    return fig

def make_coef_bar():
    coefs = model_lin.coef_
    fig = go.Figure(go.Bar(
        x=features_lin, y=coefs,
        marker_color=[C_BLUE, C_GRAY, C_ACCENT],
        text=[f'{c:.1f} USD' for c in coefs],
        textposition='outside'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=C_WHITE,
        yaxis=dict(title='Coeficiente β (USD)', gridcolor='#2a2a4a'),
        xaxis=dict(gridcolor='#2a2a4a'),
        margin=dict(t=20, b=30, l=50, r=20),
        height=280
    )
    return fig

def make_pred_vs_real():
    sample = min(200, len(y_te_l))
    idx = np.random.choice(len(y_te_l), sample, replace=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_te_l.values[idx], y=y_pred_lin[idx],
        mode='markers',
        marker=dict(color=C_ACCENT, opacity=0.6, size=5),
        name='Predicción'
    ))
    mn = min(y_te_l.min(), y_pred_lin.min())
    mx = max(y_te_l.max(), y_pred_lin.max())
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode='lines', line=dict(color=C_GRAY, dash='dash'),
        name='Línea ideal'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=C_WHITE,
        xaxis=dict(title='Real (USD)', gridcolor='#2a2a4a'),
        yaxis=dict(title='Predicho (USD)', gridcolor='#2a2a4a'),
        margin=dict(t=20, b=40, l=60, r=20),
        height=300
    )
    return fig

def make_confusion_matrix(threshold=0.30):
    y_pred_t = (y_proba_g[:, 1] >= threshold).astype(int)
    cm = confusion_matrix(y_te_g, y_pred_t)
    VP_, FN_ = cm[1,1], cm[1,0]
    FP_, VN_ = cm[0,1], cm[0,0]
    z = [[VN_, FP_], [FN_, VP_]]
    text = [[f'VN<br>{VN_}', f'FP<br>{FP_}'],
            [f'FN<br>{FN_}', f'VP<br>{VP_}']]
    fig = go.Figure(go.Heatmap(
        z=z, text=text, texttemplate='%{text}',
        colorscale=[[0, C_BLUE], [0.5, '#2a2a6a'], [1, C_ACCENT]],
        showscale=False, textfont_size=14
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=C_WHITE,
        xaxis=dict(tickvals=[0,1], ticktext=['Permanece (pred)', 'Renuncia (pred)']),
        yaxis=dict(tickvals=[0,1], ticktext=['Permanece (real)', 'Renuncia (real)']),
        margin=dict(t=20, b=50, l=120, r=20),
        height=270
    )
    return fig

def make_metrics_radar(threshold=0.30):
    y_pred_t = (y_proba_g[:, 1] >= threshold).astype(int)
    cm_t = confusion_matrix(y_te_g, y_pred_t)
    vp, fn = cm_t[1,1], cm_t[1,0]
    fp, vn = cm_t[0,1], cm_t[0,0]
    acc_  = accuracy_score(y_te_g, y_pred_t)
    sens_ = vp / (vp + fn) if (vp + fn) > 0 else 0
    esp_  = vn / (vn + fp) if (vn + fp) > 0 else 0
    prec_ = vp / (vp + fp) if (vp + fp) > 0 else 0
    cats  = ['Accuracy', 'Sensibilidad', 'Especificidad', 'Precisión']
    vals  = [acc_, sens_, esp_, prec_]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=cats + [cats[0]],
        fill='toself',
        fillcolor=f'rgba(233,69,96,0.3)',
        line=dict(color=C_ACCENT, width=2),
        name='Métricas'
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,1], color=C_WHITE, gridcolor='#333'),
            angularaxis=dict(color=C_WHITE, gridcolor='#333')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=C_WHITE,
        margin=dict(t=20, b=20, l=40, r=40),
        height=280, showlegend=False
    )
    return fig

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
prefix = os.environ.get('JUPYTERHUB_SERVICE_PREFIX', '/')
requests_pathname_prefix = f'{prefix}proxy/8050/'

app = dash.Dash(__name__, title='HR Attrition Dashboard',
                requests_pathname_prefix=requests_pathname_prefix)
server = app.server

CARD = {
    'backgroundColor': C_CARD,
    'borderRadius': '12px',
    'padding': '18px',
    'marginBottom': '16px',
    'boxShadow': '0 4px 15px rgba(0,0,0,0.4)'
}

METRIC_CARD = {
    **CARD,
    'textAlign': 'center',
    'flex': '1',
    'margin': '6px'
}

app.layout = html.Div(style={'backgroundColor': C_LIGHT, 'minHeight': '100vh',
                              'fontFamily': 'Segoe UI, Arial, sans-serif',
                              'color': C_WHITE}, children=[

    # ── HEADER ──────────────────────────────
    html.Div(style={'backgroundColor': C_BLUE, 'padding': '22px 32px',
                    'borderBottom': f'3px solid {C_ACCENT}'}, children=[
        html.H1('IBM HR Analytics — Rotación Laboral',
                style={'margin': 0, 'fontSize': '1.7rem', 'color': C_WHITE}),
        html.P('Programación para Ciencia de Datos II · Juan Camilo Rodríguez Fontecha',
               style={'margin': '4px 0 0', 'color': C_GRAY, 'fontSize': '0.85rem'})
    ]),

    # ── TABS ────────────────────────────────
    dcc.Tabs(id='tabs', value='tab-1', style={'backgroundColor': C_BLUE},
             colors={'border': C_ACCENT, 'primary': C_ACCENT, 'background': C_BLUE},
             children=[
        dcc.Tab(label='📊 Dataset', value='tab-1',
                style={'color': C_GRAY}, selected_style={'color': C_WHITE, 'backgroundColor': C_LIGHT}),
        dcc.Tab(label='🧪 Hipótesis', value='tab-2',
                style={'color': C_GRAY}, selected_style={'color': C_WHITE, 'backgroundColor': C_LIGHT}),
        dcc.Tab(label='📈 Regresión Lineal', value='tab-3',
                style={'color': C_GRAY}, selected_style={'color': C_WHITE, 'backgroundColor': C_LIGHT}),
        dcc.Tab(label='🎯 Regresión Logística', value='tab-4',
                style={'color': C_GRAY}, selected_style={'color': C_WHITE, 'backgroundColor': C_LIGHT}),
    ]),

    html.Div(id='tab-content', style={'padding': '24px 32px'})
])

# ─────────────────────────────────────────────
# CALLBACK TABS
# ─────────────────────────────────────────────
@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):

    if tab == 'tab-1':
        dept_options = [{'label': d, 'value': d} for d in sorted(df['Department'].unique())]
        dept_options.insert(0, {'label': 'Todos los departamentos', 'value': 'ALL'})
        return html.Div([
            html.Div(style={'display': 'flex', 'flexWrap': 'wrap'}, children=[
                html.Div(style=METRIC_CARD, children=[
                    html.H2('1,470', style={'color': C_ACCENT, 'margin': '0'}),
                    html.P('Total empleados', style={'margin': '4px 0 0', 'color': C_GRAY})
                ]),
                html.Div(style=METRIC_CARD, children=[
                    html.H2('16.1%', style={'color': C_ACCENT, 'margin': '0'}),
                    html.P('Tasa de renuncia', style={'margin': '4px 0 0', 'color': C_GRAY})
                ]),
                html.Div(style=METRIC_CARD, children=[
                    html.H2('35', style={'color': C_ACCENT, 'margin': '0'}),
                    html.P('Variables', style={'margin': '4px 0 0', 'color': C_GRAY})
                ]),
                html.Div(style=METRIC_CARD, children=[
                    html.H2('0', style={'color': C_ACCENT, 'margin': '0'}),
                    html.P('Valores nulos', style={'margin': '4px 0 0', 'color': C_GRAY})
                ]),
            ]),
            html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'}, children=[
                html.Div(style={**CARD, 'flex': '1', 'minWidth': '280px'}, children=[
                    html.H4('Distribución de Attrition', style={'margin': '0 0 8px', 'color': C_GRAY}),
                    dcc.Graph(figure=make_attrition_pie(), config={'displayModeBar': False})
                ]),
                html.Div(style={**CARD, 'flex': '2', 'minWidth': '340px'}, children=[
                    html.H4('Filtrar por Departamento', style={'margin': '0 0 8px', 'color': C_GRAY}),
                    dcc.Dropdown(id='dept-filter', options=dept_options, value='ALL',
                                 clearable=False,
                                 style={'backgroundColor': '#1a1a3e', 'color': C_WHITE, 'borderColor': C_BLUE}),
                    dcc.Graph(id='income-hist', config={'displayModeBar': False})
                ]),
            ]),
            html.Div(style=CARD, children=[
                html.H4('Estadísticas Descriptivas — Variables Clave',
                        style={'margin': '0 0 12px', 'color': C_GRAY}),
                dash_table.DataTable(
                    data=df[['Age','MonthlyIncome','YearsAtCompany',
                              'TotalWorkingYears','JobLevel']].describe().round(2).reset_index().to_dict('records'),
                    columns=[{'name': c, 'id': c} for c in
                              ['index','Age','MonthlyIncome','YearsAtCompany',
                               'TotalWorkingYears','JobLevel']],
                    style_table={'overflowX': 'auto'},
                    style_cell={'backgroundColor': C_CARD, 'color': C_WHITE,
                                'border': f'1px solid {C_BLUE}', 'padding': '8px'},
                    style_header={'backgroundColor': C_BLUE, 'color': C_WHITE, 'fontWeight': 'bold'},
                )
            ])
        ])

    elif tab == 'tab-2':
        rechaza = p_val < 0.05
        return html.Div([
            html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'}, children=[
                html.Div(style={**CARD, 'flex': '1', 'minWidth': '280px'}, children=[
                    html.H4('Resultado del Contraste', style={'margin': '0 0 12px', 'color': C_GRAY}),
                    html.Div(style={'backgroundColor': '#0a3020' if rechaza else '#3a0a0a',
                                    'borderRadius': '8px', 'padding': '14px',
                                    'borderLeft': f'4px solid {"#00cc77" if rechaza else C_ACCENT}'}, children=[
                        html.H3('✅ Se rechaza H₀' if rechaza else '❌ No se rechaza H₀',
                                style={'margin': '0 0 8px', 'color': '#00cc77' if rechaza else C_ACCENT}),
                        html.P(f'Prueba: {prueba_usada}', style={'margin': '2px 0'}),
                        html.P(f'Estadístico T: {t_stat:.4f}', style={'margin': '2px 0'}),
                        html.P(f'P-value: {p_val:.2e}', style={'margin': '2px 0', 'fontWeight': 'bold'}),
                        html.P(f'α = 0.05  |  Levene p = {p_lev:.4f}', style={'margin': '2px 0', 'color': C_GRAY}),
                    ]),
                    html.Div(style={'marginTop': '14px'}, children=[
                        html.P([html.B('H₀: '), 'μ_renuncia = μ_permanece'], style={'margin': '4px 0'}),
                        html.P([html.B('H₁: '), 'μ_renuncia ≠ μ_permanece'], style={'margin': '4px 0'}),
                        html.Hr(style={'borderColor': '#2a2a4a'}),
                        html.P(f'Media Renuncia:  ${g_yes.mean():,.0f} USD', style={'margin': '4px 0', 'color': C_ACCENT}),
                        html.P(f'Media Permanece: ${g_no.mean():,.0f} USD', style={'margin': '4px 0', 'color': C_GRAY}),
                        html.P(f'Diferencia:      ${g_no.mean()-g_yes.mean():,.0f} USD',
                               style={'margin': '4px 0', 'fontWeight': 'bold'}),
                    ])
                ]),
                html.Div(style={**CARD, 'flex': '2', 'minWidth': '340px'}, children=[
                    html.H4('Distribución de Ingreso por Grupo', style={'margin': '0 0 8px', 'color': C_GRAY}),
                    dcc.Graph(figure=make_income_boxplot(), config={'displayModeBar': False})
                ]),
            ]),
            html.Div(style=CARD, children=[
                html.H4('📝 Interpretación', style={'margin': '0 0 8px', 'color': C_GRAY}),
                html.P(f'Con un p-value de {p_val:.2e}, muy inferior a α = 0.05, existe evidencia '
                       f'estadística suficiente para rechazar H₀ con un nivel de confianza superior '
                       f'al 99.9%. La prueba {prueba_usada} confirma que los empleados '
                       f'que renuncian tienen un ingreso mensual promedio significativamente menor '
                       f'(${g_yes.mean():,.0f} USD) frente a los que permanecen (${g_no.mean():,.0f} USD).',
                       style={'lineHeight': '1.7', 'margin': 0})
            ])
        ])

    elif tab == 'tab-3':
        return html.Div([
            html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'}, children=[
                html.Div(style=METRIC_CARD, children=[
                    html.H2(f'{r2:.4f}', style={'color': C_ACCENT, 'margin': '0'}),
                    html.P('R² (test)', style={'margin': '4px 0 0', 'color': C_GRAY})
                ]),
                html.Div(style=METRIC_CARD, children=[
                    html.H2(f'${rmse:,.0f}', style={'color': C_ACCENT, 'margin': '0'}),
                    html.P('RMSE (USD)', style={'margin': '4px 0 0', 'color': C_GRAY})
                ]),
                html.Div(style=METRIC_CARD, children=[
                    html.H2('80/20', style={'color': C_ACCENT, 'margin': '0'}),
                    html.P('Train/Test split', style={'margin': '4px 0 0', 'color': C_GRAY})
                ]),
                html.Div(style=METRIC_CARD, children=[
                    html.H2('3', style={'color': C_ACCENT, 'margin': '0'}),
                    html.P('Predictores', style={'margin': '4px 0 0', 'color': C_GRAY})
                ]),
            ]),
            html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'}, children=[
                html.Div(style={**CARD, 'flex': '1', 'minWidth': '300px'}, children=[
                    html.H4('Coeficientes β del Modelo', style={'margin': '0 0 8px', 'color': C_GRAY}),
                    dcc.Graph(figure=make_coef_bar(), config={'displayModeBar': False}),
                    html.P(f'Intercepto (β₀): ${model_lin.intercept_:,.2f} USD',
                           style={'color': C_GRAY, 'margin': '4px 0 0', 'fontSize': '0.85rem'}),
                ]),
                html.Div(style={**CARD, 'flex': '1', 'minWidth': '300px'}, children=[
                    html.H4('Predicho vs Real (MonthlyIncome)', style={'margin': '0 0 8px', 'color': C_GRAY}),
                    dcc.Graph(figure=make_pred_vs_real(), config={'displayModeBar': False})
                ]),
            ]),
            html.Div(style=CARD, children=[
                html.H4('📝 Ecuación del Modelo', style={'margin': '0 0 8px', 'color': C_GRAY}),
                html.Code(
                    f'MonthlyIncome = {model_lin.intercept_:,.2f} '
                    f'+ {model_lin.coef_[0]:.2f}·TotalWorkingYears '
                    f'+ {model_lin.coef_[1]:.2f}·Age '
                    f'+ {model_lin.coef_[2]:.2f}·JobLevel',
                    style={'backgroundColor': '#0a0a1e', 'padding': '12px',
                           'borderRadius': '6px', 'display': 'block',
                           'color': '#7affb2', 'fontSize': '1rem'}
                ),
            ])
        ])

    elif tab == 'tab-4':
        coefs_log = model_log.coef_[0]
        intercept_log = model_log.intercept_[0]
        prob_base = 1 / (1 + np.exp(-intercept_log))

        return html.Div([
            html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'}, children=[
                html.Div(style=METRIC_CARD, children=[
                    html.H2(f'{acc_g:.1%}', style={'color': C_ACCENT, 'margin': '0'}),
                    html.P('Accuracy', style={'margin': '4px 0 0', 'color': C_GRAY})
                ]),
                html.Div(style=METRIC_CARD, children=[
                    html.H2(f'{sens:.1%}', style={'color': C_ACCENT, 'margin': '0'}),
                    html.P('Sensibilidad', style={'margin': '4px 0 0', 'color': C_GRAY})
                ]),
                html.Div(style=METRIC_CARD, children=[
                    html.H2(f'{espec:.1%}', style={'color': C_ACCENT, 'margin': '0'}),
                    html.P('Especificidad', style={'margin': '4px 0 0', 'color': C_GRAY})
                ]),
                html.Div(style=METRIC_CARD, children=[
                    html.H2(f'{prec:.1%}', style={'color': C_ACCENT, 'margin': '0'}),
                    html.P('Precisión (clase 1)', style={'margin': '4px 0 0', 'color': C_GRAY})
                ]),
            ]),
            html.Div(style=CARD, children=[
                html.H4('🎛️ Ajuste del Umbral de Clasificación', style={'margin': '0 0 8px', 'color': C_GRAY}),
                dcc.Slider(id='umbral-slider', min=0.10, max=0.60, step=0.05, value=0.30,
                           marks={v: f'{v:.2f}' for v in np.arange(0.10, 0.65, 0.10)},
                           tooltip={'placement': 'bottom'}),
            ]),
            html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'}, children=[
                html.Div(style={**CARD, 'flex': '1', 'minWidth': '300px'}, children=[
                    html.H4('Matriz de Confusión', style={'margin': '0 0 8px', 'color': C_GRAY}),
                    dcc.Graph(id='conf-matrix', figure=make_confusion_matrix(), config={'displayModeBar': False})
                ]),
                html.Div(style={**CARD, 'flex': '1', 'minWidth': '300px'}, children=[
                    html.H4('Métricas del Modelo', style={'margin': '0 0 8px', 'color': C_GRAY}),
                    dcc.Graph(id='radar-chart', figure=make_metrics_radar(), config={'displayModeBar': False})
                ]),
            ]),
            html.Div(style=CARD, children=[
                html.H4('🔮 Simulador de Probabilidad de Renuncia',
                        style={'margin': '0 0 14px', 'color': C_GRAY}),
                html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap',
                                'alignItems': 'flex-end'}, children=[
                    html.Div([
                        html.Label('Ingreso Mensual (USD)', style={'color': C_GRAY, 'fontSize': '0.85rem'}),
                        dcc.Input(id='sim-income', type='number', value=4500,
                                  style={'backgroundColor': '#0a0a2e', 'color': C_WHITE,
                                         'border': f'1px solid {C_BLUE}', 'borderRadius': '6px',
                                         'padding': '8px', 'width': '160px', 'display': 'block', 'marginTop': '4px'})
                    ]),
                    html.Div([
                        html.Label('Años en la Empresa', style={'color': C_GRAY, 'fontSize': '0.85rem'}),
                        dcc.Input(id='sim-years', type='number', value=2,
                                  style={'backgroundColor': '#0a0a2e', 'color': C_WHITE,
                                         'border': f'1px solid {C_BLUE}', 'borderRadius': '6px',
                                         'padding': '8px', 'width': '160px', 'display': 'block', 'marginTop': '4px'})
                    ]),
                    html.Div([
                        html.Label('Trabaja Horas Extra', style={'color': C_GRAY, 'fontSize': '0.85rem'}),
                        dcc.Dropdown(id='sim-overtime', options=[{'label': 'Sí', 'value': 1},
                                                                  {'label': 'No', 'value': 0}],
                                     value=1, clearable=False,
                                     style={'backgroundColor': '#0a0a2e', 'color': C_WHITE,
                                            'width': '160px', 'marginTop': '4px'})
                    ]),
                    html.Button('Calcular', id='sim-btn', n_clicks=0,
                                style={'backgroundColor': C_ACCENT, 'color': C_WHITE,
                                       'border': 'none', 'borderRadius': '6px',
                                       'padding': '10px 22px', 'cursor': 'pointer',
                                       'fontWeight': 'bold', 'fontSize': '0.95rem'}),
                ]),
                html.Div(id='sim-result', style={'marginTop': '16px'})
            ]),
            html.Div(style=CARD, children=[
                html.H4('Coeficientes del Modelo Logístico (log-odds)',
                        style={'margin': '0 0 8px', 'color': C_GRAY}),
                html.Div(style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap'}, children=[
                    html.Div(style={**METRIC_CARD, 'backgroundColor': '#0a0a2e'}, children=[
                        html.P('MonthlyIncome', style={'margin': '0', 'color': C_GRAY, 'fontSize': '0.8rem'}),
                        html.H4(f'{coefs_log[0]:.6f}', style={'margin': '4px 0 0', 'color': C_ACCENT})
                    ]),
                    html.Div(style={**METRIC_CARD, 'backgroundColor': '#0a0a2e'}, children=[
                        html.P('YearsAtCompany', style={'margin': '0', 'color': C_GRAY, 'fontSize': '0.8rem'}),
                        html.H4(f'{coefs_log[1]:.6f}', style={'margin': '4px 0 0', 'color': C_ACCENT})
                    ]),
                    html.Div(style={**METRIC_CARD, 'backgroundColor': '#0a0a2e'}, children=[
                        html.P('OverTime', style={'margin': '0', 'color': C_GRAY, 'fontSize': '0.8rem'}),
                        html.H4(f'{coefs_log[2]:.6f}', style={'margin': '4px 0 0', 'color': C_ACCENT})
                    ]),
                    html.Div(style={**METRIC_CARD, 'backgroundColor': '#0a0a2e'}, children=[
                        html.P('Intercepto (β₀)', style={'margin': '0', 'color': C_GRAY, 'fontSize': '0.8rem'}),
                        html.H4(f'{intercept_log:.4f}', style={'margin': '4px 0 0', 'color': C_GRAY})
                    ]),
                ]),
                html.P(f'Prob. base de renuncia (todos los X=0): {prob_base:.1%}',
                       style={'margin': '8px 0 0', 'color': C_GRAY, 'fontSize': '0.88rem'})
            ])
        ])


# ─────────────────────────────────────────────
# CALLBACKS INTERACTIVOS
# ─────────────────────────────────────────────

@app.callback(Output('income-hist', 'figure'), Input('dept-filter', 'value'))
def update_hist(dept):
    dff = df if dept == 'ALL' else df[df['Department'] == dept]
    fig = px.histogram(dff, x='MonthlyIncome', color='Attrition',
                       barmode='overlay', nbins=40,
                       color_discrete_map={'No': C_BLUE, 'Yes': C_ACCENT},
                       labels={'MonthlyIncome': 'Ingreso Mensual (USD)', 'count': 'Empleados'})
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=C_WHITE,
        legend_title='Attrition',
        xaxis=dict(gridcolor='#2a2a4a'),
        yaxis=dict(gridcolor='#2a2a4a'),
        margin=dict(t=10, b=40, l=50, r=20),
        height=240
    )
    return fig


@app.callback(
    Output('conf-matrix', 'figure'),
    Output('radar-chart', 'figure'),
    Input('umbral-slider', 'value')
)
def update_threshold_charts(threshold):
    return make_confusion_matrix(threshold), make_metrics_radar(threshold)


@app.callback(
    Output('sim-result', 'children'),
    Input('sim-btn', 'n_clicks'),
    State('sim-income', 'value'),
    State('sim-years', 'value'),
    State('sim-overtime', 'value'),
    prevent_initial_call=True
)
def simulate(n, income, years, overtime):
    if None in [income, years, overtime]:
        return html.P('⚠️ Completa todos los campos.', style={'color': C_ACCENT})
    X_sim = np.array([[income, years, overtime]])
    prob = model_log.predict_proba(X_sim)[0][1]
    nivel = ('🔴 ALTO RIESGO' if prob >= 0.5
             else '🟡 RIESGO MODERADO' if prob >= 0.30
             else '🟢 RIESGO BAJO')
    color = C_ACCENT if prob >= 0.5 else '#ffb300' if prob >= 0.30 else '#00cc77'
    return html.Div(style={'backgroundColor': '#0a0a1e', 'borderRadius': '10px',
                            'padding': '16px', 'borderLeft': f'4px solid {color}'}, children=[
        html.H3(f'Probabilidad de renuncia: {prob:.1%}',
                style={'margin': '0', 'color': color}),
        html.P(nivel, style={'margin': '6px 0 0', 'fontWeight': 'bold', 'fontSize': '1rem', 'color': color}),
        html.P(f'Perfil: Ingreso ${income:,} USD | {years} años | OverTime: {"Sí" if overtime else "No"}',
               style={'margin': '6px 0 0', 'color': C_GRAY, 'fontSize': '0.88rem'})
    ])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=False, use_reloader=False)
