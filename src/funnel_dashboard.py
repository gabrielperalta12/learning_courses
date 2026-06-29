import sys
import pandas as pd
import dash
from dash import dcc, html, Input, Output

sys.path.append("src")
from charts.funnel import plot_funnel, plot_dropoff, FUNNEL_STEPS, STEP_LABELS
from charts.trends import plot_activations_over_time, plot_cvr_by_channel

df = pd.read_csv("data/raw/funnel_data.csv", parse_dates=["date"])

CHANNELS = ["Todos"] + sorted(df["channel"].unique().tolist())

app = dash.Dash(__name__)
app.title = "Funnel Colocación Tarjeta"

app.layout = html.Div(style={"fontFamily": "Inter, sans-serif", "backgroundColor": "#f8f9fa", "minHeight": "100vh", "padding": "24px"}, children=[

    html.H1("Funnel de Colocación de Tarjeta de Crédito",
            style={"color": "#1a1a2e", "marginBottom": "4px"}),
    html.P("Análisis de caída por etapa del proceso digital",
           style={"color": "#666", "marginBottom": "24px"}),

    html.Div(style={"display": "flex", "gap": "24px", "marginBottom": "24px", "flexWrap": "wrap"}, children=[
        html.Div([
            html.Label("Canal", style={"fontWeight": "600", "marginBottom": "6px", "display": "block"}),
            dcc.Dropdown(id="channel-filter",
                         options=[{"label": c.capitalize(), "value": c} for c in CHANNELS],
                         value="Todos", clearable=False, style={"width": "200px"}),
        ]),
        html.Div([
            html.Label("Rango de fechas", style={"fontWeight": "600", "marginBottom": "6px", "display": "block"}),
            dcc.DatePickerRange(
                id="date-filter",
                min_date_allowed=df["date"].min(),
                max_date_allowed=df["date"].max(),
                start_date=df["date"].min(),
                end_date=df["date"].max(),
            ),
        ]),
    ]),

    html.Div(id="kpi-cards", style={"display": "flex", "gap": "16px", "marginBottom": "24px", "flexWrap": "wrap"}),

    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"}, children=[
        html.Div(style={"backgroundColor": "white", "borderRadius": "12px", "padding": "20px", "boxShadow": "0 1px 4px rgba(0,0,0,0.08)"}, children=[
            html.H3("Funnel de Conversión", style={"marginTop": 0}),
            dcc.Graph(id="funnel-chart"),
        ]),
        html.Div(style={"backgroundColor": "white", "borderRadius": "12px", "padding": "20px", "boxShadow": "0 1px 4px rgba(0,0,0,0.08)"}, children=[
            html.H3("Caída por Etapa (%)", style={"marginTop": 0}),
            dcc.Graph(id="dropoff-chart"),
        ]),
    ]),

    html.Div(style={"backgroundColor": "white", "borderRadius": "12px", "padding": "20px", "boxShadow": "0 1px 4px rgba(0,0,0,0.08)", "marginBottom": "20px"}, children=[
        html.H3("Tendencia Temporal — Activaciones por Canal", style={"marginTop": 0}),
        dcc.Graph(id="trend-chart"),
    ]),

    html.Div(style={"backgroundColor": "white", "borderRadius": "12px", "padding": "20px", "boxShadow": "0 1px 4px rgba(0,0,0,0.08)"}, children=[
        html.H3("Conversión Total por Canal (Visita → Activación)", style={"marginTop": 0}),
        dcc.Graph(id="channel-cvr-chart"),
    ]),
])


def filter_data(channel, start_date, end_date):
    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
    filtered = df[mask]
    if channel != "Todos":
        filtered = filtered[filtered["channel"] == channel]
    return filtered


def kpi_card(title, value, subtitle="", color="#4361ee"):
    return html.Div(style={
        "backgroundColor": "white", "borderRadius": "12px", "padding": "16px 20px",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.08)", "borderLeft": f"4px solid {color}", "minWidth": "180px"
    }, children=[
        html.P(title, style={"margin": 0, "color": "#888", "fontSize": "13px"}),
        html.H2(value, style={"margin": "4px 0", "color": "#1a1a2e"}),
        html.P(subtitle, style={"margin": 0, "color": "#aaa", "fontSize": "12px"}),
    ])


@app.callback(
    Output("kpi-cards", "children"),
    Output("funnel-chart", "figure"),
    Output("dropoff-chart", "figure"),
    Output("trend-chart", "figure"),
    Output("channel-cvr-chart", "figure"),
    Input("channel-filter", "value"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
)
def update_all(channel, start_date, end_date):
    filtered = filter_data(channel, start_date, end_date)
    totals = filtered[FUNNEL_STEPS].sum()

    visits = totals["visita_landing"]
    activations = totals["activacion_tarjeta"]
    overall_cvr = activations / visits * 100 if visits > 0 else 0

    drops = {}
    for i in range(1, len(FUNNEL_STEPS)):
        prev = totals[FUNNEL_STEPS[i - 1]]
        curr = totals[FUNNEL_STEPS[i]]
        drops[FUNNEL_STEPS[i]] = (prev - curr) / prev * 100 if prev > 0 else 0
    worst_step = max(drops, key=drops.get)

    kpis = [
        kpi_card("Visitas", f"{int(visits):,}", "Inicio del funnel", "#4361ee"),
        kpi_card("Activaciones", f"{int(activations):,}", "Fin del funnel", "#06d6a0"),
        kpi_card("CVR Total", f"{overall_cvr:.1f}%", "Visita → Activación", "#f72585"),
        kpi_card("Mayor caída", STEP_LABELS[worst_step], f"{drops[worst_step]:.1f}% drop", "#ff9f1c"),
    ]

    trend_filtered = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]

    return (
        kpis,
        plot_funnel(filtered),
        plot_dropoff(filtered),
        plot_activations_over_time(trend_filtered),
        plot_cvr_by_channel(trend_filtered),
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)
