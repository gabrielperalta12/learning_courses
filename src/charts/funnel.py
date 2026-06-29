import plotly.graph_objects as go
import plotly.express as px

STEP_LABELS = {
    "visita_landing":        "Visita Landing",
    "inicio_solicitud":      "Inicio Solicitud",
    "datos_personales":      "Datos Personales",
    "otp":                   "OTP",
    "datos_financieros":     "Datos Financieros",
    "carga_documentos":      "Carga Documentos",
    "evaluacion_crediticia": "Evaluación Crediticia",
    "aprobacion":            "Aprobación",
    "firma_digital":         "Firma Digital",
    "activacion_tarjeta":    "Activación Tarjeta",
}

FUNNEL_STEPS = list(STEP_LABELS.keys())


def plot_funnel(df):
    totals = df[FUNNEL_STEPS].sum()
    fig = go.Figure(go.Funnel(
        y=[STEP_LABELS[s] for s in FUNNEL_STEPS],
        x=[int(totals[s]) for s in FUNNEL_STEPS],
        textinfo="value+percent previous",
        marker={"color": px.colors.sequential.Blues_r[:len(FUNNEL_STEPS)]},
    ))
    fig.update_layout(margin={"t": 10, "b": 10}, height=380)
    return fig


def plot_dropoff(df):
    totals = df[FUNNEL_STEPS].sum()
    drops, labels = [], []
    for i in range(1, len(FUNNEL_STEPS)):
        prev = totals[FUNNEL_STEPS[i - 1]]
        curr = totals[FUNNEL_STEPS[i]]
        drop = (prev - curr) / prev * 100 if prev > 0 else 0
        drops.append(drop)
        labels.append(STEP_LABELS[FUNNEL_STEPS[i]])

    colors = ["#ef233c" if v > 30 else "#f4a261" if v > 15 else "#52b788" for v in drops]

    fig = go.Figure(go.Bar(
        x=labels,
        y=drops,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in drops],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis_title="% caída respecto al paso anterior",
        xaxis_tickangle=-35,
        margin={"t": 10, "b": 10},
        height=380,
        yaxis={"range": [0, max(drops) * 1.2]},
    )
    return fig
