import plotly.express as px
import plotly.graph_objects as go


def plot_activations_over_time(df):
    trend = df.groupby(["date", "channel"])["activacion_tarjeta"].sum().reset_index()
    fig = px.line(
        trend, x="date", y="activacion_tarjeta", color="channel",
        labels={"activacion_tarjeta": "Activaciones", "date": "Fecha", "channel": "Canal"},
        template="simple_white",
    )
    fig.update_layout(margin={"t": 10, "b": 10}, height=280)
    return fig


def plot_cvr_by_channel(df):
    rows = []
    for ch in df["channel"].unique():
        ch_df = df[df["channel"] == ch]
        v = ch_df["visita_landing"].sum()
        a = ch_df["activacion_tarjeta"].sum()
        rows.append({"canal": ch.capitalize(), "cvr": a / v * 100 if v > 0 else 0})

    import pandas as pd
    ch_df = pd.DataFrame(rows).sort_values("cvr", ascending=True)

    fig = go.Figure(go.Bar(
        x=ch_df["cvr"], y=ch_df["canal"], orientation="h",
        text=[f"{v:.2f}%" for v in ch_df["cvr"]],
        textposition="outside",
        marker_color="#4361ee",
    ))
    fig.update_layout(
        xaxis_title="CVR (%)",
        margin={"t": 10, "b": 10},
        height=250,
        xaxis={"range": [0, ch_df["cvr"].max() * 1.2]},
        template="simple_white",
    )
    return fig
