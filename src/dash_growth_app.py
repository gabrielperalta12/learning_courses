import os, pandas as pd, dash
from dash import dcc, html
from dash.dependencies import Input, Output

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
df = pd.read_csv("data/clean/unified_daily.csv")

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Growth dashboard"),
    dcc.Dropdown(id="ch",
                 options=[{"label": c, "value": c} for c in df["channel"].unique()],
                 value=df["channel"].unique()[0]),
    dcc.Graph(id="g"),
])

@app.callback(Output("g", "figure"), Input("ch", "value"))
def update(ch):
    import plotly.express as px
    return px.line(df[df["channel"] == ch], x="date", y="spend")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
