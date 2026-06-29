import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from data_loader import load_unified

app = dash.Dash(__name__)
df = load_unified()
KPI = 'spend' if 'spend' in df.columns else df.select_dtypes('number').columns[0]
GRP = next((c for c in df.columns if df[c].dtype.name == 'category' and df[c].nunique() < 10), df.columns[0])

app.layout = html.Div([
    html.H2('Growth dashboard'),
    dcc.Dropdown(id='grp', options=[{'label': c, 'value': c} for c in df[GRP].dropna().unique()],
                 value=df[GRP].dropna().unique()[0]),
    dcc.Graph(id='g'),
])

@app.callback(Output('g', 'figure'), Input('grp', 'value'))
def update(v):
    sub = df[df[GRP] == v]
    return px.line(sub, x='date' if 'date' in sub.columns else sub.columns[0], y=KPI)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
