import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import numpy as np
import pandas as pd
import mtt

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

rng = np.random.default_rng()

obj1 = np.array([[0], [0], [1], [1]])
obj2 = np.array([[1], [1], [1], [1]])
initial = {0:obj1}
dt = 1
ep_normal = 1
ep_tangent = 0.1
nu = 1
ts = 10
miss_p = 0.2
lam = 1
fa_scale = 5

gen = mtt.MultiObjSimple(initial, dt, ep_tangent, ep_normal, nu, miss_p, lam, fa_scale)
sim = mtt.Simulation(gen, mtt.KalmanFilter, mtt.Tracker)

sim.generate(ts)
sim.predict()

process = pd.DataFrame(sim.clean_process(sim.processes[0])[0].T[:, 0:2], columns = ["x","y"])
measure = pd.DataFrame(sim.clean_measure(sim.measures[0]).T, columns = ["x","y"])
trajectory = pd.DataFrame(sim.clean_trajectory(sim.trajectories[0])[0].T, columns = ["x","y"])

fig = px.line(process, x="x", y="y")
fig.add_trace(px.scatter(measure, x="x", y="y").data[0])
fig.add_trace(px.line(trajectory, x="x", y="y").data[0])


app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

app.run_server(debug=True)

