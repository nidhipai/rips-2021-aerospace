import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import mtt
from copy import copy

global sim
global prev_clicks
global num_objects
prev_clicks = 0
num_objects = 1

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

rng = np.random.default_rng()
# Default simulation values
obj1 = np.array([[0], [0], [1], [1]])
initial = {0: obj1}
dt = 1
ep_normal = 0
ep_tangent = 1
nu = 0.01
ts = 10
miss_p = 0
lam = 0
fa_scale = 1
gate_size = 10
gate_expand_size = 90

# Style Parameters
input_margin = 10
input_style = {"display": "inline-block", "margin": input_margin}

gen = mtt.MultiObjSimple(initial, dt, ep_tangent, ep_normal, nu, miss_p, lam, fa_scale)
gate = mtt.DistanceGating(gate_size, expand_gating=gate_expand_size, method="euclidean")
assoc = mtt.DataAssociation()
params = gen.get_params()
maintain = mtt.TrackMaintenance(mtt.KalmanFilter, params, num_obj = num_objects, num_init = 2, num_init_frames=3, num_delete=3)
filter_ = mtt.FilterPredict()
methods = [gate, assoc, maintain, filter_]

sim = mtt.Simulation(gen, mtt.KalmanFilter, mtt.MTTTracker, methods)

fig = go.Figure()
err = go.Figure()

app.layout = html.Div(children=[
    html.H1(children='2D Object Trajectory Tracking'),

    html.Div(children='''
        This dashboard generates sample object movement in two directions and 
        uses the RIPS Aerospace Tracker to plot the predicted object trajectories.
    '''),

    html.Button('Run Simulation', id='run', n_clicks=0),
    dcc.Checklist(id='check-options',
        options=[
            {'label': 'Process', 'value': 'process'},
            {'label': 'Measure', 'value': 'measure'},
            {'label': 'Trajectory', 'value': 'trajectory'},
            {'label': 'A Priori Error Covariance', 'value': 'apriori-covariance'},
            {'label': 'A Posteriori Error Covariance', 'value': 'aposteriori-covariance'}
        ],
        value=['process', 'measure'],
        labelStyle={'display': 'inline-block'}
    ),
    html.Div(children=[
        dcc.Graph(
            id='example-graph',
            figure=fig
        )
    ], style={"display":"inline-block"}),

    html.Div(children=[
        dcc.Graph(
            id='error-graph',
            figure=err
        )
    ], style={"display":"inline-block"}),


    html.Div(children=[
        html.H3(children="Data Generation Parameters"),
        html.Div(children=[
            html.H6(children='Time Steps'),
            dcc.Input(
                id="time-steps",
                type="number",
                min=1,
                max=1000,
                step=1,
                placeholder=10
            )
        ], style=input_style),

        html.Div(children=[
            html.H6(children='Measure Noise'),
            dcc.Input(
                id="nu",
                type="number",
                min=0.0000001,
                max=100,
                placeholder=1
            )
        ], style=input_style),

        html.Div(children=[
            html.H6(children='Along-Track Noise'),
            dcc.Input(
                id="ep_tangent",
                type="number",
                min=0,
                max=100,
                placeholder=1
            )
        ], style=input_style),

        html.Div(children=[
            html.H6(children='Cross-Track Noise'),
            dcc.Input(
                id="ep_normal",
                type="number",
                min=0,
                max=100,
                placeholder=1
            )
        ], style=input_style),

        html.Div(children=[
            html.H6(children='Miss Rate'),
            dcc.Input(
                id="miss_p",
                type="number",
                min=0,
                max=1,
                placeholder=0.1
            )
        ], style=input_style),

        html.Div(children=[
            html.H6(children='False Alarm Rate'),
            dcc.Input(
                id="lam",
                type="number",
                min=0,
                max=100,
                placeholder=1
            )
        ], style=input_style),

        html.Div(children=[
            html.H6(children='False Alarm Scale'),
            dcc.Input(
                id="fa_scale",
                type="number",
                min=0,
                max=100,
                placeholder=1
            )
        ], style=input_style),

        html.Div(children=[
            html.H6(children='Object Starting Positions'),
            dcc.Input(
                id="x0",
                type="text",
                placeholder="0 0 1 1 | 0 0 1 0"
            )
        ], style=input_style)
    ]),

    html.Div(children=[
        html.H3(children="Filter Parameters"),
        html.Div(children=[
            html.H6(children='Q'),
            dcc.Textarea(
                id="Q",
                placeholder=""
            )
        ], style=input_style),

        html.Div(children=[
            html.H6(children='R'),
            dcc.Textarea(
                id="R",
                placeholder=""
            )
        ], style=input_style),

        html.Div(children=[
            html.H6(children='P'),
            dcc.Textarea(
                id="P",
                placeholder=""
            )
        ], style=input_style),

        html.Div(children=[
            html.H6(children='Filter Starting Positions'),
            dcc.Input(
                id="x0_filter",
                type="text"
            )
        ], style=input_style)

    ])
])

@app.callback(
    Output('example-graph', 'figure'),
    Output('error-graph', 'figure'),
    Input('example-graph', 'figure'),
    Input('error-graph', 'figure'),
    Input('run', 'n_clicks'),
    Input('check-options', 'value'),
    State('time-steps', 'value'),
    State('nu', 'value'),
    State('ep_tangent', 'value'),
    State('ep_normal', 'value'),
    State('miss_p', 'value'),
    State('lam', 'value'),
    State('fa_scale', 'value'),
    State('x0', 'value'),
    State('Q', 'value'),
    State('R', 'value'),
    State('P', 'value'),
    State('x0_filter', 'value')

)
def update(prev_fig, prev_err, n_clicks, options, ts, nu, ep_tangent, ep_normal, miss_p, lam, fa_scale, x0, Q, R, P, x0_filter):
    global prev_clicks
    global sim
    fig = prev_fig
    err = prev_err
    if prev_clicks < n_clicks:
        prev_clicks = n_clicks
        # Set default parameters
        if ts is None:
            ts = 10
        if nu is None:
            nu = 0.01
        if ep_tangent is None:
            ep_tangent = 1
        if ep_normal is None:
            ep_normal = 0
        if miss_p is None:
            miss_p = 0
        if lam is None:
            lam = 0
        if fa_scale is None:
            fa_scale = 10
        if x0 is None:
            x0 = "0 0 1 1"

        # Parse the Object Starting Positions
        x0_split = x0.split("|")
        x0_parse = dict()

        for i, item in enumerate(x0_split):
            x0_parse[i] = np.array(item.strip().split(" ")).astype(float)
            x0_parse[i].shape = (4,1)

        if x0_filter is not None:
            x0_filter_split = x0_filter.split("|")
            x0_filter_parse = dict()
            for i, item in enumerate(x0_filter_split):
                x0_filter_parse[i] = np.array(item.strip().split(" ")).astype(float)
                x0_filter_parse[i].shape = (4, 1)
        else:
            x0_filter_parse = None

        # Parse input matrices to Kalman filter

        if Q is not None:
            Q_split = Q.split("\n")
            Q_parse = []
            for i, item in enumerate(Q_split):
                row = np.array(item.strip().split(" ")).astype(float)
                Q_parse.append(row)
            Q_parse = np.array(Q_parse)
        else:
            Q_parse = None

        if R is not None:
            R_split = R.split("\n")
            R_parse = []
            for i, item in enumerate(R_split):
                row = np.array(item.strip().split(" ")).astype(float)
                R_parse.append(row)
            R_parse = np.array(R_parse)
        else:
            R_parse = None

        if P is not None:
            P_split = P.split("\n")
            P_parse = []
            for i, item in enumerate(P_split):
                row = np.array(item.strip().split(" ")).astype(float)
                P_parse.append(row)
            P_parse = np.array(P_parse)
        else:
            P_parse = None

        sim.clear()
        sim.reset_generator(xt0=x0_parse, nu=nu, ep_normal=ep_normal, ep_tangent=ep_tangent, miss_p=miss_p, lam=lam, fa_scale=fa_scale)
        sim.generate(ts)
        sim.predict(ellipse_mode="plotly", x0=x0_filter_parse, Q=Q_parse, R=R_parse, P=P_parse)

    if n_clicks != 0:
        processes = sim.clean_process(sim.processes[0])
        colors = sim.clean_measure(sim.measure_colors[0])
        measures_true = sim.clean_measure(sim.measures[0])[:, colors == "black"]
        measures_false = sim.clean_measure(sim.measures[0])[:, colors == "red"]
        trajectories = sim.clean_trajectory(sim.trajectories[0])
        apriori_ellipses = sim.clean_ellipses(sim.apriori_ellipses[0], mode="plotly")
        aposteriori_ellipses = sim.clean_ellipses(sim.aposteriori_ellipses[0], mode="plotly")

        #errors = sim.signed_errors[0]

        fig = go.Figure()
        err = go.Figure()
        if 'process' in options:
            for i, process in enumerate(processes):
                fig.add_trace(go.Scatter(x=process[0], y=process[1], mode='lines', name='Object {} Process'.format(i)))
        if 'measure' in options:
            fig.add_trace(go.Scatter(x=measures_true[0], y=measures_true[1], mode='markers', name="True Measures",
                                     marker=dict(color="black")))
            fig.add_trace(go.Scatter(x=measures_false[0], y=measures_false[1], mode='markers', name="False Alarms",
                                     marker=dict(color="gray")))
        if 'trajectory' in options:
            for i, trajectory in enumerate(trajectories):
                fig.add_trace(go.Scatter(x=trajectory[0], y=trajectory[1], mode='lines+markers',
                                         name='Object {} Trajectory'.format(i)))
        if 'apriori-covariance' in options:
            xs = []
            ys = []
            for ellipse_list in apriori_ellipses:
                for ellipse in ellipse_list:
                    xs += list(ellipse[0])
                    xs.append(None)
                    ys += list(ellipse[1])
                    ys.append(None)

            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="A Priori Error Covariance"))

        if 'aposteriori-covariance' in options:
            xs = []
            ys = []
            for ellipse_list in aposteriori_ellipses:
                for ellipse in ellipse_list:
                    xs += list(ellipse[0])
                    xs.append(None)
                    ys += list(ellipse[1])
                    ys.append(None)

            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="A Posteriori Error Covariance"))

    #err.add_trace(go.Scatter(y=errors[0], x=list(range(errors[0].size)), mode='lines',
    #                         name="Cross-track Error", marker=dict(color="blue")))

    #err.add_trace(go.Scatter(y=errors[1], x=list(range(errors[1].size)), mode='lines',
    #                         name="Along-track Error", marker=dict(color="orange")))
    #err.add_trace(go.Scatter(y=errors[2], x=list(range(errors[2].size)), mode='lines',
    #                         name="Cross-track Velocity Error"))

    #err.add_trace(go.Scatter(y=errors[3], x=list(range(errors[3].size)), mode='lines',
    #                         name="Along-track Velocity Error"))

    return fig, err

app.run_server(debug=True)
