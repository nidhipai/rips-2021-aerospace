import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import mtt
from copy import copy

#Declare these as global variables to be used by all callbacks
global sim
global prev_clicks
global num_objects
prev_clicks = 0
num_objects = 1

# Define colors to use in plots.
# Note this is the maximum number of objects we can plot
DEFAULT_COLORS=['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                       'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                       'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Default simulation values, just to start with
obj1 = np.array([[0], [0], [1], [1]])
initial = {0: obj1}
dt = 1
ep_normal = 1
ep_tangent = 1
nu = 1
ts = 10
miss_p = 0
lam = 0
fa_scale = 1
gate_size = 0.95
gate_expand_size = 0.5

# Style Parameters
input_margin = 10
input_style = {"display": "inline-block", "margin": input_margin}

# Set up the necessary infrastructure to run a simulation
gen = mtt.MultiObjSimple(initial, dt, ep_tangent, ep_normal, nu, miss_p, lam, fa_scale)
"""
gate = mtt.DistanceGating(gate_size, expand_gating=gate_expand_size, method="mahalanobis")
assoc = mtt.DataAssociation()
params = gen.get_params()
maintain = mtt.TrackMaintenance(mtt.KalmanFilter, params, num_obj=num_objects, num_init = 2, num_init_frames=3, num_delete=3)
filter_ = mtt.FilterPredict()
"""

#Set up a default tracker and simulation
tracker = mtt.MTTTracker(mtt.Presets.standardSHT(num_objects, gen.get_params()))
sim = mtt.Simulation(gen, tracker, seed_value = 0)

#Create blank figures to display at start
fig = go.Figure()
err = go.Figure()

#Create app layout
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
    ], style={"display": "inline-block"}),

    html.Div(children=[
        html.Div(children=[
            html.H6(children='Seed'),
            html.Div(id='seed-output', style={'whiteSpace': 'pre-line'})
        ], style=input_style),

        html.Div(children=[
            html.H6(children='Root Mean Squared Error'),
            html.Div(id='rmse-output', style={'whiteSpace': 'pre-line'})
        ], style=input_style),
    ]),

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
                placeholder=0
            )
        ], style=input_style),

        html.Div(children=[
            html.H6(children='False Alarm Rate'),
            dcc.Input(
                id="lam",
                type="number",
                min=0,
                max=100,
                placeholder=0
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
        ], style=input_style),

        html.Div(children=[
            html.H6(children='Random Seed'),
            dcc.Input(
                id="seed",
                type="text",
                placeholder="0"
            )
        ], style=input_style)

    ]),

    html.Div(children=[
        html.H3(children="Tracker Parameters"),
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
            html.H6(children='Gate Size'),
            dcc.Input(
                id="gate_size",
                type="number",
                min=0.001,
                max=1,
                placeholder=0.95
            )
        ], style=input_style),

        html.Div(children=[
            html.H6(children='Gate Expansion'),
            dcc.Input(
                id="gate_expand_size",
                type="number",
                min=0,
                max=1,
                placeholder=0.5
            )
        ], style=input_style),
    ])
])

#Callback to update the graphs, both when check is clicked and when button is pressed
@app.callback(
    Output('example-graph', 'figure'), # Outputs are the graphs
    Output('error-graph', 'figure'),
    Output('seed-output', 'children'),
    Output('rmse-output', 'children'),
    Input('example-graph', 'figure'), # Inputs are the graphs, the button being clicked, and the check box
    Input('error-graph', 'figure'),
    Input('run', 'n_clicks'),
    Input('check-options', 'value'),
    State('time-steps', 'value'), # Outputs are drawn from text box states
    State('nu', 'value'),
    State('ep_tangent', 'value'),
    State('ep_normal', 'value'),
    State('miss_p', 'value'),
    State('lam', 'value'),
    State('fa_scale', 'value'),
    State('x0', 'value'),
    State('seed', 'value'),
    State('Q', 'value'),
    State('R', 'value'),
    State('P', 'value'),
    State('gate_size', 'value'),
    State('gate_expand_size', 'value')
)
def update(prev_fig, prev_err, n_clicks, options, ts, nu, ep_tangent, ep_normal, miss_p, lam, fa_scale, x0, seed, Q, R, P, gate_size, gate_expand_size):
    global prev_clicks
    global sim
    fig = prev_fig
    err = prev_err
    rmse = 0
    if ts is None:
        ts = 10
    if prev_clicks < n_clicks:
        prev_clicks = n_clicks
        # Set default parameters
        if nu is None:
            nu = 1
        if ep_tangent is None:
            ep_tangent = 1
        if ep_normal is None:
            ep_normal = 1
        if miss_p is None:
            miss_p = 0
        if lam is None:
            lam = 0
        if fa_scale is None:
            fa_scale = 10
        if x0 is None:
            x0 = "0 0 1 1"
        if seed is None:
            seed = "0"
        if gate_size is None:
            gate_size = 0.95
        if gate_expand_size is None:
            gate_expand_size = 0.5

        # Parse the Object Starting Positions
        x0_split = x0.split("|")
        x0_parse = dict()

        for i, item in enumerate(x0_split):
            x0_parse[i] = np.array(item.strip().split(" ")).astype(float)
            x0_parse[i].shape = (4,1)

        num_objects = len(x0_parse.items())

        """
        if x0_filter is not None:
            x0_filter_split = x0_filter.split("|")
            x0_filter_parse = dict()
            for i, item in enumerate(x0_filter_split):
                x0_filter_parse[i] = np.array(item.strip().split(" ")).astype(float)
                x0_filter_parse[i].shape = (4, 1)
        else:
            x0_filter_parse = sim.generator.xt0
        """
        # Parse input matrices to Kalman filter

        if Q is not None and Q != '':
            Q_split = Q.split("\n")
            Q_parse = []
            for i, item in enumerate(Q_split):
                row = np.array(item.strip().split(" ")).astype(float)
                Q_parse.append(row)
            Q_parse = np.array(Q_parse)
        else:
            Q_parse = sim.generator.Q

        if R is not None and R != '':
            R_split = R.split("\n")
            R_parse = []
            for i, item in enumerate(R_split):
                row = np.array(item.strip().split(" ")).astype(float)
                R_parse.append(row)
            R_parse = np.array(R_parse)
        else:
            R_parse = sim.generator.R

        if P is not None and P != '':
            P_split = P.split("\n")
            P_parse = []
            for i, item in enumerate(P_split):
                row = np.array(item.strip().split(" ")).astype(float)
                P_parse.append(row)
            P_parse = np.array(P_parse)
        else:
            P_parse = np.eye(4)

        #Set up the simulation with the newly specified parameters
        sim.seed_value = int(seed)
        sim.clear()
        sim.reset_generator(xt0=x0_parse, nu=nu, ep_normal=ep_normal, ep_tangent=ep_tangent, miss_p=miss_p, lam=lam, fa_scale=fa_scale)

        params = {
            "f": sim.generator.process_function,
            "A": sim.generator.process_jacobian,
            "h": sim.generator.measurement_function,
            "Q": Q_parse,
            "W": sim.generator.W,
            "R": R_parse,
            "H": sim.generator.measurement_jacobian(x0_parse),
            "P": P_parse
        }

        sim.reset_tracker(mtt.MTTTracker(mtt.Presets.standardSHT(num_objects, params, gate_size=gate_size, gate_expand_size=gate_expand_size)))
        sim.generate(ts)
        sim.predict(ellipse_mode="plotly")

    if n_clicks != 0:
        # Generate all variables to plot
        processes = sim.clean_process(sim.processes[0])

        colors = sim.clean_measure(sim.measure_colors[0])
        measures_true = sim.clean_measure(sim.measures[0])[:, colors == "black"]
        measures_false = sim.clean_measure(sim.measures[0])[:, colors == "red"]
        measures = sim.clean_measure2(sim.sorted_measurements[0])
        false_alarms = sim.false_alarms[0]
        false_alarms = sim.clean_false_alarms(false_alarms) if len(false_alarms) > 0 else []

        trajectories = sim.clean_trajectory(sim.trajectories[0])

        apriori_ellipses = sim.clean_ellipses(sim.apriori_ellipses[0], mode="plotly")
        aposteriori_ellipses = sim.clean_ellipses(sim.aposteriori_ellipses[0], mode="plotly")
        atct_errors = mtt.MTTMetrics.atct_signed(processes, trajectories)
        time = ["time = {}".format(t) for t in range(processes[0][0].size)]

        # Set the range manually to prevent the animation from dynamically changing the range
        measure_max = []
        measure_min = []
        if measures_true.size > 0:
            measure_max.append(measures_true[0].max())
            measure_max.append(measures_true[1].max())
            measure_min.append(measures_true[0].min())
            measure_min.append(measures_true[1].min())

        if measures_false.size > 0:
            measure_max.append(measures_false[0].max())
            measure_max.append(measures_false[1].max())
            measure_min.append(measures_false[0].min())
            measure_min.append(measures_false[1].min())

        xmax = max([max([process[0].max() for process in processes]), max([trajectory[0].max() for trajectory in trajectories] + measure_max)])
        xmin = min([min([process[0].min() for process in processes]), min([trajectory[0].min() for trajectory in trajectories] + measure_min)])
        ymax = max([max([process[1].max() for process in processes]), max([trajectory[1].max() for trajectory in trajectories] + measure_max)])
        ymin = min([min([process[1].min() for process in processes]), min([trajectory[1].min() for trajectory in trajectories] + measure_min)])
        xrange = [xmin*1.1, xmax*1.1]
        yrange = [ymin*1.1, ymax*1.1]

        desc = ''
        for key, value in sim.descs[0].items():
            if key not in ["fep_at", "fep_ct", "fnu", "P"]:
                desc += key + " = " + value.replace("\n", "<br>").replace("[[", "<br> [").replace("]]","]") + "<br>"

        print(desc)

        """
        updatemenus=[{
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 500, "redraw": False},
                                "fromcurrent": False, "transition": {"duration": 0,
                                                                    "easing": "linear"}}],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                "label": "Pause",
                "method": "animate"
            }
        ]}]
        """

        data = []
        if 'process' in options:
            for i, process in enumerate(processes):
                # NOTE: the "time" text here assumes all objects are on-screen for an equal number of time steps;
                # Otherwise "time" will be incorrect
                data.append(go.Scatter(x=process[0], y=process[1], mode='lines', name='Obj {} Process'.format(i), text=time, line=dict(color=DEFAULT_COLORS[i])))
        if 'measure' in options:
            for key, value in measures.items():
                # NOTE: no time step added
                data.append(go.Scatter(x=value[0], y=value[1], mode='markers', name="Measures Assigned Obj {}".format(key),
                                     marker=dict(color=DEFAULT_COLORS[key])))
            data.append(go.Scatter(x=false_alarms[0], y=false_alarms[1], mode='markers', name="False Alarms",
                                marker=dict(color="gray")))
        if 'trajectory' in options:
            for i, trajectory in enumerate(trajectories):
                data.append(go.Scatter(x=trajectory[0], y=trajectory[1], mode='lines',
                                         name='Obj {} Prediction'.format(i), text=time, line=dict(width=3, dash='dot', color=DEFAULT_COLORS[i])))
        if 'apriori-covariance' in options:
            xs = []
            ys = []
            for ellipse_list in apriori_ellipses:
                for ellipse in ellipse_list:
                    xs += list(ellipse[0])
                    xs.append(None)
                    ys += list(ellipse[1])
                    ys.append(None)

            data.append(go.Scatter(x=xs, y=ys, mode="lines", name="A Priori Error Covariance"))

        if 'aposteriori-covariance' in options:
            xs = []
            ys = []
            for ellipse_list in aposteriori_ellipses:
                for ellipse in ellipse_list:
                    xs += list(ellipse[0])
                    xs.append(None)
                    ys += list(ellipse[1])
                    ys.append(None)

            data.append(go.Scatter(x=xs, y=ys, mode="lines", name="A Posteriori Error Covariance"))

        #Create error figure
        errlayout = go.Layout(
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(linecolor="lightgray", gridcolor="lightgray"),
        yaxis=dict(linecolor="lightgray", gridcolor="lightgray")
        )

        err = go.Figure(layout=errlayout)

        for obj_error in atct_errors:
            err.add_trace(go.Scatter(y=obj_error[0], x=list(range(len(obj_error[0]))), mode='lines', name="Along-track Position Error", marker=dict(color="orange")))
            err.add_trace(go.Scatter(y=obj_error[1], x=list(range(len(obj_error[1]))), mode='lines', name="Cross-track Position Error", marker=dict(color="blue")))
            err.add_trace(go.Scatter(y=obj_error[2], x=list(range(len(obj_error[2]))), mode='lines', name="Along-track Velocity Error", marker=dict(color="red")))
            err.add_trace(go.Scatter(y=obj_error[3], x=list(range(len(obj_error[3]))), mode='lines', name="Cross-track Velocity Error", marker=dict(color="purple")))

        frames = []
        for t in range(ts):
            scatters = []
            if 'process' in options:
                for i, process in enumerate(processes):
                    # NOTE: the "time" text here assumes all objects are on-screen for an equal number of time steps;
                    # Otherwise "time" will be incorrect
                    scatters.append(
                        go.Scatter(x=process[0, :(t+1)], y=process[1, :(t+1)], mode='lines', name='Object {} Process'.format(i),
                                   text=time))
            """
            if 'measure' in options:
                m = np.array(sim.measures[0][:(t+1)]).squeeze().T
                mc = np.array(sim.measure_colors[0][:(t+1)]).squeeze().T
                m_t = m[:, mc == "black"]
                m_f = m[:, mc != "black"]
                scatters.append(go.Scatter(x=m_t[0], y=m_t[1], mode='markers', name="True Measures",
                                         marker=dict(color="black"), text=time))
                scatters.append(go.Scatter(x=m_f[0], y=m_f[1], mode='markers', name="False Alarms",
                                         marker=dict(color="gray"), text=time))
            """
            if 'trajectory' in options:
                for i, trajectory in enumerate(trajectories):
                    scatters.append(go.Scatter(x=trajectory[0, :(t+1)], y=trajectory[1, :(t+1)], mode='lines+markers',
                                             name='Object {} Prediction'.format(i), text=time, line=dict(width=3, dash='dash')))

            frames.append(go.Frame(data=scatters))

        layout = go.Layout(xaxis_range=xrange, yaxis_range=yrange, autosize=False,
                           plot_bgcolor='rgba(0,0,0,0)',
                           xaxis=dict(linecolor="lightgray", gridcolor="lightgray"),
                           yaxis=dict(linecolor="lightgray", gridcolor="lightgray"),
                           margin=dict(l=30, r=30, t=30, b=30),
                           annotations=[
                               go.layout.Annotation(
                                   text=desc,
                                   align='left',
                                   showarrow=False,
                                   xref='paper',
                                   yref='paper',
                                   x=1.4,
                                   y=0,
                               )]
                           )

        rmse = mtt.MTTMetrics.RMSE_euclidean(processes, trajectories)
        fig = go.Figure(data=data, layout=layout, frames=frames)

    return fig, err, sim.cur_seed, rmse

app.run_server(debug=True)
