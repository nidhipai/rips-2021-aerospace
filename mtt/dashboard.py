# Import dash, the package used to create the dashboard
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import mtt

# Set up global variables needed to run the simulation
global sim
global prev_clicks
prev_clicks = 0
num_objects = 1

# Define colors to use in plots.
# Note this is the maximum number of objects we can plot
DEFAULT_COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                       'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                       'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

# Use the dash default stylesheet
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Default simulation values with which to start
obj1 = np.array([[0], [0], [1], [1]])
initial = {0: obj1}
dt = 1
ep_normal = 1
ep_tangent = 1
nu = 1
ts = 10
miss_p = 0
lam = 1
fa_scale = 1
gate_size = 0.95
gate_expand_size = 0.5

x_lim = 50
y_lim = 50
new_obj_prop = 0

# Style Parameters
input_margin = 10
input_style = {"display": "inline-block", "margin": input_margin}
output_style = {"display": "inline-block", "margin-right": 20, "margin-left": 20, "margin-top": 10, "margin-bottom": 10}

# Set up the necessary infrastructure to run a simulation
# Uncomment the next line and comment the line after that
# if you want to change back to an omnipotent sensor rather than a fixed-width sensor
# gen = mtt.MultiObjSimple(initial, dt, ep_tangent, ep_normal, nu, miss_p, lam, fa_scale)
gen = mtt.MultiObjFixed(initial, dt, ep_tangent, ep_normal, nu, miss_p, lam=lam, fa_scale=fa_scale, x_lim=x_lim, y_lim=y_lim, new_obj_prop=new_obj_prop)


#Set up a default tracker and simulation
# Old SHT tracker - uncomment this and comment next line to use the old method
# tracker = mtt.Presets.standardSHT(num_objects, gen.get_params())
# New MHT tracker
tracker = mtt.Presets.standardMHT(gen.get_params(), miss_p, lam, starting_pos=initial)

# Set up object to manage simulation
sim = mtt.Simulation(gen, tracker, seed_value = 0)

# Create blank figures to display at start
fig = go.Figure()
err = go.Figure()

# Create app layout in HTML
# Each div contains either a plot or some sort of input to the dashboard
app.layout = html.Div(children=[
    html.H1(children='2D Object Trajectory Tracking', style={"text-align":"center"}),

    html.Div(children='''
        This dashboard generates sample object movement in two directions and 
        uses the Multi-Hypothesis Tracking algorithm to predict and plot its trajectory. This algorithm was implemented 
        by the RIPS 2021 research team for the Aerospace Corporation. 
        To use the dashboard, input desired 
        simulation parameters into the text boxes below in the first section. To test multiple objects, either 
        increase the number of object births, or write the starting positions as multiple strings of four numbers separated
        by the pipe character. For example, parallel objects can be written like so: 0 0 1 1 | 0 5 1 1. 
        Then, input the desired tracker algorithm parameters into corresponding text boxes in the second section. 
        Don't forget that you may zoom in to the track using the + or - buttons, as well as the "Autoscale" option,
        both of which can be accessed by mousing over the plot. Also note that inputting a random seed of 0 will cause
        the simulation to generate completely random seeds. 
    ''', style={"margin":10}),

    html.Button('Run Simulation', id='run', n_clicks=0, style={"margin-top": 10,"margin-left":20}),
    dcc.Checklist(id='check-options',
        options=[
            {'label': 'Process', 'value': 'process'},
            {'label': 'Measure', 'value': 'measure'},
            {'label': 'Trajectory', 'value': 'trajectory'},
            {'label': 'A Priori Error Covariance', 'value': 'apriori-covariance'},
            {'label': 'A Posteriori Error Covariance', 'value': 'aposteriori-covariance'}
        ],
        value=['process', 'measure', 'trajectory'],
        labelStyle={'display': 'inline-block'},
        style={"margin-top": 10, "margin-left":20}
    ),
    dcc.Checklist(id='display_params',
        options=[
            {'label': 'Display Parameters on Graph?', 'value': 'True'}
        ],
        value=['True'],
        style={"margin-left":20}
    ),
    html.Div(children=[
        dcc.Graph(
            id='example-graph',
            figure=fig
        )
    ], style={"display": "inline-block"}),

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
        ], style=output_style),

        html.Div(children=[
            html.H6(children='Multi-Object Tracking Precision'),
            html.Div(id='motp-output', style={'whiteSpace': 'pre-line'})
        ], style=output_style),

        html.Div(children=[
            html.H6(children='Multi-Object Tracking Accuracy'),
            html.Div(id='mota-output', style={'whiteSpace': 'pre-line'})
        ], style=output_style),

        html.Div(children=[
            html.H6(children='Time Taken (s)'),
            html.Div(id='time-taken', style={'whiteSpace': 'pre-line'})
        ], style=output_style),
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
                placeholder=15
            )
        ], style=input_style),

    html.H6(children="Gating Method"),

    dcc.RadioItems(
        options=[
            {'label': 'Mahalanobis', 'value': 'mahalanobis'},
            {'label': 'Euclidean', 'value': 'euclidean'},
        ],
    value='mahalanobis',
    id = 'gate_method',
    labelStyle={'display': 'inline-block'}
    ),

        html.Div(children=[
            html.H6(children='Measure Noise'),
            dcc.Input(
                id="nu",
                type="number",
                min=0.0000001,
                max=100,
                placeholder=0.1
            )
        ], style=input_style),

        html.Div(children=[
            html.H6(children='Along-Track Noise'),
            dcc.Input(
                id="ep_tangent",
                type="number",
                min=0,
                max=100,
                placeholder=0.1
            )
        ], style=input_style),

        html.Div(children=[
            html.H6(children='Cross-Track Noise'),
            dcc.Input(
                id="ep_normal",
                type="number",
                min=0,
                max=100,
                placeholder=0.1
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
            html.H6(children='Sensor Width'),
            dcc.Input(
                id="x_lim",
                type="number",
                min=10,
                max=10000,
                placeholder=50
            )
        ], style=input_style),

        html.Div(children=[
            html.H6(children='Sensor Height'),
            dcc.Input(
                id="y_lim",
                type="number",
                min=10,
                max=10000,
                placeholder=50
            )
        ], style=input_style),

        html.Div(children=[
            html.H6(children='New Object Proportion'),
            dcc.Input(
                id="new_obj_prop",
                type="number",
                min=0,
                max=1,
                placeholder=0
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

        html.Div(children=[
            html.H6(children='Prune Time'),
            dcc.Input(
                id="prune_time",
                type="number",
                min=0,
                max=100,
                placeholder=4
            )
        ], style=input_style),

        html.Div(children=[
            html.H6(children='Old Track Threshold'),
            dcc.Input(
                id="tot",
                type="number",
                placeholder="0.001"
            )
        ], style=input_style),

        html.Div(children=[
            html.H6(children='Missed Measurement Threshold'),
            dcc.Input(
                id="tmm",
                type="number",
                placeholder="0.1"
            )
        ], style=input_style),

        html.Div(children=[
            html.H6(children='New Track Threshold'),
            dcc.Input(
                id="tnt",
                type="number",
                placeholder="1"
            )
        ], style=input_style)
    ])
])

# Callback to update the graphs, both when check is clicked and when button is pressed
# This function manages the running of the dashboard
# See Dash documentation for more details
@app.callback(
    Output('example-graph', 'figure'), # Outputs are the graphs
    Output('error-graph', 'figure'),
    Output('seed-output', 'children'),
    Output('motp-output', 'children'),
    Output('mota-output', 'children'),
    Output('time-taken', 'children'),
    Input('example-graph', 'figure'), # Inputs are the graphs, the button being clicked, and the check box
    Input('error-graph', 'figure'),
    Input('run', 'n_clicks'),
    Input('check-options', 'value'),
    Input('display_params', 'value'),
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
    State('gate_expand_size', 'value'),
    State('prune_time', 'value'),
    State('gate_method', 'value'),
    State('tot', 'value'),
    State('tmm', 'value'),
    State('tnt', 'value'),
    State('x_lim', 'value'),
    State('y_lim', 'value'),
    State('new_obj_prop', 'value'),

)
def update(prev_fig, prev_err, n_clicks, options, display_params, ts, nu, ep_tangent, ep_normal, miss_p, lam, fa_scale, x0, seed, Q, R, P, gate_size, gate_expand_size, prune_time, gate_method, tot, tmm, tnt, x_lim, y_lim, new_obj_prop):
    # Set up the global variables and default values for plotting later
    global prev_clicks
    global sim
    fig = prev_fig
    err = prev_err
    time_taken = 0
    mota = 0
    motp = 0
    # use_best controls whether the plot shows the final hypothesis or the prediction at each time step
    # Change this to false if you want to plot the prediction at each individual time step
    use_best = True

    # For each possible input, set a default value if the user does not put anything in
    if ts is None:
        ts = 15
    if tmm is None:
        tmm = 0.1
    if tnt is None:
        tnt = 1
    if tot is None:
        tot = 0.001
    # Detect whether the "RUN" button has been clicked
    if prev_clicks < n_clicks:
        prev_clicks = n_clicks
        # For each possible input, set a default value if the user does not put anything in
        if nu is None or nu == "":
            nu = 0.1
        if ep_tangent is None or ep_tangent == "":
            ep_tangent = 0.1
        if new_obj_prop is None:
            new_obj_prop = 0
        if x_lim is None or x_lim == "":
            x_lim = 50
        if y_lim is None or y_lim == "":
            y_lim = 50
        if ep_normal is None or ep_normal == "":
            ep_normal = 0.1
        if miss_p is None or miss_p == "":
            miss_p = 0
        if lam is None or lam == "":
            lam = 0
        if fa_scale is None or fa_scale == "":
            fa_scale = 1
        if x0 is None or x0 == "":
            x0 = "0 0 1 1"
        if seed is None or seed == "":
            seed = "0"
        if gate_size is None or gate_size == "":
            gate_size = 0.95
        if gate_expand_size is None or gate_expand_size == "":
            gate_expand_size = 0.5
        if prune_time is None or prune_time == "":
            prune_time = 4
        # Parse the Object Starting Positions from string into dictionary
        x0_split = x0.split("|")
        x0_parse = dict()

        for i, item in enumerate(x0_split):
            x0_parse[i] = np.array(item.strip().split(" ")).astype(float)
            x0_parse[i].shape = (4,1)

        # Parse input matrices from string to matrix to input to Kalman filter
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

        # Set up the simulation with the newly specified parameters
        sim.seed_value = int(seed)
        sim.clear(lam, miss_p)
        sim.reset_generator(xt0=x0_parse, nu=nu, ep_normal=ep_normal, ep_tangent=ep_tangent, miss_p=miss_p, lam=lam, fa_scale=fa_scale, x_lim=x_lim, y_lim=y_lim, new_obj_prop=new_obj_prop)

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
        # Add some noise to the starting position based on P
        starting_pos = {}
        for i, pos in x0_parse.items():
            rand = np.random.multivariate_normal(np.zeros(4), P_parse)
            rand.shape = (4,1)
            starting_pos[i] = pos + rand

        # Set the parameters for the tracker and generate the data and predictions
        sim.reset_tracker(mtt.Presets.standardMHT(gen.get_params(), miss_p, lam, gate_size=gate_size, gate_expand_size=gate_expand_size, gate_method=gate_method, tot=tot, tmm=tmm, tnt=tnt, born_p=new_obj_prop, prune_time=prune_time, scoring_method="chi2", starting_pos=starting_pos))
        sim.generate(ts)
        sim.predict(ellipse_mode="plotly")
    # Make sure we have pressed the button before generating plot
    if n_clicks != 0:
        # Convert the form of the Simulation class variables so that they can be plotted
        processes = sim.clean_trajectory(sim.processes[0])
        max_dist = sim.get_max_correspondence_dist(processes)

        # If we only want to plot the best hypothesis at the most recent time step,
        # override the trajectories in the simulation by resetting them
        if use_best:
            sim.trajectories = sim.best_trajectories

        # Use the best correspondence algorithm to determine which predicted trajectories match up with which processes
        best_trajs, correspondences = sim.get_best_correspondence(max_dist)
        trajectories = sim.clean_trajectory(best_trajs)

        # Test if there are actually trajectories to plot; if not, we set the result to a variable to skip them later
        skip_traj = len(trajectories) == 0 or trajectories[-1] is None

        # Get colors to determine which measurements are false alarms
        colors = sim.clean_measure(sim.measure_colors[0])

        # If there are no measures, we must skip plotting them
        skip_measures = False

        # NOTE: In the following code, "clean" means convert the data structure into a numpy array
        # which is supported for plotting in Plotly, the package used to plot
        # Clean the measurements (convert data structure) so that they can be colored based on
        # which object they have been assigned
        if(colors.size > 0):
            if use_best:
                measures = sim.clean_measure2(sim.best_measurements[0], correspondences)
            else:
                measures = sim.clean_measure2(sim.sorted_measurements[0], correspondences)
        else:
            skip_measures = True

        # Similarly, clean the false alarms so that they can be plotted on the graph
        false_alarms = sim.false_alarms[0]
        false_alarms = sim.clean_false_alarms(false_alarms) if len(false_alarms) > 0 else []

        # Clean the ellipses so they can be plotted
        apriori_ellipses = sim.clean_ellipses(sim.apriori_ellipses[0], mode="plotly")
        aposteriori_ellipses = sim.clean_ellipses(sim.aposteriori_ellipses[0], mode="plotly")

        # As requested by Jaime, calculate the along-track and cross-track errors for plotting
        atct_errors = mtt.MTTMetrics.atct_signed(processes, trajectories)

        # Create a list of time labels to attach to the trajectories and processes in the plot
        time = ["time = {}".format(t) for t in range(processes[0][0].size)]

        # Set the range manually to prevent the animation from dynamically changing the range
        # This is done by calculating the maximum and minimum
        # x and y limits of the trajectories, process and measurements
        if isinstance(gen, mtt.MultiObjFixed) and x_lim is not None and y_lim is not None:
            xmin = -x_lim
            xmax = x_lim
            ymin = -y_lim
            ymax = y_lim
            xrange = [xmin, xmax]
            yrange = [ymin, ymax]
        else:
            measure_max = []
            measure_min = []

            if len(false_alarms[0]) > 0 and not skip_measures:
                measure_max.append(max(false_alarms[0]))
                measure_max.append(max(false_alarms[1]))
                measure_min.append(min(false_alarms[0]))
                measure_min.append(min(false_alarms[1]))

            # Check to make sure there is a trajectory to plot, and not a filler list of Nones
            if not skip_traj:
                xmax = max([max([process[0].max() for process in processes]), max(
                    [trajectory[0][trajectory[0] != None].max() for trajectory in trajectories] + measure_max)])
                xmin = min([min([process[0].min() for process in processes]), min(
                    [trajectory[0][trajectory[0] != None].min() for trajectory in trajectories] + measure_min)])
                ymax = max([max([process[1].max() for process in processes]), max(
                    [trajectory[1][trajectory[0] != None].max() for trajectory in trajectories] + measure_max)])
                ymin = min([min([process[1].min() for process in processes]), min(
                    [trajectory[1][trajectory[0] != None].min() for trajectory in trajectories] + measure_min)])
                xrange = [xmin * 1.1, xmax * 1.1]
                yrange = [ymin * 1.1, ymax * 1.1]
            else:
                xmax = max([max([process[0].max() for process in processes])])
                xmin = min([min([process[0].min() for process in processes])])
                ymax = max([max([process[1].max() for process in processes])])
                ymin = min([min([process[1].min() for process in processes])])
                xrange = [xmin * 1.1, xmax * 1.1]
                yrange = [ymin * 1.1, ymax * 1.1]

        desc = ''
        # Set up a string to print out the parameters on the plot
        # This includes all of the parameters which have been input
        if "True" in display_params:
            for key, value in sim.descs[0].items():
                if key not in ["Q", "R", "fep_at", "fep_ct", "fnu", "P", "Time Steps"]:
                    desc += key + " = " + value.replace("\n", "<br>").replace("[[", "<br> [").replace("]]","]") + "<br>"

        data = []

        # Get labels for the trajectory
        # Need a mapping from trajectory list index to process list index
        all_keys = sim.get_traj_keys(best_trajs)

        # Test if each set of processes, measurements, and trajectories has been "checked" on the dashboard
        # If so, plot using Plotly (see plotly documentation on these functions for more information)

        # Plot the processes
        if 'process' in options:
            for i, process in enumerate(processes):
                data.append(go.Scatter(x=process[0], y=process[1], mode='lines', name='Obj {} Process'.format(i), text=time, line=dict(color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)])))
        # Plot the measurements
        if 'measure' in options and not skip_measures:
            for i, key in enumerate(all_keys):
                if key in measures.keys():
                    # NOTE: no time step added to measurements
                    data.append(go.Scatter(x=measures[key][0], y=measures[key][1], mode='markers', name="Measures Assigned Obj {}".format(key),
                                         marker=dict(color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)])))
            # Plot the false alarms which have not been assigned to an object
            data.append(go.Scatter(x=false_alarms[0], y=false_alarms[1], mode='markers', name="False Alarms",
                                marker=dict(color="black", symbol="x")))
        # Plot the trajectories
        if 'trajectory' in options:
            for i, key in enumerate(all_keys):
                if not np.all(np.isnan(trajectories[i])):
                    data.append(go.Scatter(x=trajectories[i][0], y=trajectories[i][1], mode='lines',
                                         name='Obj {} Prediction'.format(key), text=time, line=dict(width=3, dash='dot', color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)])))
        # Plot the a priori ellipses
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
        # Plot the a posteriori ellipses
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

        # Create error figure
        errlayout = go.Layout(
        width=700,
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(linecolor="lightgray", gridcolor="lightgray"),
        yaxis=dict(linecolor="lightgray", gridcolor="lightgray")
        )

        err = go.Figure(layout=errlayout)

        # Set font size for figures
        # UPdate error figure with font size
        fontsize = 13
        err.update_xaxes(tickfont_size=fontsize, title = "Time Step")
        err.update_yaxes(tickfont_size=fontsize)
        err.update_layout(
        legend=dict(
            font=dict(
                size=fontsize,
                color="black"
            )
        ))

        # Plot the along-track and cross-track errors for each object
        for i, obj_error in enumerate(atct_errors):
            err.add_trace(go.Scatter(y=obj_error[0], x=list(range(len(obj_error[0]))), mode='lines', name="Obj {} Along-track Position Error".format(i)))
            err.add_trace(go.Scatter(y=obj_error[1], x=list(range(len(obj_error[1]))), mode='lines', name="Obj {} Cross-track Position Error".format(i)))
            err.add_trace(go.Scatter(y=obj_error[2], x=list(range(len(obj_error[2]))), mode='lines', name="Obj {} Along-track Velocity Error".format(i)))
            err.add_trace(go.Scatter(y=obj_error[3], x=list(range(len(obj_error[3]))), mode='lines', name="Obj {} Cross-track Velocity Error".format(i)))

        # Set up plotly animations
        frames = []

        # Iterate through each time step to add the plot for that time as an animation frame
        for t in range(ts):
            scatters = []
            # Plot animation frames for each process
            if 'process' in options:
                for i, process in enumerate(processes):
                    scatters.append(
                        go.Scatter(x=process[0, :(t+1)], y=process[1, :(t+1)], mode='lines', name='Object {} Process'.format(i),
                                   text=time))
            # Plot animation frames for each set of measurements
            if 'measure' in options and not skip_measures:
                for i, key in enumerate(all_keys):
                    if key in measures.keys():
                        # NOTE: no time step added
                        scatters.append(go.Scatter(x=measures[key][0][:(t+1)], y=measures[key][1][:(t+1)], mode='markers', name="Measures Assigned Obj {}".format(key),
                                       marker=dict(color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)])))
                scatters.append(go.Scatter(x=false_alarms[0][:(t+1)], y=false_alarms[1][:(t+1)], mode='markers', name="False Alarms",
                                       marker=dict(color="black", symbol="x")))
            # Plot animation frames for each predicted trajectory
            if 'trajectory' in options and not skip_traj:
                for i, key in enumerate(all_keys):
                    if not np.all(np.isnan(trajectories[i])):
                        scatters.append(go.Scatter(x=trajectories[i][0, :(t+1)], y=trajectories[i][1, :(t+1)], mode='lines',
                                                 name='Object {} Prediction'.format(key), text=time, line=dict(width=3, dash='dash')))
            # Add all plots as a single frame to the list of animation frames
            frames.append(go.Frame(data=scatters))

        # Set up the layout of the plots and add the buttons and animation frames
        layout = go.Layout(xaxis_range=xrange, yaxis_range=yrange, autosize=False,
                           width=800,
                           plot_bgcolor='rgba(0,0,0,0)',
                           xaxis=dict(linecolor="lightgray", gridcolor="lightgray"),
                           yaxis=dict(linecolor="lightgray", gridcolor="lightgray"),
                           xaxis_title="x",
                           yaxis_title="y",
                           margin=dict(l=30, r=30, t=30, b=30),
                           annotations=[
                               go.layout.Annotation(
                                   text=desc,
                                   align='left',
                                   showarrow=False,
                                   xref='paper',
                                   yref='paper',
                                   x=1.5,
                                   y=0,
                               )],
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
                               ],"pad": {"r": 30, "t": 30}}]
                           )
        # Previously-used error metrics:
        # rmse = mtt.MTTMetrics.RMSE_euclidean(processes, trajectories)
        # num_measures = sum([len(time_step) for time_step in sim.measures[0]])

        # If we have trajectories, calculate their MOTA and MOTP to display on the plot
        if not skip_traj:
            mota, motp = mtt.MTTMetrics.mota_motp(processes, trajectories, all_keys)
        # Set up the figure with all of the plots which we generated previously
        fig = go.Figure(data=data, layout=layout, frames=frames)
        fig.update_xaxes(tickfont_size=fontsize)
        fig.update_yaxes(tickfont_size=fontsize)
        fig.update_layout(
        legend=dict(
            font=dict(
                size=fontsize,
                color="black"
            )
        ))
        # Include the time taken on the dashboard itself as a text output
        time_taken = sim.time_taken[0]

    # Return all of the HTML elements we plan to display
    return fig, err, sim.cur_seed, str(mota), str(motp), time_taken

# The line below allows the dashboard to be hosted by running dashboard.py
# This must be changed if hosting on a remote web server such as PythonAnywhere
# Please see the Dash (or Flask) documentation for more information on running dashboards
app.run_server(debug=True)
