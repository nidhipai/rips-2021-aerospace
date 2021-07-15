"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
Simulation
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from .single_target_evaluation import SingleTargetEvaluation
from itertools import repeat

from .gating import DistanceGating
from .data_association import DataAssociation
from .kalmanfilter2 import KalmanFilter
from .track_maintenance import TrackMaintenance
from .filter_predict import FilterPredict


from mpl_toolkits import mplot3d
from matplotlib.patches import Ellipse
plt.rcParams["figure.figsize"] = (12, 8)

font = {'size': 18}

plt.rc('font', **font)


# The Simulation class runs the data generator and the kalman filter to simulate an object in 2D.
class Simulation:
	def __init__(self, generator, kFilter, tracker, methods=None, predict_params=None, seed_value=1):
		"""
		Constructs a simulation environment for one-line plotting data
		Args:
			generator: Data Generator object
			kFilter: function for predicting the trajectory
			tracker: constructor for a tracker object
			predict_params: params for the kalman filter that should override those from the generator - not necessary
				if methods is not None
			methods: a list of objects that the pipeline uses
			seed_value: random seed value to get the same trajectories each time
		"""

		self.rng = np.random.default_rng(seed_value)
		self.generator = generator
		self.kFilter = kFilter
		self.tracker = tracker
		self.tracker_model = None
		self.n = generator.n
		self.processes = dict()
		self.measures = dict()
		self.measure_colors = dict()
		self.trajectories = dict()
		self.signed_errors = dict()
		self.descs = dict()
		self.ellipses = dict()
		self.false_alarms = dict()
		self.sorted_measurements = dict()
		if methods is not None:
			self.methods = methods
		else: # for backwards compatibility
			gen_params = self.generator.params

			# distance_gating = DistanceGating(10, method="euclidean") Not strictly necessary
			data_association = DataAssociation(method="euclidean")
			track_maintenance = TrackMaintenance(KalmanFilter, gen_params, 3, 4, 7, predict_params=predict_params)
			filter_predict = FilterPredict()
			self.methods = [data_association, track_maintenance, filter_predict]

	# the generate functions takes in the number of time_steps of data to be generated and then proceeds to use the
	# data generator object to create the dictionary of processes and measures.
	def generate(self, time_steps):
		"""
		The generate function takes in the number of time_steps of data to be generated and then proceeds to use the
		data generator object to create the dictionary of processes and measures.

		Args:
			time_steps (int): the number of time steps to simulate
		"""

		#we generate the process data and the measure data and assign it to the instances of processes and measures
		process = self.generator.process(time_steps - 1, self.rng)
		self.processes[len(self.processes.keys())] = process
		self.measures[len(self.measures.keys())], self.measure_colors[len(self.measure_colors.keys())] = self.generator.measure(process, self.rng)

		self.descs[len(self.descs.keys())] = {
			"Tangent Variance": str(self.generator.Q[2, 2]),
			"Normal Variance": str(self.generator.Q[3, 3]),
			"Measurement Noise": str(self.generator.R[1, 1]),
			"Missed Measures": str(self.generator.miss_p),
			"FA Rate": str(self.generator.lam),
			"FA Scale": str(self.generator.fa_scale),
			"Time Steps": str(time_steps)
		}

	#We use the kalman filter and the generated data to predict the trajectory of the simulated object
	def predict(self, ellipse_mode="mpl", index=None, x0=None, Q=None, R=None, P=None, H=None, u=None):
		"""
		The predict function uses Tracker to create an estimated trajectory for our simulated object.

		Args:
			index (int): the stored trajectory to predict
			x0 (ndarray): a single 2D column vector representing the starting states
			Q (ndarray): the process noise covariance matrix
			R (ndarray): the measurement noise covariance matrix
			H (ndarray): the measurement function jacobian
			u (ndarray): the input control vector
		"""

		if index is None:
			index = len(self.processes.keys()) - 1
		output = np.empty((self.n, 1))

		self.tracker_model = self.tracker(self.methods)
		# {0: first}
		# Iterate through each time step for which we have measurements
		for i in range(len(self.processes[index])):
			# Obtain a set of guesses for the current location of the object given the measurements
			self.tracker_model.predict(deepcopy(self.measures[index][i]))

		# Store our output as an experiment
		latest_trajectory = self.tracker_model.get_trajectories()
		self.trajectories[len(self.trajectories.keys())] = latest_trajectory

		#Now store the errors at each time step
		"""
		self.signed_errors[index] = []
		for i, next_guess in enumerate(latest_trajectory):
			# Calculate the along-track and cross track error using rotation matrix
			for j, value in next_guess.items():
				true_val = self.processes[index][i][j]
				step_error = self.generator.W(true_val) @ (true_val - next_guess[0])
				self.signed_errors[index].append(step_error)
				
		self.signed_errors[index] = np.array(self.signed_errors[index]).squeeze().T
		"""

		# Store our output as an experiment
		self.ellipses[len(self.ellipses.keys())] = self.tracker_model.get_ellipses()
		self.false_alarms[len(self.false_alarms.keys())] = self.tracker_model.false_alarms
		self.sorted_measurements[len(self.sorted_measurements)] = self.tracker_model.get_sorted_measurements()

		for method in self.methods:
			if isinstance(method, TrackMaintenance):
				kalman_params = method.filter_params
				break
		# this code will throw an error if there's no track maintenance object in the pipeline

		self.descs[len(self.descs.keys()) - 1] = {**self.descs[len(self.descs.keys()) - 1], **{
			"Q": str(kalman_params['Q']),
			"R": str(kalman_params['R']),
			"fep_at": str(kalman_params['Q'][2][2]),
			"fep_ct": str(kalman_params['Q'][3][3]),
			"fnu": str(kalman_params['R'][0][0]),
			"P": str(kalman_params['P'][0][0])
		}}

		#process = self.processes[index]
		#process = self.clean_process(process)[0]  # get first two position coordinates
		#traj = self.trajectories[index]
		#traj = self.clean_trajectory(traj)[0]
		#center_errors = (np.sqrt(np.power(process[:2, :][0] - traj[0], 2) + np.power(process[:2, :][1] - traj[1], 2)))
		#center_errors = center_errors[20:]
		#print("PROC", process[:2, :])
		#print("TRAJ", traj)
		#print(center_errors)
		#center_errors = SingleTargetEvaluation.center_error(process[:2, :], traj)
		#self.RMSE = np.sqrt(np.dot(center_errors, center_errors) / len(center_errors))
		#print(self.RMSE)
		#self.AME = sum(center_errors) / len(center_errors)


	def experiment(self, ts, test="data", **kwargs):
		"""
		Runs an experiment, generating data and producing a trajectory

		Args:
			ts (int): the number of time steps to simulate
			test (str): Whether the experiment is affecting the underlying data generation or the parameters of the filter. Options = ("data", "filter")
			kwargs: values to test in experiments. These should be inputs to the data generator (for test="data") or the filter (for test="filter")
		"""

		# Cast to a list if time steps are not already in a list
		if type(ts) != list:
			ts_modified = [ts]
		else:
			ts_modified = ts

		# If we are testing multiple potential values of parameters for data generation, we generate
		# several sets of data and predict for all of them
		if test == "data":
			for ts_item in ts_modified:
				for arg in kwargs.items():
					for value in arg[1]:
						self.generator = self.generator.mutate(**{arg[0]: value})
						self.generate(ts_item)
						self.predict()
		# If we are testing multiple potential values of parameters for the filter, we generate one set of data and for
		# each experiment we run, copy it and run the filter
		elif test == "filter":
			for ts_item in ts_modified:
				self.generate(ts_item)
				for arg in kwargs.items():
					for i, value in enumerate(arg[1]):
						if i != 0:
							self.processes[len(self.processes)] = self.processes[len(self.processes) - 1]
							self.measures[len(self.measures)] = self.measures[len(self.measures) - 1]
							self.descs[len(self.descs)] = self.descs[len(self.descs)-1]
						self.predict(index = i, **{arg[0]: value})
		else:
			print("Not a valid test type. Choose either data or filter")

	def experiment_plot(self, ts, var, plot_error_q=False, test="data", **kwargs):
		"""
		Function to both run an experiment and plot the results

		Args:
			ts (int): the number of time steps to simulate
			var (str) : variable to display in title. This should change across experiments.
			test (str): Whether the experiment is affecting the underlying data generation or the parameters of the filter. Options = ("data", "filter")
			plot_error_q (bool): Whether error should be plotted.
			kwargs: values to test in experiments. These should be inputs to the data generator (for test="data") or the filter (for test="filter")
		"""

		# Must start by clearing so previous experiments are not added to the plot
		self.clear()
		self.experiment(ts, test, **kwargs)
		self.plot_all(var)
		if plot_error_q:
			self.plot_all(error=True, var=var)

	def plot_error(self, index=None, ax=None, title="Error", var="Time Steps"):
		"""
		Plot our trajectory based on the predicted trajectories given by the kalman filter object.

		Args:
			index (int): The experiment to plot the error
			ax (matplotlib.pyplot): Matplotlib multiplot, only used if plotting multiple experiments
			title (String): Text that appears at the top of the plot. Default = "Error"
			var (String): Variable that you want to display in the title. Default = "Time Steps"

		"""

		# THIS CURRENTLY ONLY HANDLES THE FIRST OBJECT
		# NEED TO UPDATE
		if index is None:
			index = len(self.processes.keys()) - 1
		process = self.processes[index]
		process = self.clean_process(process)[0]  # get first two position coordinates
		traj = self.trajectories[index]
		traj = self.clean_trajectory(traj)[0]

		# legend should be true if the plot needs a legend (it's only one plot and the legend isn't on an outside axis)
		legend = False
		if ax is None:
			fig, ax = plt.subplots()
			legend = True
		plt.rcParams.update({'font.size': 10})

		# Plot in two dimensions
		if self.n // 2 == 2:

			#Calculate the errors over time
			center_errors = SingleTargetEvaluation.center_error(process[:2, :], traj)

			line1, = ax.plot(center_errors)
			lines = [line1]

			ax.set_title(title + "\n" + var + " = " + str(self.descs[index][var]))
			ax.set_xlabel("Time")
			ax.set_ylabel("Error")
			#ax.axis('square')
			if legend is True:
				legend_size = 16
				ax.legend(["Error"],prop={'size': legend_size})
			return lines
		else:
			print("Number of dimensions cannot be graphed (error plot).")


	'''We plot our trajectory based on the predicted trajectories given by the kalman filter object. '''
	def plot(self, var="Time Steps", index=None, title="Object Position", x_label="x", y_label="y", z_label="z", ax=None, ellipse_freq=0, tail = 0):
		"""
		Plot our trajectory based on the predicted trajectories given by the kalman filter object.

		Args:
			var (str): variable to plot
			index (int): The index of the stored data to plot
			title (str): title of the plot
			x_label (str): label for the x-axis
			y_label (str): label for the y-axis
			z-label (str) : label for the z-axis if applicable
			ax (pyplot): an already created plot
			ellipse_freq (int) : an int representing how often ellipses are drawn.
				At 1, ellipse is drawn at every point. At 2, ellipse is drawn at every other point
		"""
		labs = []
		if index is None:
			index = len(self.processes.keys()) - 1

		# Convert stored experiments into numpy matrices for plotting
		# (or list for measures)
		if len(self.processes) > 0:
			process = self.processes[index]
			process = self.clean_process(process)

		if len(self.measures) > 0:
			sorted_measures = self.sorted_measurements[index]
			measure = self.clean_measure2(sorted_measures) #THIS IS CHANGED TO 2
		if len(self.trajectories) > 0:
			output = self.trajectories[index]
			output = self.clean_trajectory(output)

		colors_process = ['skyblue', 'seagreen', 'darkkhaki'] # DOESN"T WORK FOR MORE THAN 3 OBJECTS
		colors_filter = ['orange', 'violet', 'hotpink']
		false_alarm_color = 'red'
		proc_size = 3
		traj_size = 1.5
		measure_dot_size = 20

		if len(self.false_alarms) > 0:
			false_alarms = self.false_alarms[index]
			false_alarms = self.clean_false_alarms(false_alarms) if len(false_alarms) > 0 else []

		# Select proper ellipses to plot
		ellipses = None
		if len(self.ellipses) > index:
			ellipses = self.ellipses[index]
			ellipses = self.clean_ellipses(ellipses)

		# Modify the legend
		legend_size = 14
		legend = False

		# Create subplots if we have not already passed an axis into the function
		if ax is None:
			fig, ax = plt.subplots()
			legend = True
		plt.rcParams.update({'font.size': 10})

		# Plot in two dimensions
		if self.n // 2 == 2:
			lines = []

			# Add each object's process to the plot
			if len(self.processes) > 0:
				for i, obj in enumerate(process):
					if tail > 0:
						line1, = ax.plot(obj[0][-tail:], obj[1][-tail:], lw=proc_size, markersize=8, marker=',', color=colors_process[i])
					else:
						line1, = ax.plot(obj[0], obj[1], lw=proc_size, markersize=8, marker=',', color=colors_process[i])
					lines.append(line1)
					labs.append("Obj" + str(i) + " Process")

			# Add the predicted trajectories to the plot
			if len(self.trajectories) > 0:
				for i, out in enumerate(output):
					if out is not None:
						if tail > 0:
							line3, = ax.plot(out[0][-tail:], out[1][-tail:], lw=traj_size, markersize=8, marker=',',
											 color=colors_filter[i])
						else:
							line3, = ax.plot(out[0], out[1], lw=traj_size, markersize=8, marker=',',
											 color=colors_filter[i])
						lines.append(line3)
						labs.append("Obj" + str(i) + " Filter")

			# Add the measures to the plot - the colors of a measurement correspond to which track the filter thinks it belongs to
			if len(measure.values()) != 0:
				for key, value in measure.items():
					linex = ax.scatter(value[0], value[1], s=measure_dot_size, marker='x', color=colors_filter[key])
					lines.append(linex)
					labs.append("Obj" + str(key) + " Associated Measure")

			# plot what we think are false_alarms
			if len(false_alarms) > 0:
				line_fa = ax.scatter(false_alarms[0], false_alarms[1], s=measure_dot_size, marker='x', color=false_alarm_color)
				lines.append(line_fa)
				labs.append("False Alarms from Tracker")


			# Add the parameters we use. Note that nu is hardcoded as R[0,0] since the measurement noise is independent in both directions
			if tail > 0:
				title = "Zoom to End of " + title
			ax.set_title(title + "\n" + var + " = " + str(self.descs[index][var]), fontsize=20)
			ax.set_xlabel(x_label)
			ax.set_ylabel(y_label)
			ax.patches = []
			if ellipse_freq != 0 and ellipses is not None:
				for track_e in ellipses:
					for j, ellipse in enumerate(track_e):
						if j % ellipse_freq == 0:
							if tail > 0:
								if j >= len(ellipses) - tail:
									new_c = copy(ellipse)
									ax.add_patch(new_c)
								else:
									new_c = copy(ellipse)
									ax.add_patch(new_c)
							else:
								new_c = copy(ellipse)
								ax.add_patch(new_c)
				labs.append("Covariance")
			ax.set_aspect(1)
			ax.axis('square')
			#ax.set_xlim(-self.generator.x_lim, self.generator.x_lim)
			#ax.set_ylim(-self.generator.y_lim, self.generator.y_lim)

			# Add the velocity vectors to the plot
			for i, obj in enumerate(process):
				a = .05
				if tail > 0:
					ax.quiver(obj[0][-tail:], obj[1][-tail:], obj[2][-tail:], obj[3][-tail:], alpha = a)
				else:
					ax.quiver(obj[0], obj[1], obj[2], obj[3], alpha = a)

			if legend is True:
				ax.legend(handles=lines, labels=labs, fontsize=legend_size)
			# Plot labels
			true_noises = "true ep_at = " + str(self.generator.ep_tangent) + ", true ep_ct = " + str(self.generator.ep_normal)
			filter_noises = "filter ep_at = " + self.descs[0]["fep_at"] + ", filter ep_ct = " + self.descs[0]["fep_ct"]
			measurement_noise = "measurement noise = " + str(self.generator.R[0][0])
			filter_measurement_noise = "filter measurement noise = " + str(self.generator.get_params()["R"][0][0])
			#true_state = "true state = " + "[" + self.descs[0]["x0"] + ", " + self.descs[0]["y0"] + ", " + self.descs[0]["vx0"] + ", " + self.descs[0]["vy0"] + "]"
			#filter_state = "filter state = " + "[" + self.descs[0]["fx0"] + ", " + self.descs[0]["fy0"] + ", " + self.descs[0]["fvx0"] + ", " + self.descs[0]["fvy0"] + "]"
			covariance = "P = " + self.descs[0]["P"]

			caption = true_noises + "\n" + filter_noises + "\n" + measurement_noise + "\n" + filter_measurement_noise + "\n" + covariance + "\n"# + "RMSE of plot = " + str(self.RMSE) + "\nAME of plot = " + str(self.AME)
			if tail >= 0:
				fig.text(1, 0.5, caption, ha='center', fontsize = 14)
			#else:
				#print(caption)
			return lines;

		#Plot in 3 dimensions
		elif self.n // 2 == 3:
			# title = title + ", seed=" + str(self.seed_value)
			ax = plt.axes(projection='3d')
			ax.scatter3D(process[0], process[1], process[2], lw=1.5, marker=',')
			ax.scatter3D(measure[0], measure[1], measure[2], lw=0.4, marker='+')
			ax.scatter3D(output[0], output[1], output[2], lw=0.4, marker='.')
			ax.set_xlabel(x_label)
			ax.set_ylabel(y_label)
			ax.set_zlabel(z_label)
			ax.set_title(title)
			plt.legend(labs, fontsize=legend_size)
			plt.show()
		else:
			print("Number of dimensions cannot be graphed.")

	'''the plot_all function takes in a variable name, and an ellipse frequency between 0 and 1. Then, all stored experiments
	are plotted in one single figure with subplots'''
	def plot_all(self, var="Time Steps", error=False, test="data", labs=("Process", "Filter", "Measure"), ellipse_freq = 0):
		"""
		This function takes in a variable name, and an ellipse frequency between 0 and 1.
		Then, all stored experiments are plotted in one single figure with subplots

		Args:
			var (str): variable to plot
			error (bool): Whether to plot the error
			test (str): Whether the experiment is affecting the underlying data generation or the parameters of the filter. Options = ("data", "filter")
			ellipse_freq (float): between 0 and 1, the relative frequency of how often an ellipse should be plotted
		"""
		legend_size=14
		num_plots = len(self.processes)
		num_rows = int(np.ceil(num_plots / 3))
		if num_plots > 1:
			fig, ax = plt.subplots(num_rows, 3, sharey=True) if error else plt.subplots(num_rows, 3)
			#fig.set_figheight(8)
			fig.set_figwidth(12)
			plt.subplots_adjust(hspace=.5, bottom=.2)
			lines = []
			for i in range(0, len(self.processes)):
				single_ax = ax[i // 3, i % 3] if num_plots > 3 else ax[i % 3]
				if error:
					lines = self.plot_error(index=i, ax=single_ax, var=var)
				else:
					lines = self.plot(index=i, var=var, ax=single_ax, ellipse_freq=ellipse_freq)
			legend_labels = ["Error"] if error else ["Process", "Measure", "Filter"]
			if num_plots % 3 == 1:	# one plot on last row
				ax[num_rows - 1, 1].remove() if num_plots > 3 else ax[1].remove()
				# the second part is redundant at the moment because if there's only one plot it won't use this section
			if num_plots % 3 != 0:	# one or two plots
				ax[num_rows - 1, 2].remove() if num_plots > 3 else ax[2].remove()
				fig.legend(handles=lines, labels=legend_labels, loc='center',
						   bbox_to_anchor=(.73, .25), fontsize = legend_size)
			else: # three plots on last row
				fig.legend(handles=lines, labels=legend_labels, loc='lower center', bbox_to_anchor=(.5, -.015), fontsize=legend_size)
		else: # just normally plots one plot
			self.plot(ellipse_freq=ellipse_freq) if not error else self.plot_error(var=var)
		plt.tight_layout()

	def clear(self):
		"""
		This function clears all the processes, measures, trajectories, descriptions, and the ellipses.
		"""
		self.processes = dict()
		self.measures = dict()
		self.sorted_measurements = dict()
		self.signed_errors = dict()
		self.trajectories = dict()
		self.descs = dict()
		self.ellipses = dict()
		self.measure_colors = dict()

	def reset_generator(self, **kwargs):
		"""
		Updates the generator with the new keyword arguments (kwargs)

		Args:
			kwargs: Inputs to the data generator to change
		"""


		for arg in kwargs.items():
			self.generator = self.generator.mutate(**{arg[0]: arg[1]})

	@staticmethod
	def cov_ellipse(mean, cov, zoom_factor=1, p=0.95):
		"""
		The cov ellipse returns an ellipse path that can be added to a plot based on the given mean, covariance matrix
		zoom_factor, and the p-value

		Args:
			mean (ndarray): set of coordinates representing the center of the ellipse to be plotted
			cov (ndarray): covariance matrix associated with the ellipse
			zoom_factor (int) : can be tweaked to make ellipses larger
			p (float): the confidence interval

		Returns:
			Ellipse: return the Ellipse created.
		"""
		# s takes into account the p-value given
		s = -2 * np.log(1 - p)
		a = s*cov
		a = a.round(decimals=16)
		# w and v give the eigenvalues and the eigenvectors of the covariance matrix scaled by s
		w, v = np.linalg.eig(a)
		w = np.sqrt(w)
		#calculate the tilt of the ellipse
		ang = 90 - np.arctan2(v[0, 0], v[1, 0]) / np.pi * 180
		# Figure out which eigenvector is associated with which direction
		width = w[0]
		height = w[1]

		ellipse = Ellipse(xy=mean, width=zoom_factor*width, height=zoom_factor*height, angle=ang, edgecolor='g', fc='none', lw=1)
		return ellipse

	def cov_ellipse_plotly(self, mean, cov, zoom_factor=1, p=0.95):

		if type(mean) != np.ndarray:
			mean = np.array(mean)
		mean.shape = (2,1)
		N = 100
		s = -2 * np.log(1 - p)
		a = s * cov
		a = a.round(decimals=16)
		# w and v give the eigenvalues and the eigenvectors of the covariance matrix scaled by s
		w, v = np.linalg.eig(a)
		ang = np.arctan2(v[0, 0], v[1, 0])
		width = w[0]
		height = w[1]

		# ellipse parameterization with respect to a system of axes of directions a1, a2
		t = np.linspace(0, 2 * np.pi, N)
		xs = width*np.cos(t)
		ys = height*np.sin(t)
		R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
		return np.dot(R, [xs, ys]) + mean[:,-1][:, np.newaxis]

	@staticmethod
	def clean_process(processes):
		"""
		Converts a single process from a dictionary of lists of state vectors to a list of numpy arrays
		representing the position at each time step for plotting
		"""
		output = list(repeat(np.empty((4, 1)), max([key for step in processes for key in step.keys()]) + 1))
		for step in processes:
			for key, value in step.items():
				output[key] = np.append(output[key], value, axis=1)
		# Remove the filler values from the start of each array
		# and only keep the values representing position
		for i, arr in enumerate(output):
			output[i] = arr[:, 1:]
		return output

	@staticmethod
	def clean_measure(measure):
		output = [point for sublist in measure for point in sublist]
		output = np.array(output).squeeze().T
		return output

	@staticmethod
	def clean_trajectory(trajectories):
		"""
		Converts a single trajectory from a dictionary of lists of state vectors to a list of numpy arrays
		representing the position at each time step for plotting
		"""
		output = list(repeat(np.empty((4, 1)), max([key for step in trajectories for key in step.keys()]) + 1))
		for step in trajectories: # iterate over each of the timesteps
			for key, value in step.items(): # each timestep is a dict of object predictions
				output[key] = np.append(output[key], value, axis=1) if value is not None else None
		# Remove the filler values from the start of each array
		# and only keep the values representing position
		for i, arr in enumerate(output):
			output[i] = arr[:, 1:] if output[i] is not None else None
		return output

	@staticmethod
	def clean_measure2(measures):
		"""
		Converts a dict of key: track, value: measures -> the measures array becomes a array of two arrays
		"""
		output = dict()
		for key, track in measures.items():
			track_x = []
			track_y = []
			for measure in track:
				x = None if measure is None else measure[0][0]
				y = None if measure is None else measure[1][0]
				track_x.append(x)
				track_y.append(y)
			output[key] = [track_x, track_y]
		return output

	@staticmethod
	def clean_false_alarms(false_alarms):
		"""
		Takes in a dict of key: timestep, value: array of false alarm vectors, returns a array of 2 arrays:
		first array is x-cor and second is y-cor
		"""
		output_x = []
		output_y = []
		for ts, arr in false_alarms.items():
			for fa in arr:
				output_x.append(fa[0][0])
				output_y.append(fa[1][0])
		output = [output_x, output_y]
		return output

	@staticmethod
	def clean_ellipses(ellipses):
		"""
		Returns: an array, each index represents a track and is an array of ellipse objects
		"""
		output = []
		for key, track in ellipses.items():
			track_output = []
			for param_set in track:
				track_output.append(Simulation.cov_ellipse(param_set[0], param_set[1]))
			output.append(track_output)
		return output

'''The same as the cov_ellipse function, but draws multiple p-values depending on the passed on list. One can also 
plot the scattered values using this function to see which points are outliers. '''
def cov_ellipse_fancy(X, mean, cov, p=(0.99, 0.95, 0.90)):
	"""
	The same as the cov_ellipse function, but draws multiple p-values depending on the passed on list. One can also
	plot the scattered values using this function to see which points are outliers.

	Args:
		X (ndarray): list of points to be plotted on a scatterplot
		mean (ndarray): set of coordinates representing the center of the plotted ellipse
		cov (ndarray): the covariance matrix
		p (list): list of confidence ellipses to be plotted
	"""
	plt.rcParams.update({'font.size': 22})
	fig = plt.figure(figsize=(12, 12))
	colors = ["black", "red", "purple", "blue"]
	#colors = Cube1_4.mpl_colors
	axes = plt.gca()
	colors_array = np.array([colors[0]] * X.shape[0])

	#for loop to individually draw each of the p-ellipses.
	for i in range(len(p)):
		s = -2 * np.log(1 - p[i])
		w, v = np.linalg.eig(s * cov)
		w = np.sqrt(w)
		ang = np.arctan2(v[0, 0], v[1, 0]) / np.pi * 180
		ellipse = Ellipse(xy=mean, width=2 * w[0], height=2 * w[1], angle=ang,edgecolor=colors[i+1], lw=2, fc="none", label=str(p[i]))
		cos_angle = np.cos(np.radians(180. - ang))
		sin_angle = np.sin(np.radians(180. - ang))

		x_val = (X[:, 0] - mean[0]) * cos_angle - (X[:, 1] - mean[1]) * sin_angle
		y_val = (X[:, 0] - mean[0]) * sin_angle + (X[:, 1] - mean[1]) * cos_angle

		#calculating whether a point is inside an ellipse. If it is, we change the color of the point to a specific desired color.
		rad_cc = (x_val ** 2 / (w[0]) ** 2) + (y_val ** 2 / (w[1]) ** 2)
		colors_array[np.where(rad_cc <= 1.)[0]] = colors[i+1]

		axes.add_patch(ellipse)
	#plot the scattered points with the ellipses.
	axes.scatter(X[:, 0], X[:, 1], linewidths=0, alpha=1, c = colors_array)
	plt.legend(title="p-value", loc=2, prop={'size': 15}, handleheight=0.01)
	plt.show()
