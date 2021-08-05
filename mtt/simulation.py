"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
Simulation
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import time
from .single_target_evaluation import SingleTargetEvaluation
from itertools import repeat
from scipy.stats import chi2

from .pipeline.track_maintenance import TrackMaintenance
from .pipeline.gating import DistanceGating
from .tracker2 import MTTTracker
from .mht.tracker3 import MHTTracker

from .mtt_metrics import MTTMetrics

import sys, os


from matplotlib.patches import Ellipse
plt.rcParams["figure.figsize"] = (12, 8)

font = {'size': 18}

plt.rc('font', **font)


# The Simulation class runs the data generator and the kalman filter to simulate an object in 2D.
class Simulation:
	def __init__(self, generator, tracker, seed_value=0):
		"""
		Constructs a simulation environment for one-line plotting data
		Args:
			generator: Data Generator object
			tracker: Tracker object
			seed_value: random seed value to get the same trajectories each time
		"""
		self.seed_value = seed_value
		if self.seed_value == 0:
			self.cur_seed = np.random.randint(10**7)
		else:
			self.cur_seed = seed_value
		self.rng = np.random.default_rng(self.cur_seed)
		self.generator = generator
		self.tracker_model = tracker
		self.n = generator.n
		self.processes = dict()
		self.measures = dict()
		self.measure_colors = dict()
		self.apriori_traj = dict()
		self.trajectories = dict()
		self.signed_errors = dict()
		self.descs = dict()
		self.apriori_ellipses = dict()
		self.aposteriori_ellipses = dict()
		self.false_alarms = dict()
		self.sorted_measurements = dict()
		self.time_taken = dict()
		self.atct_error = dict()
		self.mota = dict()
		self.motp = dict()
		self.track_count = dict()
		self.DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

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
			"Along-track Variance": str(self.generator.Q[2, 2]),
			"Cross-track Variance": str(self.generator.Q[3, 3]),
			"Measurement Noise": str(self.generator.R[1, 1]),
			"Missed Measures": str(self.generator.miss_p),
			"FA Rate": str(self.generator.lam),
			"FA Scale": str(self.generator.fa_scale),
			"Time Steps": str(time_steps),
			"Seed": str(self.cur_seed)
		}

	def predict(self, ellipse_mode="mpl", index=None):
		"""
		The predict function uses Tracker to create an estimated trajectory for our simulated object.

		Args:
			index (int): the stored trajectory to predict
		"""

		if index is None:
			index = len(self.processes.keys()) - 1
		output = np.empty((self.n, 1))

		# {0: first}
		# Iterate through each time step for which we have measurements
		self.trajectories[len(self.trajectories.keys())] = []
		self.apriori_traj[len(self.apriori_traj.keys())] = []
		self.apriori_ellipses[len(self.apriori_ellipses.keys())] = dict()
		self.aposteriori_ellipses[len(self.aposteriori_ellipses.keys())] = dict()
		self.false_alarms[len(self.false_alarms.keys())] = dict()
		self.sorted_measurements[len(self.sorted_measurements)] = dict()

		# MTTTracker stores false alarms and has a pipeline, but with MHT, we cannot do this until the end
		# Therefore, we divide into two instances

		total_time = 0
		for i in range(len(self.processes[index])):
			# Obtain a set of guesses for the current location of the object given the measurements
			next_measurement = deepcopy(self.measures[index][i])
			start_time = time.process_time()
			self.tracker_model.predict(next_measurement)
			total_time += time.process_time() - start_time

			if isinstance(self.tracker_model, MHTTracker):
				self.trajectories[len(self.trajectories.keys())-1].append(self.tracker_model.get_trajectories())
				self.apriori_traj[len(self.apriori_traj.keys())-1].append(self.tracker_model.get_apriori_traj())
				#self.apriori_ellipses[len(self.apriori_ellipses.keys())-1].append(self.tracker_model.get_ellipses("apriori"))
				#self.aposteriori_ellipses[len(self.aposteriori_ellipses.keys())-1].append(self.tracker_model.get_ellipses("aposteriori"))
				self.false_alarms[len(self.false_alarms.keys())-1][i] = self.tracker_model.get_false_alarms()

				apriori_ellipses = self.tracker_model.get_ellipses("apriori")
				for key, value in apriori_ellipses.items():
					if key not in self.apriori_ellipses[len(self.apriori_ellipses.keys())-1].keys():
						self.apriori_ellipses[len(self.apriori_ellipses)-1][key] = []
					self.apriori_ellipses[len(self.apriori_ellipses)-1][key].append(value)

				aposteriori_ellipses = self.tracker_model.get_ellipses("aposteriori")
				for key, value in aposteriori_ellipses.items():
					if key not in self.aposteriori_ellipses[len(self.aposteriori_ellipses.keys())-1].keys():
						self.aposteriori_ellipses[len(self.aposteriori_ellipses)-1][key] = []
					self.aposteriori_ellipses[len(self.aposteriori_ellipses)-1][key].append(value)

				sort = self.tracker_model.get_sorted_measurements()
				for key, value in sort.items():
					if key not in self.sorted_measurements[len(self.sorted_measurements)-1].keys():
						self.sorted_measurements[len(self.sorted_measurements)-1][key] = []
					self.sorted_measurements[len(self.sorted_measurements)-1][key].append(value)

				self.descs[len(self.descs.keys()) - 1] = {**self.descs[len(self.descs.keys()) - 1], **{
					"Q": str(self.tracker_model.kalman.Q),
					"R": str(self.tracker_model.kalman.R),
					"Gate Size": str(self.tracker_model.gating.error_threshold),
					"Gate Expansion %": str(self.tracker_model.gating.expand_gating),
					"Pruning": str(self.tracker_model.pruning.n),
					"fep_at": str(self.tracker_model.kalman.Q[2][2]),
					"fep_ct": str(self.tracker_model.kalman.Q[3][3]),
					"fnu": str(self.tracker_model.kalman.R[0][0]),
					"P": str(self.tracker_model.track_maintenance.P[0][0]),
				}}

		# Store the total time taken by the predict method of the tracker
		self.time_taken[len(self.time_taken.keys())] = total_time

		# Store our output as an experiment
		if isinstance(self.tracker_model, MTTTracker):
			latest_trajectory = self.tracker_model.get_trajectories()
			self.trajectories[len(self.trajectories.keys())] = latest_trajectory

			latest_apriori_traj = self.tracker_model.get_apriori_traj()
			self.apriori_traj[len(self.apriori_traj.keys())] = latest_apriori_traj
			# Store our output as an experiment
			self.apriori_ellipses[len(self.apriori_ellipses.keys())] = self.tracker_model.get_ellipses("apriori")
			self.aposteriori_ellipses[len(self.aposteriori_ellipses.keys())] = self.tracker_model.get_ellipses("aposteriori")

			self.false_alarms[len(self.false_alarms.keys())] = self.tracker_model.false_alarms
			gate_size = 0
			gate_expand = 0
			for method in self.tracker_model.methods:
				if isinstance(method, TrackMaintenance):
					kalman_params = method.filter_params
				if isinstance(method, DistanceGating):
					gate_size = method.error_threshold
					gate_expand = method.expand_gating

			self.descs[len(self.descs.keys()) - 1] = {**self.descs[len(self.descs.keys()) - 1], **{
				"Q": str(kalman_params['Q']),
				"R": str(kalman_params['R']),
				"Gate Size": str(gate_size),
				"Gate Expansion %": str(gate_expand),
				"fep_at": str(kalman_params['Q'][2][2]),
				"fep_ct": str(kalman_params['Q'][3][3]),
				"fnu": str(kalman_params['R'][0][0]),
				"P": str(kalman_params['P'][0][0]),

			}}

			self.sorted_measurements[len(self.sorted_measurements)] = self.tracker_model.get_sorted_measurements()

		# this code will throw an error if there's no track maintenance object in the pipeline

		# ASDF
		process = self.clean_trajectory(self.processes[index])
		max_dist = self.get_max_correspondence_dist(process)
		best_trajs, correspondences = self.get_best_correspondence(max_dist, index = index)
		trajectory = self.clean_trajectory(best_trajs)
		self.atct_error[len(self.atct_error)] = MTTMetrics.atct_signed(process, trajectory)
		all_keys = self.get_traj_keys(best_trajs)
		# TODO: motp/mota stuff might break fixed frame
		#self.motp[len(self.motp)], self.mota[len(self.mota)] = MTTMetrics.mota_motp(process, trajectory, all_keys)
		self.track_count[len(self.track_count.keys())] = len(self.tracker_model.tracks)

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
						# Reset rng value so we can run the same experiment but with different parameters
						self.rng = np.random.default_rng(self.cur_seed)
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
			raise Exception("Not a valid test type. Choose either \"data\" or \"filter\"")

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

	def test_tracker_model(self, ts, name, iter=3, test="data", **kwargs):
		self.clear()
		metrics = dict()
		for k in range(iter):
			if self.seed_value == 0:
				self.cur_seed = np.random.randint(10 ** 7)
			self.experiment(ts, test, **kwargs)

		file = open(name, "w")
		output = "Parameter\tValue\tMOTP\tMOTA\tTime\tTrackCount\n"
		file.write(output)

		i = 0
		rows = sum([len(kwargs[key]) for key in kwargs.keys()])
		for key in kwargs.keys():
			for param in kwargs[key]:
				# Calculate the average values for MOTP, MOTA, and Time
				motp = 0
				mota = 0
				time_taken = 0
				track_count = 0
				for j in range(iter):
					motp += self.motp[i + j*rows]
					mota += self.mota[i + j*rows]
					time_taken += self.time_taken[i + j*rows]
					track_count += self.track_count[i + j*rows]
				motp /= iter
				mota /= iter
				time_taken /= iter
				track_count /= iter

				# Format data and output to file
				data = "{}\t{}\t{}\t{}\t{}\t{}\n".format(key, param, motp, mota, time_taken, track_count)
				file.write(data)
				output += data
				i += 1

		file.close()
		return output

	def plot_error(self, index=None, ax=None, title="Error", var="Seed"):
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
		# ASDF
		process = self.clean_trajectory(process)[0]  # get first two position coordinates
		if isinstance(self.tracker_model, MHTTracker):
			max_dist = self.get_max_correspondence_dist(process)
			traj, correspondences = self.get_best_correspondence(max_dist, index=index)
		else:
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

	def plot(self, var="Seed", index=None, title="Object Position", x_label="x", y_label="y", z_label="z", ax=None, ellipse_freq=0, tail = 0):
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
			# ASDF
			process = self.clean_trajectory(self.processes[index])

		correspondences = None
		if len(self.trajectories) > 0:
			# TO DO: Need a better distance gate than inf
			if isinstance(self.tracker_model, MHTTracker):
				max_dist = self.get_max_correspondence_dist(process)
				best_trajs, correspondences = self.get_best_correspondence(max_dist, index)
			else:
				best_trajs = self.trajectories[index]

			output = self.clean_trajectory(best_trajs)
			all_keys = self.get_traj_keys(best_trajs)

		if len(self.measures) > 0:
			sorted_measures = self.sorted_measurements[index]
			measure = self.clean_measure2(sorted_measures, correspondences)


		colors_process = ['skyblue', 'seagreen', 'darkkhaki'] # DOESN"T WORK FOR MORE THAN 3 OBJECTS
		colors_filter = ['orange', 'violet', 'hotpink','red']
		false_alarm_color = 'red'
		proc_size = 2
		traj_size = 1.5
		measure_dot_size = 20

		if len(self.false_alarms) > 0:
			false_alarms = self.false_alarms[index]
			false_alarms = self.clean_false_alarms(false_alarms) if len(false_alarms) > 0 else []

		# Select proper ellipses to plot
		ellipses = None
		if len(self.apriori_ellipses) > index:
			ellipses = self.clean_ellipses(self.apriori_ellipses[index])

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
						line1, = ax.plot(obj[0][-tail:], obj[1][-tail:], lw=proc_size, markersize=8, marker=',', color=self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)])
					else:
						line1, = ax.plot(obj[0], obj[1], lw=proc_size, markersize=8, marker=',', color=self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)])
					lines.append(line1)
					labs.append("Obj " + str(i) + " Process")

			# Add the predicted trajectories to the plot
			if len(self.trajectories) > 0:
				for i, out in enumerate(output):
					if out is not None:
						if tail > 0:
							line3, = ax.plot(out[0][-tail:], out[1][-tail:], lw=traj_size, markersize=8, marker=',', linestyle='--', color=self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)])
						else:
							line3, = ax.plot(out[0], out[1], lw=traj_size, markersize=8, marker=',', linestyle='--', color=self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)])
						lines.append(line3)
						labs.append("Obj " + str(all_keys[i]) + " Filter")

			# Add the measures to the plot - the colors of a measurement correspond to which track the filter thinks it belongs to
			if len(measure.values()) != 0:
				for i, key in enumerate(all_keys):
					if key in measure.keys():
						linex = ax.scatter(measure[key][0], measure[key][1], s=measure_dot_size, marker='o', color=self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)])
						lines.append(linex)
						labs.append("Obj " + str(key) + " Associated Measure")

			# plot what we think are false_alarms
			if len(false_alarms) > 0:
				line_fa = ax.scatter(false_alarms[0], false_alarms[1], s=measure_dot_size, marker='x', color=false_alarm_color)
				lines.append(line_fa)
				labs.append("False Alarms from Tracker")

			## Add the predicted trajectories to the plot
			#if len(self.trajectories) > 0:
				#for i, out in enumerate(output):
					#if out is not None:
						#line3, = ax.plot(out[0], out[1], lw=0.4, markersize=7, marker=',')
						#lines.append(line3)
						#labs.append("Obj" + str(i) + " Filter")
					##if tail > 0:
						##line3, = ax.plot(out[0][-tail:], out[1][-tail:], lw=0.4, markersize=8, marker=',')
					##else:
						##line3, = ax.plot(out[0], out[1], lw=0.4, markersize=8, marker=',')

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
				ax.legend(handles=lines, labels=labs, fontsize=legend_size, loc = "upper left", bbox_to_anchor = (1.05, 1))
			# Plot labels
			if isinstance(self.tracker_model, MTTTracker) or True:
				true_noises = "True Error Variances: AT = {}, CT = {}".format(self.generator.ep_tangent, self.generator.ep_normal)
				measurement_noise = "Measurement Noise Variance = {}".format(self.generator.R[0][0])
				other_noise = "Miss Rate = {}, FA Rate = {}".format(self.generator.miss_p, self.generator.lam)
				filter_noise = "Filter Error Variances: AT = {}, CT = {}".format(self.descs[0]["fep_at"], self.descs[0]["fep_ct"])
				filter_measurement_noise = "Filter Measurement Noise Variance = {}".format(self.generator.get_params()["R"][0][0])
				#true_state = "true state = " + "[" + self.descs[0]["x0"] + ", " + self.descs[0]["y0"] + ", " + self.descs[0]["vx0"] + ", " + self.descs[0]["vy0"] + "]"
				#filter_state = "filter state = " + "[" + self.descs[0]["fx0"] + ", " + self.descs[0]["fy0"] + ", " + self.descs[0]["fvx0"] + ", " + self.descs[0]["fvy0"] + "]"
				covariance = "Starting P = " + self.descs[0]["P"]
				# TODO: commented out for fixed frame testing
				#mos = "MOTP = {}, MOTA = {}".format(np.round(self.motp[index], 3), np.round(self.mota[index],3))
				mos = "IDK"

				caption = true_noises + "\n" + measurement_noise + "\n" + other_noise + "\n" + filter_noise + "\n" + filter_measurement_noise + "\n" + covariance + "\n" + mos + "\n"
				if tail >= 0:
					fig.text(1, 0.1, caption, ha='center', fontsize = 14)
			#else:
				#print(caption)
			return lines;

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

	def clear(self, lam=None, miss_p=None):
		"""
		This function clears all the processes, measures, trajectories, descriptions, and the ellipses.
		"""
		if self.seed_value == 0:
			self.cur_seed = np.random.randint(10**7)
			self.rng = np.random.default_rng(self.cur_seed)
		else:
			self.cur_seed = self.seed_value
			self.rng = np.random.default_rng(self.cur_seed)

		self.processes = dict()
		self.measures = dict()
		self.sorted_measurements = dict()
		self.signed_errors = dict()
		self.trajectories = dict()
		self.descs = dict()
		self.apriori_ellipses = dict()
		self.aposteriori_ellipses = dict()
		self.measure_colors = dict()
		self.false_alarms = dict()
		self.time_taken = dict()
		self.atct_error = dict()
		self.mota = dict()
		self.motp = dict()
		self.track_count = dict()

		# Clear stored tracks from the tracker
		self.tracker_model.clear_tracks(lam=lam, miss_p=miss_p)

	def reset_generator(self, **kwargs):
		"""
		Updates the generator with the new keyword arguments (kwargs)

		Args:
			kwargs: Inputs to the data generator to change
		"""

		# Reset the generator parameters to the new parameters
		for arg in kwargs.items():
			self.generator = self.generator.mutate(**{arg[0]: arg[1]})

	def reset_tracker(self, tracker):
		"""
		Update the tracker model to a new tracker object
		"""
		self.tracker_model = tracker

	@staticmethod
	def cov_ellipse(mean, cov, zoom_factor=1, p=0.67, mode="mpl"):
		"""
		The cov ellipse returns an ellipse path that can be added to a plot based on the given mean, covariance matrix
		zoom_factor, and the p-value

		Args:
			mean (ndarray): set of coordinates representing the center of the ellipse to be plotted
			cov (ndarray): covariance matrix associated with the ellipse
			zoom_factor (int) : can be tweaked to make ellipses larger
			p (float): the confidence interval. Default: 0.67, which is 1 sigma

		Returns:
			Ellipse: return the Ellipse created.
		"""
		# s takes into account the p-value given
		if type(mean) != np.ndarray:
			mean = np.array(mean)
		mean.shape = (2,1)

		# Eigendecompose the covariance matrix
		cov = cov.round(decimals=10)
		w, v = np.linalg.eig(cov)
		# Calculate the rotation of the ellipse and size of axes
		ang = (np.pi / 2) - np.arctan2(v[0, 0], v[1, 0])
		width = 2 * np.sqrt(chi2.ppf(p, 2) * w[0])
		height = 2 * np.sqrt(chi2.ppf(p, 2) * w[1])

		#Decide whether we are drawing an ellipse for Plotly or Matplotlib (default matplotlib)
		if mode == "plotly":
			# Set number of points to draw in the ellipse
			N = 100
			# ellipse parameterization with respect to a system of axes of directions a1, a2
			t = np.linspace(0, 2 * np.pi, N)
			xs = width * np.cos(t)
			ys = height * np.sin(t)
			R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
			#Return an array of points that will be plotted to form an ellipse
			return np.dot(R, np.array([xs, ys])) + mean[:, -1][:, np.newaxis]
		else:
			#Create ellipse object for use in Matplotlib
			ellipse = Ellipse(xy=mean, width=zoom_factor*width, height=zoom_factor*height, angle=ang, edgecolor='g', fc='none', lw=1)
			return ellipse

	@staticmethod
	def clean_measure(measure):
		output = [point for sublist in measure for point in sublist]
		output = np.array(output).squeeze().T
		return output

	@staticmethod
	def clean_trajectory(trajectories):
		"""
		Converts a single trajectory or process from a dictionary of lists of state vectors to a list of numpy arrays
		representing the position at each time step for plotting.
		"""
		output = []

		# Determine how many unique trajectories are contained within the current
		# trajectories dictionary
		potential_keys = []
		for step in trajectories:
			potential_keys += list(step.keys())

		all_keys = []
		for key in potential_keys:
			if key not in all_keys:
				all_keys.append(key)

		# Need to ensure all_keys is sorted with integers first
		# so that trajectories are plotted correctly
		true_keys = [key for key in all_keys if type(key) is int]
		true_keys.sort()
		false_keys = [key for key in all_keys if type(key) is not int]
		all_keys = true_keys + false_keys

		num_keys = len(all_keys)

		# Iterate through each time step and allocate either the given xk or None
		# to the trajectory arrays
		i = 0
		first_keys = list(trajectories[0].keys())
		while i < num_keys:
			if all_keys[i] in first_keys:
				output.append(trajectories[0][all_keys[i]])
			else:
				output.append(np.array([[np.nan],[np.nan],[np.nan],[np.nan]]))
			i+=1

		for traj in trajectories[1:]:
			for i, key in enumerate(all_keys):
				if key in traj.keys():
					output[i] = np.append(output[i], traj[key], axis=1)
				else:
					output[i] = np.append(output[i], np.array([[np.nan],[np.nan],[np.nan],[np.nan]]), axis=1)

		return output



	@staticmethod
	def clean_measure2(measures, correspondences=None):
		"""
		Converts a dict of key: track, value: measures -> the measures list becomes a list of two list

		Args:
			measures: a dictionary of measurements received at each time step. Requires same data structure as output of DataGenerator class
			correspondences: a dictionary mapping the keys in the "measures" parameter dictionary to the desired key for the output. Output from "get_best_correspondences" method
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

			# If a dictionary of correspondence keys has been provided, map measurements to the appropriate keys
			# Otherwise, just use the default key
			if correspondences is not None:
				if key in correspondences.keys():
					output[correspondences[key]] = [track_x, track_y]
				else:
					output["Unassigned {}".format(key)] = [track_x, track_y]
			else:
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
	def clean_ellipses(ellipses, mode="mpl"):
		"""
		Returns: an array, each index represents a track and is an array of ellipse objects
		"""
		output = []
		for key, track in ellipses.items():
			track_output = []
			for param_set in track:
				track_output.append(Simulation.cov_ellipse(param_set[0][0:2], param_set[1][2:,2:], mode=mode))
			output.append(track_output)
		return output

	def get_max_correspondence_dist(self, clean_processes):
		"""
		Heuristic for max correspondence distance for the correspondence algorithm below.
		"""
		if len(clean_processes) > 0:
			return 2*max([np.linalg.norm(proc[:,0:-1] - proc[:,1:], axis=0).max() for proc in clean_processes]) + 3 * np.sqrt(self.generator.R[0,0])
		else:
			return np.inf

	def get_best_correspondence(self, max_dist, index=0):
		"""
		Restructures trajectories to assign them the best corresponding object ids from the process
		"""
		# Heuristic for determining the cutoff between a poor filter prediction and an object miss
		# This is a custom heuristic created by the Aerospace research team
		output = []
		# Maintain a list of the objects : trajectory correspondences that have already been generated
		correspondences = {}

		# Iterate through each time step
		for i, proc_step in enumerate(self.processes[index]):
			# Set up a dictionary to represent the new time step that will be output
			new_step = dict()
			# Iterate through each process at the time step that has not already been tracked
			cost = []
			proc_ids_considering = []
			traj_ids_considering = []

			for proc_id, proc in proc_step.items():
				dists = []
				if proc_id not in correspondences.values():
					proc_ids_considering.append(proc_id)
					# Calculate the distance from the not-yet-tracked process to each trajectory
					# that has not yet been assigned
					for traj_id, traj in self.trajectories[index][i].items():
						if traj_id not in correspondences.keys():
							dists.append(np.linalg.norm(proc - traj))
							if traj_id not in traj_ids_considering:
								traj_ids_considering.append(traj_id)
					# Store the distances for this process in a cost array
					# Thus, ROWS are processes and COLS are trajectories
					cost.append(dists)
			cost = np.array(cost)
			# Ensure objects that are too far away are not assigned by making the distance infinity
			if cost.size > 0:
				cost[cost > max_dist] = np.inf

			# Find the best combinations of trajectory and process using the cost matrix
			while cost.size > 0 and (cost != np.inf).all():
				# Calculate each subsequent minimum traj-proc pair and remove this from consideration
				# (greedy algorithm)
				best_proc, best_traj = np.unravel_index(cost.argmin(), cost.shape)
				cost[best_proc, :] = np.inf
				cost[:, best_traj] = np.inf

				# Set the new step to be a key : value pair where...
				# key = the id of the process
				# value = the value of the trajectory associated with said process at this iteration
				correspondences[traj_ids_considering[best_traj]] = proc_ids_considering[best_proc]
			# Generate the output for this time step using the correspondences dictionary
			for traj_id, traj in self.trajectories[index][i].items():
				if traj_id in correspondences.keys():
					new_step[correspondences[traj_id]] = traj
				else:
					new_step["Unassigned {}".format(traj_id)] = traj

			#Add this step as the current time step in the new trajectory output
			output.append(new_step)

		return output, correspondences

	# New algorithm pseudocode below:

	# I do not believe we can use linear sum assignment.
	# This is because if the tracker tracks a "fake" object but fails to track a "real" object,
	# then the fake object predicted track will be allocated to the "real" object
	# even if it is quite far away

	# From https://link.springer.com/content/pdf/10.1155%2F2008%2F246309.pdf we get the following:

	# For each time step:
	#   Establish best possible correspondence between hypotheses and objects
	#   For each correspondence compute error in position estimation
	#   Count all objects for which no hypothesis was output as misses
	#   Count all hypotheses for which no real object exists as FPs
	#   Count all occurrences where wrong object is identified as mismatch errors

	# How do we establish best correspondence?
	# For each time step:
	# For each NEW trajectory start point:
	#   Assign trajectory object id to the id of the closest process that is...
	#       NOT being tracked at the current time step by a closer process (bias towards prev hypothesis)
	#       AND is NOT being tracked by a previous process that is CLOSEST to this process compared to other processes
	#       AND is WITHIN a certain expected distance of the process (otherwise it is a false alarm)
	#   If the trajectory cannot be assigned it is considered a false alarm at that time step
	#
	# After this, all hypotheses will be allocated to processes that are not already being tracked by a previous hypotheses,
	# and whose starting positions are closest to said hypotheses compared to other possible hypotheses
	# The advantage of this method is that it ensures ids are based on the FIRST identification of the object
	# which we want because that is how the satellites will be identified
	# and the hypotheses are matched to the processes in the optimal (closest) manner

	@staticmethod
	def get_traj_keys(best_trajectories):
		"""
		Returns a list of keys from the result of get_best_correspondences to allow correct plotting.
		Used in the dashboard.

		Args:
			best_trajectories (dict): dict storing the measurements associated with each trajectory as output from Simulation.get_best_correspondences

		Returns:
			all_keys (list): list of keys representing the order in which each trajectory should be plotted; aligned with process keys
		"""
		potential_keys = []
		for step in best_trajectories:
			potential_keys += list(step.keys())
		all_keys = []
		for key in potential_keys:
			if key not in all_keys:
				all_keys.append(key)

		# Need to ensure all_keys is sorted with integers first
		# so that trajectories are plotted correctly
		true_keys = [key for key in all_keys if type(key) is int]
		true_keys.sort()
		false_keys = [key for key in all_keys if type(key) is not int]
		all_keys = true_keys + false_keys

		return all_keys

	def get_true_fa_and_num_measures(self, measures, colors):
		true_false_alarms = []
		i = 0
		count = 0
		for color_block in colors:
			k = 0
			for color in color_block:
				count += 1
				if color == 'red':
					true_false_alarms.append([measures[i][k][0][0] + measures[i][k][1][0] * 1j])
				k += 1
			i += 1
		return [true_false_alarms, count]

	def compute_metrics(self, m = 'ame', cut = 10):
		index = len(self.processes.keys()) - 1
		if len(self.processes) > 0:
			process = self.processes[index]
			# ASDF
			process = self.clean_trajectory(process)
		else:
			print("ERROR PROCESS LENGTH 0")
			return
		if len(self.measures) > 0:
			sorted_measures = self.sorted_measurements[index]
			measure = self.clean_measure2(sorted_measures) #THIS IS CHANGED TO 2
		else:
			print("ERROR MEASURE LENGTH 0")
			return
		if len(self.trajectories) > 0:
			output1 = self.trajectories[index]
			output1 = self.clean_trajectory(output1)
		else:
			print("ERROR TRAJECTORY LENGTH 0")
			return
		if len(self.apriori_traj) > 0:
			output2 = self.apriori_traj[index]
			output2 = self.clean_trajectory(output2)
		if len(self.measure_colors) > 0:
			true_false_alarms = self.get_true_fa_and_num_measures(self.measures[index], self.measure_colors[index])
		else:
			print("ERROR COLLORS LENGTH 0")
			return
		if len(self.false_alarms) > 0:
			false_alarms = self.false_alarms[index]
			false_alarms = self.clean_false_alarms(false_alarms) if len(false_alarms) > 0 else []
		else:
			print("ERROR FA LENGTH 0")
			return

		if m == 'ame':
			return Metrics.AME_euclidean(process, output1, cut)
		if m == 'rmse':
			return [Metrics.RMSE_euclidean(process, output1, cut), Metrics.RMSE_euclidean(process, output2, cut)]
		if m == 'atct':
			return Metrics.atct_signed(process, output, cut)
		if m == 'fa':
			return Metrics.false_id_rate(true_false_alarms, false_alarms)
		print("ERROR INVALID METRIC")
