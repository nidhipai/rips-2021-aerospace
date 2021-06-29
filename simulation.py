"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
Simulation
"""
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True

import random as random
from palettable.colorbrewer.qualitative import Dark2_4
from mpl_toolkits import mplot3d
from SmartTruck import SmartTruck
from kalmanfilter2 import KalmanFilter

class Simulation:
	def __init__(self, generator, kFilter, seed_value=1):
		"""
		Constructs a simulation environment for one-line plotting data

		:param generator: Data Generator object
		:param kFilter: function for predicting the trajectory
		"""
		self.seed_value = seed_value
		self.generator = generator
		self.kFilter = kFilter
		self.kFilter_model = None
		self.n = generator.n
		self.processes = dict()
		self.measures = dict()
		self.trajectories = dict()

	def set_generator(self, generator):
		self.generator = generator

	def set_filter(self, kFilter):
		self.kFilter = kFilter

	def generate(self, time_steps):
		"""
		Generates process and measurement data
		"""
		random.seed(self.seed_value)
		process = self.generator.process(time_steps)
		self.processes[len(self.processes.keys())] = process
		self.measures[len(self.measures.keys())] = self.generator.measure(process)

	def predict(self, index=None, x0=None, Q=None, R=None, H=None, u=None):
		output = np.empty((self.n, 1))
		# if any necessary variables for the filter have not been defined, assume we know them exactly
		if x0 is None:
			x0 = self.generator.xt0
		if Q is None:
			Q = self.generator.Q
		if R is None:
			R = self.generator.R
		if H is None:
			H = self.generator.H
		if index is None:
			index = len(self.measures.keys())-1

		f = self.generator.process_function
		jac = self.generator.process_jacobian
		h = self.generator.measurement_function

		self.kFilter_model = self.kFilter(x0, f, jac, h, Q, R, H, u)
		measures = []

		for i in range(self.measures[index][0].size):
			measure_t = self.measures[index][:, i]
			measure_t.shape = (self.n//2, 1)
			measures.append(measure_t)
			self.kFilter_model.predict(measure_t, np.array(measures))
			kalman_output = self.kFilter_model.get_current_guess()
			output = np.append(output, kalman_output, axis=1)
		self.trajectories[len(self.trajectories.keys())] = output[:, 1:]  # delete the first column (initial data)

	def plot(self, index=None, title="Object Position", x_label="x", y_label="y", z_label="z"):
		color = Dark2_4.mpl_colors
		if index is None:
			process = self.processes[len(self.processes.keys())-1]
			measure = self.measures[len(self.measures.keys())-1]
			output = self.trajectories[len(self.processes.keys())-1]
		else:
			process = self.processes[index]
			measure = self.measures[index]
			output = self.trajectories[index]

		# plt.style.use('dark_background')
		fig = plt.figure(figsize = (12,8))
		plt.rcParams.update({'font.size': 22})

		if self.n//2 == 2:
			title = "{}\n x0 = ({},{})\n Q={}, R={}\n seed={}".format(title, str(self.generator.xt0[0,0]), str(self.generator.xt0[1,0]), str(self.generator.Q), str(self.generator.R), self.seed_value)
			plt.plot(process[0], process[1], lw=1.5, markersize = 15,color=color[2], marker=',')
			plt.scatter(measure[0], measure[1], s = 50, lw=1.5,color=color[1], marker='+')
			plt.plot(output[0], output[1], lw=0.4, markersize = 15, color=color[0], marker='.')
			plt.title(title)
			plt.xlabel(y_label)
			plt.ylabel(x_label)
			plt.legend(["Process", "Filter", "Measure"])
			plt.show()
		elif self.n//2 == 3:
			title = title + ", seed=" + str(self.seed_value)
			ax = plt.axes(projection='3d')
			ax.scatter3D(process[0], process[1], process[2], lw=1.5, color=color[2], marker=',')
			ax.scatter3D(measure[0], measure[1], measure[2], lw=0.4, color=color[1], marker='+')
			ax.scatter3D(output[0], output[1], output[2], lw=0.4, color=color[0], marker='.')
			ax.set_xlabel(x_label)
			ax.set_ylabel(y_label)
			ax.set_zlabel(z_label)
			plt.legend(["Process", "Filter", "Measure"])
			plt.show()
		else:
			print("Number of dimensions cannot be graphed.")

		def plot2(self):
			plt.scatter(measure[0], measure[1])


	
if __name__ == "__main__":
	main()
