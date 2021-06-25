"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
Simulation
"""
import numpy as np
import matplotlib.pyplot as plt
import random as random
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
		self.n = generator.n
		self.processes = dict()
		self.measures = dict()
		self.trajectories = dict()

	def set_generator(self, generator):
		self.generator = generator

	def set_filter(self, kFilter):
		self.kFilter = kFilter

	def generate(self):
		"""
		Generates process and measurement data
		"""
		random.seed(self.seed_value)
		process = self.generator.process()
		self.processes[len(self.processes.keys())] = process
		self.measures[len(self.measures.keys())] = self.generator.measure(process)

	def predict(self, index=None, x0=None, Q=None, R=None, H=None, u=None):
		output = np.empty((self.n * 2, 1))
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

		f = self.generator.process_function()
		jac = self.generator.process_jacobian()
		h = self.generator.measurement_function()

		kfilter_model = self.kFilter(x0, f, jac, h, Q, R, H, u)

		for i in range(self.generator.ts):
			measure_t = self.measures[index][:, i]
			measure_t.shape = (self.n, 1)
			self.kfilter.predict(measure_t)
			kalman_output = self.kfilter.get_current_guess()
			output = np.append(output, kalman_output, axis=1)
		self.trajectories[len(self.trajectories.keys())] = output[:, 1:]  # delete the first column (initial data)

	def plot(self, index=None, title="Position of Object", x_label="x", y_label="y", z_label="z"):
		if index is None:
			process = self.processes[len(self.processes.keys())-1]
			measure = self.measures[len(self.measures.keys())-1]
			output = self.trajectories[len(self.processes.keys())-1]
		else:
			process = self.processes[index]
			measure = self.measures[index]
			output = self.trajectories[index]

		if self.n == 2:
			plt.scatter(process[0], process[1], s=5, alpha=0.8)
			plt.scatter(measure[0], measure[1], s=5, alpha=0.8)
			plt.scatter(output[0], output[1], s=5, alpha=0.8, color='black')
			plt.title(title)
			plt.xlabel(y_label)
			plt.ylabel(x_label)
			plt.legend(["Process", "Measure", "Filter"])
			plt.show()
		elif self.n == 3:
			ax = plt.axes(projection='3d')
			ax.scatter3D(process[0], process[1], process[2], s=5, alpha=0.8)
			ax.scatter3D(measure[0], measure[1], measure[2], s=5, alpha=0.8)
			ax.scatter3D(output[0], output[1], output[2], s=5, alpha=0.8, color='black')
			ax.set_xlabel(x_label)
			ax.set_ylabel(y_label)
			ax.set_zlabel(z_label)
			plt.legend(["Process", "Measure", "Filter"])
			plt.show()
		else:
			print("Number of dimensions cannot be graphed.")

def main():
	ds = 2  # dimensions
	t = 100  # number of time steps to run
	dt = 0.1  # time step length
	ep_mag = 0.1  # process noise variation
	ep_dir = np.array([1, 1])
	nu = 0.2  # measurement noise variation (both position and velocity)
	initial = np.ones(ds * 2)
	initial.shape = (ds * 2, 1)

	gen = SmartTruck(initial, t, dt, ep_mag, ep_dir, nu)
	zero_matrix = np.zeros((2 * ds, ))  # create an array of zeros for future use
	# Q = np.block([[zero_matrix, zero_matrix], [zero_matrix, np.eye(2 * ds) * ep]])  # process covariance matrix
	# R = np.eye(n) * nu  # measurement covariance matrix
	kfilter = KalmanFilter(initial, gen.A, gen.Q * 2, gen.R * 2, gen.H)
	sim = Simulation(gen, kfilter)
	sim.generate()
	sim.predict()
	sim.scatter()
	
if __name__ == "__main__":
	main()
