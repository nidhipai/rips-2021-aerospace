"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
Simulation
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from SmartTruck import SmartTruck
from kalmanfilter2 import KalmanFilter

class Simulation:
	def __init__(self, data_generator, filter):
		self.data_generator = data_generator
		self.filter = filter
		self.n = data_generator.n

	def generate(self):
		self.process = self.data_generator.process()
		self.measure = self.data_generator.measure(self.process)

	def predict(self):
		output = np.empty((self.n * 2, 1))
		for i in range(self.data_generator.ts):
			measure_t = self.measure[:, i]
			measure_t.shape = (self.n, 1)
			self.filter.predict(measure_t)
			kalman_output = self.filter.get_current_guess()
			output = np.append(output, kalman_output, axis=1)
		self.output = output[:, 1:]  # delete the first column (initial data)

	def scatter(self, title="", x_label="x", y_label="y", z_label="z"):
		if self.n == 2:
			plt.scatter(self.process[0], self.process[1], s=5, alpha=0.8)
			plt.scatter(self.measure[0], self.measure[1], s=5, alpha=0.8)
			plt.scatter(self.output[0], self.output[1], s=5, alpha=0.8, color='black')
			plt.title(title)
			plt.xlabel(y_label)
			plt.ylabel(x_label)
			plt.legend(["Process", "Measure", "Filter"])
			plt.show()
		elif self.n == 3:
			ax = plt.axes(projection='3d')
			ax.scatter3D(self.process[0], self.process[1], self.process[2], s=5, alpha=0.8)
			ax.scatter3D(self.measure[0], self.measure[1], self.measure[2], s=5, alpha=0.8)
			ax.scatter3D(self.output[0], self.output[1], self.output[2], s=5, alpha=0.8, color='black')
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
