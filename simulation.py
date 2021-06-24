"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
Simulation
"""
import numpy as np
import numpy.linalg as linalg

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from SmartTruck import SmartTruck
from kalmanfilter2 import KalmanFilter

class Simulation:
	def __init__(self, time_steps, dimensions, time_step_length, ep, nu):
		self.num_steps = time_steps
		self.n = dimensions
		self.interval = time_step_length
		self.ep = ep
		self.nu = nu

	def simulate(data_generator):
		process = data_generator.process()
		measure = data_generator.measure(process)
		output = np.empty((self.n*2, 1))
		zero_matrix = np.zeros((self.n,self.n)) # create an array of zeros for future use
		Q = np.block([[zero_matrix,zero_matrix],[zero_matrix,np.eye(self.n)*self.ep]]) # process covariance matrix
		R = np.eye(self.n) * self.nu # measurement covariance matrix
		kfilter = KalmanFilter(initial, gen.A, Q, R, gen.H)
		for i in range(t+1):
		    measure_t = measure[:,i]
		    measure_t.shape = (self.n,1)
		    kfilter.predict(measure_t)
		    kalman_output = kfilter.get_current_guess()
		    output = np.append(output, kalman_output, axis=1)
		self.output = output[:,1:] # delete the first column (initial data)
		self.process = process
		self.measure = measure

	def scatter(self, title, x_label, y_label, z_label, legend):
		if self.n == 2:
			plt.scatter(self.process[0], self.process[1], s=5, alpha=0.8)
			plt.scatter(self.measure[0], self.measure[1], s=5, alpha=0.8)
			plt.scatter(self.output[0], self.output[1], s=5, alpha=0.8, color='black')
			plt.title(title)
			plt.xlabel(y_label)
			plt.ylabel(x_label)
			plt.legend(legend)
			plt.show()
		elif self.n == 3:
		    ax = plt.axes(projection='3d')
		    ax.scatter3D(self.process[0], self.process[1], self.process[2], s=5, alpha=0.8)
		    ax.scatter3D(self.measure[0], self.measure[1], self.measure[2], s=5, alpha=0.8)
		    ax.scatter3D(self.output[0], self.output[1], self.output[2], s=5, alpha=0.8, color='black')
		    plt.zlabel(z_label)
		    plt.show()

		else:
			print("Number of dimensions cannot be graphed.")

def main():
	sim = Simulation(100, 3, 0.1, 0.1, 0.4)
	initial = np.ones(6)
	initial.shape = (6,1)
	gen = SmartTruck(initial, 100, 0.1, 0.1, 0.4)
	sim.simulate(gen)
	sim.scatter("Position","x", "y", "z", ["Process","Measure","Filter"] )
	
if __name__ == "__main__":
	main()
