"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
Simulation
"""

import numpy as np
from copy import copy
from .data_generator import DataGenerator

class MultiObjFixed(DataGenerator):
	def __init__(self, xt0, dt, ep_tangent, ep_normal, nu, miss_p=0, lam=0, fa_scale=10, x_lim = 30, y_lim = 30, new_obj_prop = 0):
		"""
		Constructor for the 2DObject Data Generator.

		Args:
			xt0 (list): List of initial state vectors
			dt (int) : Length of one single time step
			ep_normal (float) : Variance of the change in velocity vector in the normal direction.
			ep_tangent (float): Variance of the change in velocity vector in the tangent direction.
			nu (float): Variance of the measurement noise
			miss_p (float): Probability of missing a measurement between 0 and 1.
			lam (int): Expected number of false alarms per time step
			fa_scale (int): Scaling of measurement noise on a false alarm

		"""
		self.dim = 2					# We work in a two dimensional space
		self.n = 4						# dimension of the state vector

		self.ep_tangent = ep_tangent	# variance of the process noise tangent to the velocity vector
		self.ep_normal = ep_normal		# variance of the process noise normal to the velocity vector
		self.nu = nu					# variance of the measuremet noise.
		self.miss_p = miss_p			# proportion of missed measurements in the generation of the data.
		self.lam = lam
		self.fa_scale = fa_scale

		self.x_lim = x_lim					# half-width of frame
		self.y_lim = y_lim					# half-height of frame

		self.new_obj_prop = new_obj_prop	# prob of new object spawning
		self.num_objs = len(xt0) - 1

		if xt0[0].size != 4:
			raise Exception("Length of initial state vector does not equal 4")

		# We set the process noise covariance matrix to
		self.Q = np.diag(np.append(np.zeros(self.dim), np.append(np.array([ep_tangent]), np.array(ep_normal))))
		self.R = np.eye(self.n) * nu

		super().__init__(xt0, dt, self.Q, self.R)

		# Jacobian matrices for the h function and the f function.
		self.H = np.eye(self.n)
		self.A = np.append(np.append(np.eye(self.dim), np.eye(self.dim) * self.dt, axis=1),
						   np.append(np.zeros((self.dim, self.dim)), np.eye(self.dim), axis=1), axis=0)
		self.nu = nu  # measurement noise variance
		self.dt = dt

	def process_step(self, xt_prevs, rng):
		"""
		Generate the next process state from the previous

		Args:
			xt_prevs (list): Previous process states for each object
			rng (numpy.Generator): numpy rng object to generate random variable

		Returns:
			output (ndarray): State vector of next step in the process
		"""
		output = dict()
		# Iterate through each state in the list of previous object states
		for xt_key, xt_prev in xt_prevs.items():
			if abs(xt_prev[0]) > self.x_lim + 1 or abs(xt_prev[1]) > self.y_lim + 1:
				continue
			# calculate the next state and add to output
			output[xt_key] = self.A @ xt_prev + self.dt*self.process_noise(xt_prev, rng)

		if rng.random() < self.new_obj_prop:
			self.num_objs += 1
			new_x = rng.uniform(-self.x_lim, self.x_lim)
			new_y = rng.uniform(-self.y_lim, self.y_lim)
			ang = np.arctan2(new_y, new_x)
			perturb = (rng.random() - 0.5) / 10
			ang += perturb
			new_state = np.array([[new_x], [new_y], [np.cos(ang + np.pi)], [np.sin(ang + np.pi)]])
			output[self.num_objs] = new_state
		return output

	def measure_step(self, xts, rng):
		"""
		Generate the next measure from the current process state vector
		Args:
			xts (list): Current list of state vectors
			rng (numpy.Generator): numpy rng object to generate random variable
		
		Returns:
			output (list): A list containing the next measurements.
			colors (list): A list of colors. 
		"""
		output = []
		colors = []
		for xt in xts.values():
			# Calculate whether the measurement is missed
			if rng.random() > self.miss_p:
				possible_measurement = self.H @ xt + self.measure_noise(rng)
				if abs(possible_measurement[0][0]) < self.x_lim and abs(possible_measurement[1][0]) < self.y_lim:
					output.append(possible_measurement)
					colors.append("black")

		for i in range(rng.poisson(self.lam)):
			possible_measurement = np.append(rng.uniform((-self.x_lim, -self.y_lim), (self.x_lim, self.y_lim), 2), rng.normal(0, np.sqrt(self.nu), 2)*self.fa_scale)
			possible_measurement.shape = (4, 1)
			if abs(possible_measurement[0][0]) < self.x_lim and abs(possible_measurement[1][0]) < self.y_lim:
				output.append(possible_measurement)
				colors.append("red")

		return output, colors

	def measure_noise(self, rng):
		"""
		Generates the measurement noise.
		Args:
			rng (numpy.Generator): numpy rng object to generate random variable
		
		Returns:
			output (ndarray): A list containing the measurement noise.
		"""
		output = rng.normal(0, np.sqrt(self.nu), self.n)
		output.shape = (4, 1)
		return output

	def process_noise(self, xt, rng):
		"""
		Generates the process noise.
		Args:
			xt (ndarray): The current state vector estimate
			rng (numpy.Generator): numpy rng object to generate random variable
		
		Returns:
			output (ndarray): A list containing the process noise.
		"""
		pad = np.array([0, 0])
		rotation = self.W(xt)[2:4, 2:4]
		noise = rng.multivariate_normal((0, 0), rotation @ self.Q[2:4, 2:4] @ rotation.T)
		output = np.append(pad, noise)
		output.shape = (4, 1)
		return output

	def W(self, xt):
		"""
		returns the rotation matrix W.
		Args:
			xt (ndarray): The current state vector estimate
		
		Returns:
			(ndarray): The rotation matrix W.
		"""
		ang = np.arctan2(xt[3, 0], xt[2, 0])
		c = np.cos(ang)
		s = np.sin(ang)

		return np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, c, -s], [0, 0, s, c]])

	def process_function(self, xt, u):
		"""
		returns A@xt
		Args:
			xt (ndarray): The current state vector estimate
			u (ndarray): Placeholder. 
		
		Returns:
			(ndarray): returns A@xt
		"""
		return self.A @ xt

	def process_jacobian(self, xt, u):
		"""
		returns the matrix A. 
		Args:
			xt (ndarray): The current state vector estimate
			u (ndarray): Placeholder. 
		
		Returns:
			(ndarray): returns A.
		"""
		return self.A

	def measurement_function(self, xt):
		"""
		returns H@xt. 
		Args:
			xt (ndarray): The current state vector estimate
		Returns:
			(ndarray): returns H@xt.
		"""
		return self.H @ xt

	def measurement_jacobian(self, xt):
		"""
		returns the matrix H. 
		Args:
			xt (ndarray): The current state vector estimate
		
		Returns:
			(ndarray): returns H.
		"""
		return self.H

	def mutate(self, **kwargs):
		clone = copy(self)
		for arg in kwargs.items():
			setattr(clone, arg[0], arg[1])
		return MultiObjFixed(clone.xt0, clone.dt, clone.ep_tangent, clone.ep_normal, clone.nu, clone.miss_p, lam = clone.lam, fa_scale = clone.lam, x_lim = clone.x_lim, y_lim = clone.y_lim, new_obj_prop = clone.new_obj_prop)
