"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
Simulation
"""

import numpy as np
from copy import copy
from .data_generator import DataGenerator

# Child class for data_generator where we simulate a two-D object
# with its position and velocity as part of the state vector


class MultiObjFixed(DataGenerator):
	def __init__(self, xt0, dt, ep_tangent, ep_normal, nu, miss_p=0, lam=0, fa_scale=10, x_lim = 30, y_lim = 30, new_obj_prop = 0):
		"""
		Constructor for the 2DObject Data Generator.

		:param xt0: List of initial state vectors
		:param dt: Length of one single time step
		:param ep_normal: Variance of the change in velocity vector in the normal direction.
		:param ep_tangent: Variance of the change in velocity vector in the tangent direction.
		:param nu: Variance of the measurement noise
		:param miss_p: Probability of missing a measurement
		:param lam: Expected number of false alarms per time step
		:param fa_scale: Scaling of measurement noise on a false alarm

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

		print("NOP", new_obj_prop)
		self.new_obj_prop = new_obj_prop	# prob of new object spawning	TODO
		print("NOP2", self.new_obj_prop)
		self.num_objs = len(xt0) - 1		#								TODO

		# We require our initial state vector to have all 4 needed components:
		# x,y, velocity in the x direction, velocity in the y direction
		if xt0[0].size != 4:
			raise Exception("Length of initial state vector does not equal 4")

		# We set the process noise covariance matrix to
		self.Q = np.diag(np.append(np.zeros(self.dim), np.append(np.array([ep_tangent]), np.array(ep_normal))))
		self.R = np.eye(self.n) * nu

		super().__init__(xt0, dt, self.Q, self.R)

		# Jacobian matrices for the h function and the f function.
		#self.H = np.append(np.eye(self.dim), np.zeros((self.dim, self.dim)), axis=1)
		self.H = np.eye(self.n)
		self.A = np.append(np.append(np.eye(self.dim), np.eye(self.dim) * self.dt, axis=1),
						   np.append(np.zeros((self.dim, self.dim)), np.eye(self.dim), axis=1), axis=0)
		self.nu = nu  # measurement noise variance
		self.dt = dt

	def process_step(self, xt_prevs, rng):
		"""
		Generate the next process state from the previous
		:param xt_prevs: Previous process states for each object
		:param rng: numpy rng object to generate random variable
		:return: State vector of next step in the process
		"""
		output = dict()
		# Iterate through each state in the list of previous object states
		for xt_key, xt_prev in xt_prevs.items():
			if abs(xt_prev[0]) > self.x_lim or abs(xt_prev[1]) > self.y_lim:
				continue
			# calculate the next state and add to output
			output[xt_key] = self.A @ xt_prev + self.dt*self.process_noise(xt_prev, rng)

		# With probability self.new_obj_prop, create new object on side of frame
		if rng.random() < self.new_obj_prop:
			self.num_objs += 1
			side = rng.random()
			c = rng.random() - 0.5
			buff = 0.125
			print("LIMITS", self.x_lim, self.y_lim)
			if side <= 0.25:
				new_x = -self.x_lim + buff
				new_y = c * 2 * self.y_lim
			elif side <= 0.5:
				new_x = c * 2 * self.x_lim
				new_y = self.y_lim - buff
			elif side <= 0.75:
				new_x = self.x_lim - buff
				new_y = c * 2 * self.y_lim
			else:
				new_x = c * 2 * self.x_lim
				new_y = -self.y_lim + buff
			ang = np.arctan2(new_y, new_x)
			perturb = (rng.random() - 0.5) / 10
			ang += perturb
			new_state = np.array([[new_x], [new_y], [np.cos(ang + np.pi)], [np.sin(ang + np.pi)]])
			output[self.num_objs] = new_state

		return output

	def measure_step(self, xts, rng):
		"""
		Generate the next measure from the current process state vector
		:param xts: Current list of state vectors
		:param rng: numpy rng object to generate random variable
		:return: State vector representing measure at the current process state
		"""

		# Iterate through each object state in the input
		output = []
		colors = []
		for xt in xts.values():
			# Calculate whether the measurement is missed
			if rng.random() > self.miss_p:
				output.append(self.H @ xt + self.measure_noise(rng))
				colors.append("black")

			for i in range(rng.poisson(self.lam)):
				output.append(self.H @ xt + self.measure_noise(rng) * self.fa_scale)
				colors.append("red")

		return output, colors

	def measure_noise(self, rng):
		"""
		Generate measure noise
		"""
		#return rng.normal(scale=self.nu, size=(self.dim, 1))
		output = rng.normal(0, np.sqrt(self.nu), self.n)
		output.shape = (4, 1)
		return output

	def process_noise(self, xt, rng):

		"""
		Generate process noise
		:param xt: current state vector
		:param rng: numpy rng object to generate random variable
		:return: vector of noise for each parameter in the state vector
		"""

		# NOTE: if the angle is 90 degrees then 0 is returned
		# Also this uses radians
		pad = np.array([0, 0])
		rotation = self.W(xt)[2:4, 2:4]
		noise = rng.multivariate_normal((0, 0), rotation @ self.Q[2:4, 2:4] @ rotation.T)
		output = np.append(pad, noise)
		output.shape = (4, 1)
		return output

	def W(self, xt):
		ang = np.arctan2(xt[3, 0], xt[2, 0])
		c = np.cos(ang)
		s = np.sin(ang)

		return np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, c, -s], [0, 0, s, c]])

	def process_function(self, xt, u):
		return self.A @ xt

	def process_jacobian(self, xt, u):
		return self.A

	def measurement_function(self, xt):
		return self.H @ xt

	def measurement_jacobian(self, xt):
		return self.H

	def mutate(self, **kwargs):
		clone = copy(self)
		for arg in kwargs.items():
			setattr(clone, arg[0], arg[1])
		return MultiObjFixed(clone.xt0, clone.dt, clone.ep_tangent, clone.ep_normal, clone.nu, clone.miss_p, x_lim = clone.x_lim, y_lim = clone.y_lim, new_obj_prop = clone.new_obj_prop)
