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


class MultiObjSimple(DataGenerator):
	def __init__(self, xt0, dt, ep_tangent, ep_normal, nu, miss_p=0, lam=0, fa_scale=10):
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

		self.ep_tangent = ep_tangent    # variance of the process noise tangent to the velocity vector
		self.ep_normal = ep_normal		# variance of the process noise normal to the velocity vector
		self.nu = nu					# variance of the measuremet noise.
		self.miss_p = miss_p			# proportion of missed measurements in the generation of the data.
		self.lam = lam
		self.fa_scale = fa_scale

		# We require our initial state vector to have all 4 needed components:
		# x,y, velocity in the x direction, velocity in the y direction
		if xt0[0].size != 4:
			raise Exception("Length of initial state vector does not equal 4")

		# We set the process noise covariance matrix to
		self.Q = np.diag(np.append(np.zeros(self.dim), np.append(np.array(ep_tangent), np.array(ep_normal))))
		self.R = np.eye(self.n) * nu
		#self.R = np.diag(np.append(np.zeros(self.dim), np.append(np.array(nu), np.array(nu))))

		super().__init__(xt0, dt, self.Q, self.R)

		# Jacobian matrices for the h function and the f function.

		#self.H = np.append(np.eye(self.dim), np.zeros((self.dim, self.dim)), axis=1) # Position-only jacobian
		self.H = np.eye(self.n)
		self.A = np.append(np.append(np.eye(self.dim), np.eye(self.dim) * self.dt, axis=1),
						   np.append(np.zeros((self.dim, self.dim)), np.eye(self.dim), axis=1), axis=0)
		self.nu = nu  # measurement noise variance
		self.dt = dt

	def process_step(self, xt_prevs, rng):
		"""
		Generate the next process state from the previous

		Args:
			xt_prevs (list of ndarray): Previous process states for each object
			rng (numpy.Generator): numpy rng object, used to generate random variables

		Returns:
			output (dict of ndarray): Each value is a state vector of next step in the process, with each
			key a unique identifying integer
		"""
		output = dict()
		# Iterate through each state in the list of previous object states
		for xt_key, xt_prev in xt_prevs.items():
			# calculate the next state and add to output
			output[xt_key] = self.A @ xt_prev + self.dt*self.process_noise(xt_prev, rng)
		return output

	def measure_step(self, xts, rng):
		"""
		Generate the next measure from the current process state vector

		Args:
			xts: Current list of state vectors
			rng (numpy.Generator): numpy rng object, used to generate random variable

		Returns:
			output (list of ndarray): list of State vectors representing measures at the current process state
		"""

		# Iterate through each object state in the input
		output = []
		colors = []
		for xt in xts.values():
			# Calculate whether the measurement is missed
			if np.random.rand() > self.miss_p:
				output.append(self.H @ (xt + self.measure_noise(rng)))
				colors.append("black")

			for i in range(rng.poisson(self.lam)):
				output.append(self.H @ (xt + self.measure_noise(rng) * self.fa_scale))
				colors.append("red")

		return output, colors

	def measure_noise(self, rng):
		"""
		Generate measure noise for an individual measurement

		Args:
			rng (numpy.Generator): numpy rng object, used to generate random variable

		Returns:
			ndarray: Random changes in state vector
		"""
		output = rng.normal(0, self.nu, self.n)
		output.shape = (4,1)
		return output

	def process_noise(self, xt, rng):

		"""
		Generate process noise for an individual measurement

		Args:
			xt (ndarray): current state vector
			rng (numpy.Generator): numpy rng object to generate random variable

		Returns:
			ndarray: vector of noise for each parameter in the state vector
		"""

		pad = np.array([0, 0])
		rotation = self.W(xt)[2:4, 2:4]
		noise = rng.multivariate_normal((0, 0), rotation @ self.Q[2:4, 2:4] @ rotation.T)
		output = np.append(pad, noise)
		output.shape = (4, 1)
		return output

	def W(self, xt):
		"""
		Creates rotation matrix which rotates the input state vector to align with the x-y plane

		Args:
			xt (ndarray): current state vector

		Returns:
			ndarray: matrix that rotates the velocity portion of a generated state vector
		"""
		ang = np.arctan2(xt[3, 0], xt[2, 0])
		c = np.cos(ang)
		s = np.sin(ang)

		return np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, c, -s], [0, 0, s, c]])

	def process_function(self, xt, u):
		"""
		Function representing how the state vector changes over one time step

		Args:
			xt (ndarray): current state vector

		Returns:
			ndarray: state vector at next time step
		"""
		return self.A @ xt

	def process_jacobian(self, xt, u):
		"""
		Returns Jacobian of the process function

		Args:
			xt (ndarray): current state vector

		Returns:
			ndarray: Jacobian of the process function; in this case, just the state transition matrix
		"""
		return self.A

	def measurement_function(self, xt):
		"""
		Function that transforms a state transition matrix to a measurement by multiplying by H

		Args:
			xt (ndarray): current state vector

		Returns:
			ndarray: vector representing just the position component of xt
		"""
		return self.H @ xt

	def measurement_jacobian(self, xt):
		"""
		Returns Jacobian of the measurement function

		Args:
			xt (ndarray): current state vector

		Returns:
			ndarray: Jacobian of the measurement function; in this case, just the H matrix
		"""
		return self.H

	def mutate(self, **kwargs):
		"""
		Creates a copy of the current object, but with changed parameters specified in kwargs

		Args:
			**kwargs: May contain any argument in the constructor of this class; represents the parameters to be changed when the object is copied.

		Returns:
			MultiObjSimple: version of the current object with updated parameters specified in kwargs
		"""
		clone = copy(self)
		for arg in kwargs.items():
			setattr(clone, arg[0], arg[1])
		return MultiObjSimple(clone.xt0, clone.dt, clone.ep_tangent, clone.ep_normal, clone.nu,
		                      clone.miss_p, clone.lam, clone.fa_scale)
