"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
Kalman Filter - Discrete
"""

import numpy as np
import numpy.linalg as linalg

class KalmanFilter:
	def __init__(self, xt0, f, A, h, Q, W, R, P=None, H=None, u=0, dt=1):
		"""
		Initialize the Extended Kalman Filter object
		x_hat0 (ndarray): initial state vector
		f (func) : function representing the process
		A (ndarray): Jacobian matrix of the process function
		h (func): function representing the measurement
		Q (ndarray): Covariance matrix of the process noise
		R (ndarray): Covariance matrix of the measurement noise
		H (ndarray): Matrix transforming the process state vector into a measurement state vector
		u (ndarray): Control vector
		"""
		self.Q = Q * dt  # process noise covariance
		self.W = W # rotation matrix
		self.R = R	# measurement noise covariance
		self.f = f	# process function
		self.h = h	# measurement function
		self.A = A	# jacobian of the process function

		# calculate dimension of x
		self.n = xt0.shape[0]


		# set default H if it is not defined
		if H is None:
			self.H = np.eye(self.n, self.n)
		else:
			self.H = H	# jacobian of the measurement function

		self.u = u # optional control input

		# set a priori and a posteriori estimate error covariances to all ones (not all zeros)
		if P is None:
			self.P = np.eye(self.n) # posteriori estimate error covariance initialized to the identity matrix
		else:
			self.P = P # posteriori estimate error covariance initialized to the identity matrix
		self.P_minus = self.P # priori estimate error coviariance matrix initialized to the identity matrix
		self.x_hat = xt0  # set a priori estimate to initial guess
		self.x_hat_minus = xt0	# set a posteriori estimate to initial guess
		self.error_array = [] # array to store our innovations
		self.xt0 = xt0 # for plotting

	def predict(self, measurement=None, measurement_array=None):
		"""
		Update a posteriori estimate based on a priori estimate and measurement
		In case measurements are missing, we can handle this by not accounting for the missed measurement and only the process.

		Args:
			measurement (ndarray): the measurement vector

			measurement_array(ndarray): the measurement array containing all previous measurements
		"""
		if measurement is None:
			self.x_hat_minus = self.f(self.x_hat, self.u)
			self.P_minus = self.A(self.x_hat_minus, self.u) @ self.P @ self.A(self.x_hat_minus, self.u).T + (self.W(self.x_hat_minus) @ self.Q @ self.W(self.x_hat_minus).T)
			self.x_hat = self.x_hat_minus
		else:
			self.x_hat_minus = self.f(self.x_hat, self.u)
			#print((self.W(self.x_hat_minus) @ self.Q @ self.W(self.x_hat_minus).T))
			self.P_minus = self.A(self.x_hat_minus, self.u) @ self.P @ self.A(self.x_hat_minus, self.u).T + (self.W(self.x_hat_minus) @ self.Q @ self.W(self.x_hat_minus).T)
			self.K = self.P_minus @ self.H.T @ linalg.inv(self.H @ self.P_minus @ self.H.T + self.R)
			self.x_hat = self.x_hat_minus + self.K @ (measurement - self.h(self.x_hat_minus))
			self.P = (np.eye(self.n) - self.K @ self.H) @ self.P_minus


	def get_current_guess(self):
		"""
		Returns:
			self.x_hat (ndarray): return the current state vector
		"""
		return self.x_hat

	def get_current_apriori_guess(self):
		"""
		Returns:
			self.x_hat_minus (ndarray): return the current a priori (estimated) state vector
		"""
		return self.x_hat_minus
