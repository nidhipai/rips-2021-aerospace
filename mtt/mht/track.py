"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import numpy as np

class Track:
	def __init__(self, starting_observations, score, object_id, x_hat, P = None, ts = 0):
		self.score = score
		self.x_hat = x_hat
		self.n = self.x_hat[0].shape[0]
		self.x_hat_minus = self.x_hat
		self.observations = starting_observations  # list of (ts, k), where ts is the timestep and k is the number of the measurement

		# essentially this is the index in tracker.observations
		self.possible_observations = []  # lists possible observations for this timestep, indexes
		self.status = 0
		self.object_id = object_id

		# set a priori and a posteriori estimate error covariances to all ones (not all zeros)
		if P is None:
			self.P = np.eye(self.n) # posteriori estimate error covariance initialized to the identity matrix
		else:
			self.P = P # posteriori estimate error covariance initialized to the identity matrix
		self.P_minus = self.P
		self.ts = ts
		self.missed_measurements = 0

	def run_kalman(self, kalman_filter, measurements, ts):
		if self.ts != 0:
			self.x_hat_minus, self.P_minus = kalman_filter.time_update(self.x_hat, self.P)
			if len(self.possible_observations) != 0:
				observation = measurements[self.possible_observations[0]]
				self.observations[ts] = self.possible_observations[0]
				self.x_hat, self.P = kalman_filter.measurement_update(self.x_hat_minus, self.P_minus, observation)
				self.possible_observations = []
		self.ts += 1
