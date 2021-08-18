"""
Sal Balkus, Nidhi Pai, Eduardo Sosa, Tony Zeng
RIPS 2021 Aerospace Team
"""

import numpy as np

class Track:
	"""
	Records of observations, score, and prediction of a track
	"""
	def __init__(self, starting_observations, score, x_hat, obj_id, pruning_n, P = None):
		"""
		Constructor for the Track object

		Args:
			starting_observations (list): Initial observations of the track.
			score (float): Track score.
			x_hat (ndarray): Current a priori estimate of the track.
			obj_id (int): Number identifying which object the track believes it follows.
			pruning_n (int): Pruning parameter
		"""
		self.obj_id = obj_id
		self.score = score
		self.x_hat = x_hat
		self.n = self.x_hat.shape[0]
		self.x_hat_minus = self.x_hat
		self.observations = starting_observations  # list of (ts, k), where ts is the timestep and k is the number of the measurement
		self.all_scores = dict()

		# Storage for plotting output
		# Each key is a time step
		self.apriori_estimates = dict()
		self.aposteriori_estimates = dict()
		self.apriori_P = dict()
		self.aposteriori_P = dict()

		# essentially this is the index in tracker.observations
		self.possible_observations = []  # lists possible observations for this timestep, indexes
		self.status = 0
		self.pruning_n = pruning_n

		# set a priori and a posteriori estimate error covariances to all ones (not all zeros)
		if P is None:
			self.P = np.eye(self.n) # posteriori estimate error covariance initialized to the identity matrix
		else:
			self.P = P # posteriori estimate error covariance initialized to the identity matrix
		self.P_minus = self.P
		self.missed_measurements = 0

		#testing
		self.test_stats = {}
		self.diff = {}

	def __str__(self):
		return "[OBJ ID: " + str(self.obj_id) + "	OBSERVATIONS: " + str(self.observations) + "   SCORE: " + str(self.score) + "]"

	def time_update(self, kalman_filter, ts):
		"""
		We use the kalman filter to do the time update.

		Args:
			kalman_filter (KalmanFilter): A kalman filter object. 
			ts (int): the current time step. 
		"""
		self.x_hat_minus, self.P_minus = kalman_filter.time_update(self.x_hat, self.P)
		self.apriori_estimates[ts] = self.x_hat_minus
		self.apriori_P[ts] = self.P_minus

	def measurement_update(self, kalman_filter, measurements, ts):
		"""
		We use the kalman filter to do the measurement update.

		Args:
			kalman_filter (KalmanFilter): A kalman filter object. 
			ts (int): the current time step. 
			measurements (list): list of measurements to be used in the measurement update. 
		"""
		self.x_hat_minus = np.array(self.x_hat_minus)
		obs = self.observations[max(list(self.observations.keys()))]
		if obs is not None:
			observation = measurements[obs]
		else:
			observation = None

		# Calculate the Kalman measurement update
		self.x_hat, self.P = kalman_filter.measurement_update(self.x_hat_minus, self.P_minus, observation)

		# Store the new values for plotting
		self.aposteriori_estimates[ts] = self.x_hat
		self.aposteriori_P[ts] = self.P

	# def confirmed(self):
	#	  num_observation = len(self.observations.values())
	#	  return num_observation > self.pruning_n

	def confirmed(self):
		"""
		Test if a track is sufficiently established enough to be considered in best hypotheses.
		Returns:
			conf (boolean): TRUE if and only if track has length greater than 1
		"""
		conf = self.num_observations() > 1
		return conf

	def num_observations(self):
		"""
		Returns:
			n (int):
		"""
		n = sum(x is not None for x in list(self.observations.values()))
		return n

	def num_missed_measurements(self):
		"""
		Returns:
			n (int):
		"""
		n = sum(x is None for x in list(self.observations.values()))
		return n

	def num_mm_latest(self):
		"""
		Returns:
			n (int):
		"""
		earliest_index = max(max(self.observations.keys()) - self.pruning_n + 1, min(self.observations.keys()))
		max_index = max(self.observations.keys())
		n = sum(self.observations[ts] is None for ts in range(earliest_index, max_index))
		return n

	def test_stat(self):
		"""
		Computes the chi^2 test statistic of the track for scoring.
		Returns:
			s (float): chi^2 test statistic of the track.
		"""
		if len(self.test_stats) == 0:
			return 0
		earliest_index = max(max(self.test_stats.keys()) - self.pruning_n + 1, min(self.test_stats.keys()))
		max_index = max(self.observations.keys())
		test_stats = [self.test_stats[ts] if ts in self.test_stats.keys() else 0 for ts in range(earliest_index, max_index)]
		s = sum(test_stats)
		return s

