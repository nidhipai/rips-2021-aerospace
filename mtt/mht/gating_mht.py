"""
Sal Balkus, Nidhi Pai, Eduardo Sosa, Tony Zeng
RIPS 2021 Aerospace Team
"""

from mtt.mht.distances_mht import *


class DistanceGatingMHT:
	"""
	Removes observations from possible consideration as belonging to a track, for computational efficiency.
	"""
	def __init__(self, error_threshold, expand_gating=0, method="mahalanobis"):
		"""
		Choose the distance metric and distance metric for gating.
		Args:
			error_threshold (float): Distance if method="euclidean", p-value if method="mahalanobis".
					Higher error_threshold leads to a larger gate so it's easier to be under the cutoff
			method ({"euclidean", "mahalanobis"}, optional): Metric of distance, see Distances class.
			expand_gating (float, optional): Interval at which gate should be expanded (a percent of error_threshold).
		"""
		self.error_threshold = error_threshold
		switcher = {
			"euclidean": DistancesMHT.euclidean_threshold,
			"mahalanobis": DistancesMHT.mahalanobis_threshold
		}
		self.distance_function = switcher.get(method)
		self.expand_gating = expand_gating
		self.kalman = None

	def predict(self, measurements, tracks):
		"""
		Removes possible observations from tracks if they are further than the threshold.

		Args:
			measurements (list): List of lists of ndarray representing the measurements at each time step
			tracks (list): list of tracks from MTTTracker
		"""
		for track in tracks:
			expanded_gate_threshold = self.error_threshold + track.num_missed_measurements() * self.expand_gating
			for obs_index in track.possible_observations:
				if not self.distance_function(measurements[obs_index], track, expanded_gate_threshold, kfilter=self.kalman):
					track.possible_observations.remove(obs_index)
