"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
"""

import numpy as np
from scipy.optimize import linear_sum_assignment as linsum
import sys
from mtt import Distances


class DataAssociation:
	"""
	Finds 1-1 matchings between observations and tracks and adds measurements to tracks, also does a bit of gating
	"""
	def __init__(self, method="euclidean"):
		switcher = {
			"euclidean": Distances.euclidean,
			"mahalanobis": Distances.mahalanobis
		}
		self.distance_function = switcher.get(method)

	def predict(self, tracks=None, measurements=None, time=0, false_alarms=None):
		"""
		Main method of the DataAssociation that performs the matching
		Args:
			tracks: dictionary of tracks from MTTTracker
			measurements: list of column vector measurements
			time: current timestep
			false_alarms not used
		"""

		# BUG: NOT ALL MEASUREMENTS ARE BEING ADDED HERE
		linsum_matrix = [] # matrix of distances that we'll use GNN on

		for track_key, track in tracks.items():  # iterate over all the tracks
			# TODO - should we only do this for undeleted tracks??
			linsum_matrix.append([None] * len(measurements))
			for i, obs in enumerate(measurements):  # iterate over all the measurements for that round
				for p_obs in track.possible_observations.values():  # check if obs is in poss obs but the slow way (redo)
					if np.array_equiv(obs, p_obs):  # this is the only way (that I know) to check if a vector is in an array
						linsum_matrix[track_key][i] = self.distance_function(obs, track.filter_model)
						break
		if len(tracks) == 0:
			return

		# go back in and fill in all the infinities for the pairs of tracks and obs that are too far apart
		for row in range(0, len(linsum_matrix)):
			for col in range(0, len(linsum_matrix[row])):
				linsum_matrix[row][col] = sys.maxsize if linsum_matrix[row][col] is None else linsum_matrix[row][col]

		# solve the linear sum assignment/weighted bipartite matching problem
		row_ind, col_ind = linsum(linsum_matrix)



		# for the pairs that were found, add the measurement to the track
		"""
		for index_track in row_ind:
			for index_measurement in col_ind:
				if linsum_matrix[index_track][index_measurement] < sys.maxsize:
					tracks[index_track].add_measurement(time, measurements[index_measurement])

					# Set the entries that we have checked already to -1 to denote they have been added
					linsum_matrix[index_track][index_measurement] = -1
					measurements[index_measurement] = None # so that we know which obs were attributed to tracks
		"""

		for i in range(len(row_ind)):
			if linsum_matrix[row_ind[i]][col_ind[i]] < sys.maxsize:
				tracks[row_ind[i]].add_measurement(time, measurements[col_ind[i]])
				# Set the entries that we have checked already to -1 to denote they have been added
				linsum_matrix[row_ind[i]][col_ind[i]] = -1
				measurements[col_ind[i]] = None  # so that we know which obs were attributed to tracks



		# address all the tracks without measurements - consider it as a missed measurement
		for row in range(0, len(linsum_matrix)):
			if -1 not in linsum_matrix[row]:
				tracks[row].add_measurement(time, None)
