'''Aerospace Team 2021 - Eduardo Sosa, Nidhi Pai, Sal V Balkus, Tony Zeng'''

import numpy as np
from scipy.optimize import linear_sum_assignment as linsum
from scipy.stats import chi2
import sys

class DataAssociation:
	def predict(self, tracks=None, measurements=None, pvalue = 0.95):

		''' Update the known tracks to the Gating object.

		Args:
		tracks (list) : list of track objects with the current tracks

		'''
		cutoff = chi2.ppf(pvalue, 2)
		linsum_matrix = []

		for track_key, track in tracks.items():
			linsum_matrix.append([None] * len(measurements.keys()))
			if track.stage != 3:  # right now it makes a row for the deleted tracks too, can fix later
				for i, obs in enumerate(measurements):
					if obs in track.possible_observations.values():
						dis, inside_ellipse = self.calculate_mhlb_dis(obs, track.get_current_guess(), track.get_measurement_cov(), cutoff)
						if inside_ellipse:
							linsum_matrix[track_key][i] = dis
		if len(tracks) == 0:
			return

		linsum_matrix = np.array(linsum_matrix)
		# go back in and fill in all the infinities
		for row in range(0, len(linsum_matrix)):
			for col in range(0, len(linsum_matrix[row])):
				linsum_matrix[row][col] = sys.maxsize if linsum_matrix[row][col] is None else linsum_matrix[row][col]

		row_ind, col_ind = linsum(linsum_matrix)

		for index_track in row_ind:
			for index_measurement in col_ind:
				if linsum_matrix[index_track][index_measurement] < sys.maxsize:
					tracks[index_track].add_measurement(measurements[index_measurement])
					linsum_matrix[index_track][index_measurement] = -1
					measurements[index_measurement] = None # so that we know which ones were attributed to tracks

		# get all the tracks without measurements
		for row in range(0, len(linsum_matrix)):
			if -1 not in linsum_matrix[row]:
				tracks[row].add_measurement(None)

		# # get all measurements without tracks
		# for i in range(linsum_matrix.shape[1]):
		# 	if np.all(np.array(linsum_matrix[:, i]) != -1):
		# 		tracks[len(tracks)] = Track()
		# 		#unassigned_measurements.append(measurements[i])


	def calculate_mhlb_dis(self, measurement, prediction, cov, cutoff):
		''' Update the known tracks to the Gating object.

		Args:
		tracks (list) : list of track objects with the current tracks

		'''
		inside_ellipse = False
		error = measurement - prediction
		dis = np.sqrt(error.T @ np.linalg.inv(cov) @ error)
		if dis < cutoff:
			inside_ellipse = True
			return dis, inside_ellipse





