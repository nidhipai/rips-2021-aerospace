'''Aerospace Team 2021 - Eduardo Sosa, Nidhi Pai, Sal V Balkus, Tony Zeng'''

import numpy as np
import scipy.optimize as linsum
import scipy.stats.chi2 as chi2
import sys
import kalmanfilter2 as kf

class DataAssociation:
	'''--'''
	def __init__(self, tracks = None):
		''' ---

		Args:
		tracks (list): 

		'''

		self.tracks = tracks

	def update_tracks(self, tracks):
		''' Update the known tracks to the Gating object.

		Args:
		tracks (list) : list of track objects with the current tracks

		'''

		self.tracks = tracks 


	def predict(self, measurements, pvalue = 0.95):

		''' Update the known tracks to the Gating object.

		Args:
		tracks (list) : list of track objects with the current tracks

		'''
		cutoff = chi2.ppf(pvalue, 2)
		linsum_matrix = []
		track_index = 0
		for track in self.tracks:
			linsum_matrix.append([])
			for i, measurement in enumerate(measurements):
				if measurement in track.possible_observations:
					dis, inside_ellipse = self.calculate_mhlb_dis(measurement, track.get_current_guess(), track.get_measurement_cov(), cutoff)
					if inside_ellipse:
						linsum_matrix[track_index][i] = dis
					else:
						linsum_matrix[track_index][i] = sys.maxsize
			track_index +=1
		row_ind, col_ind = linsum.linear_sum_assignment(np.array(linsum_matrix))

		for index_track in row_ind:
			for index_measurement in col_ind:
				if linsum_matrix[index_track][index_measurement] < sys.maxsize:
					self.tracks[index_track].add_measurement(measurements[index_measurement])
					linsum_matrix[index_track][index_measurement] = 0

		unassigned_measurements = []
		for i in range(linsum_matrix.shape[1]):
			if np.all(np.array(linsum_matrix[:,i]) != 0):
				unassigned_measurements.append(measurements[i])
		return unassigned_measurements


	def calculate_mhlb_dis(self, measurement, prediction, cov):
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





