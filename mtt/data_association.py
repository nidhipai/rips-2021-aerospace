'''Aerospace Team 2021 - Eduardo Sosa, Nidhi Pai, Sal V Balkus, Tony Zeng'''

import numpy as np
from scipy.optimize import linear_sum_assignment as linsum

class DataAssociation:
	# def __init__(self, kalman_params): MOVE ME TO TRACK MAINTENANCE
	# 	self.kalman_params = kalman_params

	def predict(self, tracks, measurements=None, kalman_params = None):
		''' Update the known tracks to the Gating object.

		Args:
		tracks (list) : list of track objects with the current tracks

		'''
		#first, see which measurements are in the ellipse
		#then, pick the closest one
		#if two observations are picked by the same track, the measurement with the lowest distance gets the measurement
		#the second
		if measurements is None:
			print('error: measurements is none in data_association')
		linsum_matrix = []
		track_index = 0
		for track in tracks:
			linsum_matrix.append([])
			for i, measurement in enumerate(measurements):
				if measurement in track.possible_observations:
					distance = self.calculate_mhlb_dis(measurement, track.get_current_guess(), track.get_measurement_cov())
					linsum_matrix[track_index][i] = distance
			track_index +=1
		linsum_matrix = np.array(linsum_matrix)
		row_ind, col_ind = linsum(linsum_matrix)

		# TO DO: check to make sure that the values aren't infinity
		print('sum ' + str(linsum_matrix[row_ind, col_ind].sum()))

		# for i, track_row in enumerate(linsum_matrix):
		# 	WE NEED TO ADD THE RETURNED MEASUREMENTS TO THE CORRECT TRACK TO PASS TO THE KALMAN FILER
		# also track initiation and deletion


	def calculate_mhlb_dis(self, measurement, prediction, cov):
		''' Update the known tracks to the Gating object.

		Args:
		tracks (list) : list of track objects with the current tracks

		'''

		error = measurement - prediction
		return np.sqrt(error.T @ np.linalg.inv(cov) @ error)




