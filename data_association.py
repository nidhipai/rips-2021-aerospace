'''Aerospace Team 2021 - Eduardo Sosa, Nidhi Pai, Sal V Balkus, Tony Zeng'''

import numpy as np
import scipy.optimize.linear_sum_assignment as linsum

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


	def predict(self, measurements, sigma = 2):
		''' Update the known tracks to the Gating object.

		Args:
		tracks (list) : list of track objects with the current tracks

		'''
		#first, see which measurements are in the ellipse
		#then, pick the closest one
		#if two observations are picked by the same track, the measurement with the lowest distance gets the measurement
		#the second 
		linsum_matrix = []
		index = 0
		for track in self.tracks:
			linsum_matrix.append([])
			for measurement in measurements:
				distance = self.calculate_mhlb_dis(measurement, track.get_current_guess(), track.get_measurement_cov())
				linsum_matrix[index].append(distance)
				index +=1
		linsum_matrix = linsum(linsum_matrix)

		

		return self.tracks

	def calculate_mhlb_dis(self, measurement, prediction, cov):
		''' Update the known tracks to the Gating object.

		Args:
		tracks (list) : list of track objects with the current tracks

		'''

		error = measurement - prediction
        return np.sqrt(error.T @ np.linalg.inv(cov) @ error)




