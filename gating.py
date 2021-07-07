'''Aerospace Team 2021 - Eduardo Sosa, Nidhi Pai, Sal V Balkus, Tony Zeng'''

import numpy as np

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


	def predict(self, measurements):
		''' Update the known tracks to the Gating object.

		Args:
		tracks (list) : list of track objects with the current tracks

		'''
		for measurement in measurements:
			distance = 

	def calculate_mhlb_dis(self, measurement):
		''' Update the known tracks to the Gating object.

		Args:
		tracks (list) : list of track objects with the current tracks

		'''
	def calculate_euclidian_dis(self, measurement):
		''' Update the known tracks to the Gating object.

		Args:
		tracks (list) : list of track objects with the current tracks

		'''



