"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import numpy as np
from .track import Track
from copy import deepcopy
from mtt.mht.distances_mht import DistancesMHT

class TrackMaintenanceMHT:
	"""
	Scores potential new tracks and creates them if the score is above the threshold
	"""
	def __init__(self, threshold_old_track, threshold_miss_measurement, threshold_new_track, prob_detection, obs_dim, lambda_fa, R, kFilter_model):
		"""
		Args:
			threshold_old_track (numeric): score threshold for creating a new track from an existing object
			threshold_miss_measurement (numeric): score threshold for when a track misses a measurement
			threshold_new_track (numeric): score threshold for creating a new track from a single measurement
			prob_detection (numeric in [0,1]: probability that an object will be detected, 1 - P(missed measurement)
			obs_dim: dimension of observations
			lambda_fa: false alarm density
			R: observation residual covariance matrix
		"""
		self.threshold_old_track = threshold_old_track
		self.threshold_miss_measurement = threshold_miss_measurement
		self.threshold_new_track = threshold_new_track
		self.M = obs_dim
		self.pd = prob_detection
		self.lambda_fa = lambda_fa
		self.R = R
		self.kFilter_model = kFilter_model

	def predict(self, ts, tracks, measurements, num_obj):
		"""
		Scores potential tracks, scores them, immediately deletes tracks with too low a score
		Args:
			ts: current timestep
			tracks: list of tracks from Tracker
			measurements: array of measurements, the values, from Tracker
			num_obj: number of objects we've been keeping track of, used for creating object IDs

		Returns: list of new tracks for this ts, number of objects

		"""
		new_tracks = []
		for track in tracks:
			# consider the case of missed measurement, replicate each of these tracks as if they missed a measurement
			missed_measurement_score = track.score + self.score_no_measurement()
			if missed_measurement_score >= self.threshold_miss_measurement:
				mm_track = deepcopy(track)
				mm_track.score = missed_measurement_score
				mm_track.possible_observations = []
				new_tracks.append(mm_track)

			#Now, for every possible observation in a track, create a new track
			for possible_observation in track.possible_observations:
				score = track.score + self.score_measurement_received(possible_observation, track)
				if score >= self.threshold_old_track:  
					observations = deepcopy(track.observations)
					po_track = Track(observations, score, track.object_id, [measurements[possible_observation]], None,	track.ts)
					po_track.possible_observations = [possible_observation]
					new_tracks.append(po_track)

		# finally, for every measurement, make a new track (assume it is a new object)
		for i, measurement in enumerate(measurements):
			score = 1 
			if score >= self.threshold_new_track:
				starting_observations = {ts: i}
				new_tracks.append(Track(starting_observations, score, num_obj, [measurement]))
				num_obj += 1
		return new_tracks, num_obj

	def score_measurement_received(self, obs, track):

		#scoring occurs here
		m_dis_sq = DistancesMHT.mahalanobis(obs, track, self.kFilter_model) ** 2 # TODO fix
		norm_S = np.linalg.norm(self.R, ord=2) # TODO this may not be the right norm
		score = np.log(self.pd / ((2 * np.pi) ** (self.M / 2) * self.lambda_fa * np.sqrt(norm_S))) - m_dis_sq / 2
		return score

	def score_no_measurement(self):
		#scoring without measurement occurs here
		return np.log(1 - self.pd)
