"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import numpy as np
from .track import Track
from copy import deepcopy
from mtt import Distances

class Track_Maintenance:
    """
    Scores potential new tracks and creates them if the score is above the threshold
    """
    def __init__(self, threshold_old_track, threshold_miss_measurement, threshold_new_track, prob_detection, obs_dim, lambda_fa, R):
        """
        Args:
            threshold_old_track: score threshold for creating a new track from an existing object
            threshold_miss_measurement: score threshold for when a track misses a measurement
            threshold_new_track: score threshold for creating a new track from a single measurement
            prob_detection: probability that an object will be detected, 1 - P(missed measurement)
            obs_dim: dimension of observations, probably 2
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
            for possible_observation in track.possible_observations:
                score = track.score + self.score_measurement_received(possible_observation, track)
                if score >= self.threshold_old_track:  # make a new track if the score is above the threshold
                    starting_observations = deepcopy(track.observations)
                    starting_observations[ts] = possible_observation
                    new_tracks.append(Track(starting_observations, score, track.object_id))
            # consider the case of missed measurement
            missed_measurement_score = track.score + self.score_no_measurement()
            if missed_measurement_score >= self.threshold_miss_measurement:
                track.score = missed_measurement_score
                new_tracks.append(track)
        # consider each as a new track - measurement is an index
        for measurement in measurements:
            score = 1 + measurement  # TODO compute real score for new track
            if score >= self.threshold_new_track:
                starting_observations = {ts: measurement}
                new_tracks.append(Track(starting_observations, score, num_obj))
                num_obj += 1
        return new_tracks, num_obj

    def score_measurement_received(self, obs, track):
        # TODO make sure that this integrates with the Distances class
        # TODO figure out the variables in the score equation
        m_dis_sq = Distances.mahalanobis() ** 2 # TODO fix
        norm_S = np.linalg.norm(self.R, ord=2) #TODO this may not be the right norm
        return np.log(self.pd / ((2 * np.pi) ** (self.M / 2) * self.lambda_fa * np.sqrt(norm_S))) - m_dis_sq / 2

    def score_no_measurement(self):
        return np.log(1 - self.pd)
