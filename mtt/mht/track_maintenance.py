"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import numpy as np
from .track import Track
from copy import deepcopy
from mtt import Distances

class TrackMaintenance:
    def __init__(self, threshold, prob_detection, obs_dim, lambda_fa, R):
        self.threshold = threshold 
        # TODO might want many thresholds for the 3 uses
        self.M = obs_dim
        self.pd = prob_detection
        self.lambda_fa = lambda_fa
        self.R = R

    def predict(self, ts, tracks, measurements, num_obj):
        # create new tracks and score them, delete tracks that immediately have too low a score
        new_tracks = []
        for track in tracks:
            for possible_observation in track.possible_observations:
                score = track.score + self.score_measurement_received(possible_observation, track)
                if score >= self.threshold:  # make a new track if the score is above the threshold
                    starting_observations = deepcopy(track.observations)
                    starting_observations[ts] = possible_observation
                    new_tracks.append(Track(starting_observations, score, track.object_id))
            # consider the case of missed measurement
            missed_measurement_score = track.score + self.score_no_measurement()
            if missed_measurement_score >= self.threshold:
                track.score = missed_measurement_score
                new_tracks.append(track)
        # consider each as a new track - measurement is an index
        for measurement in measurements:
            score = 1 + measurement  # TODO compute real score for new track
            if score >= self.threshold:
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
