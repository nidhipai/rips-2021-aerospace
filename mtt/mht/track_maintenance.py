"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import numpy as np
from .track import Track
from copy import deepcopy

class Track_Maintenance:
    def __init__(self, threshold):
        self.threshold = threshold

    def predict(self, ts, tracks):
        # create new tracks and score them, delete tracks that immediately have too low a score
        new_tracks = []
        for track in tracks:
            # consider the case of missed measurement
            missed_measurement_score = np.random.rand()
            if missed_measurement_score >= self.threshold:
                new_tracks.append(track)
            for possible_observation in track.possible_observations:
                # TODO calculate score of this new track including the new observation
                score = track.score + np.random.rand()
                if score >= self.threshold:  # make a new track if the score is above the threshold
                    starting_observations = deepcopy(track.observations)
                    starting_observations[ts] = possible_observation
                    new_tracks.append(Track(starting_observations))
