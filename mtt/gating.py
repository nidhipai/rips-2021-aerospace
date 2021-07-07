import numpy as np

# class Gating:
#     def __init__(self):


class DistanceGating:
    def __init__(self, error_threshold, method="euclidean"):
        self.error_threshold = error_threshold
        # rewrite this when we get there to include mahalanobis distance and stuff
        if method == "euclidean":
            self.distance_function = self.euclidean
        elif method == "mahalanobis":
            self.distance_function = self.mahalanobis

    def predict(self, tracks=None):
        if tracks is None:
            print("Error tracks is none in gating")
        for track in tracks:
            for observation in track.possible_observations:
                if self.distance_function(observation, track) < self.error_threshold:
                    track.possible_observations.remove(observation)

    def euclidean(self, measurement, track):
        return np.linalg.norm(measurement, track.get_latest_prediction())

    def mahalanobis(self, measurement, track):
        error = measurement - track.kfilter.h(track.kfilter.x_hat_minus)
        return np.sqrt(error.T @ np.linalg.inv(track.kfilter.R) @ error)

# include short arc gating and the other gating methods
