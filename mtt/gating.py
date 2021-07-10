"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
"""

from .distances import Distances

class DistanceGating:
    def __init__(self, error_threshold, method="euclidean"):
        """
        Choose what kind of distance metric and also the error thresold
        Args:
            error_threshold: distance if method="Euclidean", p-value if method="Mahalanobis" higher means larger gate so it's easier to be under the cutoff
            method: metric of measuring distance - see Distances class
        """
        self.error_threshold = error_threshold
        switcher = {
            "euclidean": Distances.euclidean_threshold,
            "mahalanobis": Distances.mahalanobis_threshold
        }
        self.distance_function = switcher.get(method)

    def predict(self, tracks=None, measurements=None, time=None):
        """
        Removes possible observations from tracks based on distance

        Args:
            tracks: dict of tracks from MTTTracker
            measurements: not used
            time: not used
        """
        if tracks is None:
            print("Error tracks is none in gating")
        for key, track in tracks.items():
            remove_keys = []
            for obs_key, obs in track.possible_observations.items():
                if not self.distance_function(obs, track.filter_model, self.error_threshold):
                    remove_keys.append(obs_key)
            for k in remove_keys:
                track.possible_observations.pop(k)