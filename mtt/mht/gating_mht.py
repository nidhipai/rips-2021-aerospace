"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

from mtt.mht.distances_mht import *

# TODO THIS IS NOT DONE AND IS NOT COMPATIBLE WITH MHT

class DistanceGatingMHT:
    def __init__(self, error_threshold, method="mahalanobis", expand_gating=0):
        """
        Choose what kind of distance metric and also the error thresold
        Args:
            error_threshold: distance if method="euclidean", p-value if method="mahalanobis" higher means larger gate so it's easier to be under the cutoff
            method: metric of measuring distance - see Distances class
            expand_gating: interval at which gate should be expanded (a percent of error_threshold, which is pval for mahalanobis distance)
        """
        self.error_threshold = error_threshold
        switcher = {
            "euclidean": DistancesMHT.euclidean_threshold,
            "mahalanobis": DistancesMHT.mahalanobis_threshold
        }
        self.distance_function = switcher.get(method)
        self.expand_gating = expand_gating
        self.kalman = None

    def predict(self, measurements, tracks=None,):
        """
        Removes possible observations from tracks based on distance

        Args:
            measurements (list): List of list of ndarray representing the measurements at each time step
            tracks (list): list of tracks from MTTTracker
        """

        if tracks is None:
            print("Error. Tracks in none in gating.")
        for track in tracks:
            print(track)
            expanded_gate_threshold = self.error_threshold + track.missed_measurements * self.expand_gating
            for obs_index in track.possible_observations:
                # if not self.distance_function(obs, track.filter_model, self.error_threshold):
                if not self.distance_function(measurements[obs_index], track, self.kalman, expanded_gate_threshold):
                    track.possible_observations.remove(obs_index)
