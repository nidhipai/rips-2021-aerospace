"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""
from itertools import repeat

class MHTTracker:
    def __init__(self, global_kalman, gating, track_maintenance, hypothesis_comp, pruning):
        self.tracks = []
        self.kalman = global_kalman
        self.measurements = [] # 2D array of state vectors - each row is a time step
        self.ts = 0
        self.num_objects = 0 # total number of objects, including dead ones

        # all the methods
        self.gating = gating
        self.track_maintenance = track_maintenance
        self.hypothesis_comp = hypothesis_comp
        self.pruning = pruning
        self.gating.kalman = global_kalman

    def predict(self, measurements):
        # measurements is an array of state vectors
        self.measurements.append(measurements)

        # 1) assign all measurements to all tracks in all children of tree, AND...
        # 2) calculate the expected next position for each track using the time update equation

        for track in self.tracks:
            track.possible_observations = list(range(0, len(measurements)))
            track.time_update(self.kalman)

        # 3) call each method's predict to process measurements through the MHT pipeline

        # First, remove measurements that are determined to be outliers by the gating
        self.gating.predict(measurements, self.tracks)

        # Next, calculate track scores
        self.tracks, self.num_objects = self.track_maintenance.predict(self.ts, self.tracks, measurements, self.num_objects)
        best_tracks_indexes = self.hypothesis_comp.predict(self.tracks)
        # print(best_tracks_indexes)
        # TODO save most likely hypothesis (can print to the user)
        # self.pruning.predict(self.tracks, best_tracks_indexes)

        # Run the Kalman Filter measurement update for each track
        for track in self.tracks:
            track.measurement_update(self.kalman, self.measurements, self.ts)

        self.ts += 1

    def get_trajectories(self):
        result = []
        for ts in range(0, self.ts):  # iterate over timesteps
            result.append(dict())
            for i, track in enumerate(self.tracks):
                if ts in track.observations:
                    result[ts][i] = self.measurements[ts][track.observations[ts]]
                else:
                    result[ts][i] = list(repeat([None], 4))
        return result
