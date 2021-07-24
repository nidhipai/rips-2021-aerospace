"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import numpy as np

class MHT_Tracker:
    def __init__(self, global_kalman, gating, track_maintenance, hypothesis_comp, pruning):
        self.tracks = []
        self.kalman = global_kalman
        self.measurements = [] # 2D array of state vectors - each row is a time step
        self.ts = 0

        # all the methods
        self.gating = gating
        self.track_maintenance = track_maintenance
        self.hypothesis_comp = hypothesis_comp
        self.pruning = pruning

    def predict(self, measurements):
        # measurements is an array of state vectors
        self.measurements.append(measurements)

        # 1) assign all measurements to all tracks in all children of tree
        for track in self.tracks:
            track.possible_measurements = measurements

        # 2) call each method's predict
        self.gating.predict(self.tracks, self.measurements)
        self.tracks = self.track_maintenance(self.ts, self.tracks)
        self.hypothesis_comp(self.tracks)

        for track in self.tracks:
            track.run_kalman(global_kalman, self.measurements)

        self.ts += 1
        for track in self.tracks:
            track.possible_observations = []
