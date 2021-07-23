"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import numpy as np

class MHT_Tracker:
    def __init__(self, global_kalman, gating, track_maintenance, hypothesis_comp, pruning, filter_update):
        self.tracks = []
        self.kalman = global_kalman
        self.measurements = [] # 2D array of state vectors - each row is a time step
        self.ts = 0

        # all the methods
        self.gating = gating
        self.track_maintenance = track_maintenance
        self.hypothesis_comp = hypothesis_comp
        self.pruning = pruning
        self.filter_update = filter_update

    def predict(self, measurements):
        # measurements is an array of state vectors
        self.measurements.append(measurements)

        # 1) assign all measurements to all tracks in all children of tree
        for track in self.tracks:
            track.possible_measurements = list(range(0, len(measurements)))

        # 2) call each method's predict
        self.gating.predict(self.tracks, measurements)
        self.tracks = self.track_maintenance(self.ts, self.tracks)
        self.hypothesis_comp.predict(self.tracks)
        self.pruning.predict(self.ts, self.tracks)

        for tracks in self.tracks:
            # call the filter update for each track

        self.ts += 1
        for track in self.tracks:
            track.possible_observations = []
