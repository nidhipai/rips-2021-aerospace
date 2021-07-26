"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

class MHT_Tracker:
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

    def predict(self, measurements):
        # measurements is an array of state vectors
        self.measurements.append(measurements)

        # 1) assign all measurements to all tracks in all children of tree
        for track in self.tracks:
            track.possible_measurements = list(range(0, len(measurements)))

        # 2) call each method's predict
        self.gating.predict(self.tracks, measurements)
        self.tracks, self.num_objects = self.track_maintenance(self.ts, self.tracks, measurements, self.num_objects)
        best_tracks_indexes = self.hypothesis_comp.predict(self.tracks)
        # TODO save most likely hypothesis (can print to the user)
        self.pruning.predict(self.ts, self.tracks, best_tracks_indexes)

        for track in self.tracks:
            x_hat_minus, P_minus = self.kalman.time_update(track.x_hat[-1], track.P[-1])
            measurement = track.observations[self.ts] if self.ts in track.observations.keys() else None
            new_x_hat, new_P = self.kalman.measurement_update(x_hat_minus[-1], P_minus[-1], measurement)
            track.x_hat.append(new_x_hat)
            track.P.append(new_P)

        self.ts += 1
        for track in self.tracks:
            track.possible_observations = []
