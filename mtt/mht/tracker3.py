"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""
from itertools import repeat

class MHTTracker:
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
        self.gating.kalman = global_kalman
        self.cur_best_hypothesis = []

    def predict(self, measurements):
        """
        Takes in a list of measurements and performs gating, track maintenance (adding and deleting tracks), hypothesis
        computation, and pruning on the current set of tracks

        Args:
            measurements (list): A list of ndarray representing state vectors of all measurements at the current time
        """
        # measurements is an array of state vectors
        self.measurements.append(measurements)
        print("___________ Time step: {} _________________________________".format(self.ts))
        print("Measurements:\n", measurements)

        # 1) assign all measurements to all tracks in all children of tree, AND...
        # 2) calculate the expected next position for each track using the time update equation

        for track in self.tracks:
            track.possible_observations = list(range(0, len(measurements)))
            track.time_update(self.kalman)
            # print("A priori estimate:\n", track.x_hat_minus)

        # 3) call each method's predict to process measurements through the MHT pipeline

        # First, remove possible observations from each track that are determined to be outliers by the gating
        self.gating.predict(measurements, self.tracks)

        # for i, track in enumerate(self.tracks):
            # print("Possible Observations {}:".format(i), track.possible_observations)

        # Next, calculate track scores and create new potential tracks
        self.tracks = self.track_maintenance.predict(self.ts, self.tracks, measurements)

        # Calculate the maximum weighted clique
        best_tracks_indexes = self.hypothesis_comp.predict(self.tracks)
        # print("Best tracks:", best_tracks_indexes)

        # Save the current best hypothesis to output
        self.cur_best_hypothesis = best_tracks_indexes
        print("Length of best hypothesis: ", len(self.cur_best_hypothesis))
        #self.pruning.predict(self.tracks, best_tracks_indexes)

        # Run the Kalman Filter measurement update for each track
        for track in self.tracks:
            track.measurement_update(self.kalman, measurements)
            # print("A posteriori estimate:\n", track.x_hat)

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

    def get_best_trajectory(self):
        result = []
        for track in self.cur_best_hypothesis:
            print("Number of Posteriori estimates:", len(self.tracks[track].aposteriori_estimates))
        for t in range(self.ts):
            step = {}
            for i, track_id in enumerate(self.cur_best_hypothesis):
                step[i] = self.tracks[track_id].aposteriori_estimates[t]
            result.append(step)
        return result