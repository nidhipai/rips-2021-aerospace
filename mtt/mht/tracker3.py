"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""
from itertools import repeat
from copy import deepcopy
import numpy as np

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
        self.prev_best_hypotheses = []

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
        # print("Measurements:\n", measurements)

        # 1) assign all measurements to all tracks in all children of tree, AND...
        # 2) calculate the expected next position for each track using the time update equation

        for track in self.tracks:
            track.possible_observations = list(range(0, len(measurements)))
            track.time_update(self.kalman, self.ts)
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
        self.cur_best_tracks = np.array(self.tracks)[self.cur_best_hypothesis]
        # print("Length of best hypothesis: ", len(self.cur_best_hypothesis))


        if len(best_tracks_indexes) > 0:
            self.prev_best_hypotheses.append(best_tracks_indexes)

        # Remove tracks that do not lead to the best hypothesis within a certain number of time steps
        # USING PRUNING CAUSES ERRORS ATM
        if self.ts > 0:
            self.pruning.predict(self.tracks, best_tracks_indexes)

        # Run the Kalman Filter measurement update for each track
        i = 0
        for track in self.tracks:
            # print("A posteriori estimate:\n", track.x_hat)
            # print("Track {} Score:".format(i), track.score)
            track.measurement_update(self.kalman, measurements, self.ts)
            # print("A posteriori estimate:\n", track.x_hat)
            i += 1

        # Indicate that one time step has passed
        self.ts += 1

    def get_all_trajectories(self):
        # Outputs all trajectories
        result = []
        for ts in range(0, self.ts):  # iterate over timesteps
            result.append(dict())
            for i, track in enumerate(self.tracks):
                if ts in track.observations:
                    result[ts][i] = self.measurements[ts][track.observations[ts]]
                else:
                    result[ts][i] = list(repeat([None], 4))
        return result

    def get_trajectories(self):
        """
        Outputs hypothesized trajectory prediction from best hypothesis at
        current time step in format used by the Simulation class.

        This is used to obtain predictions over time, so we can analyze how well
        the algorithm performs.
        """
        result = dict()
        for track in self.cur_best_tracks:
            result[track.obj_id] = track.x_hat
        return result

    def get_apriori_traj(self):
        """
        Outputs hypothesized a priori estimates from best hypothesis at
        current time step in format used by the Simulation class.

        This is used to obtain predictions over time, so we can analyze how well
        the algorithm performs.
        """

        result = dict()
        for track in self.cur_best_tracks:
            result[track.obj_id] = track.x_hat_minus
        return result

    def get_ellipses(self, mode="apriori"):

        ellipses = dict()

        # Iterate through the hypothesis at each time step
        for track in self.cur_best_tracks:
            if mode == "apriori":
                ellipses[track.obj_id] = [track.x_hat_minus, track.P_minus]
            else:
                ellipses[track.obj_id] = [track.x_hat, track.P]
        return ellipses

    def get_sorted_measurements(self):
        # OLD; NOT DONE
        result = dict()

        for track in self.cur_best_tracks:
            result[track.obj_id] = self.measurements[-1][track.observations[max(track.observations.keys())]]
        """
        for t, prev_hypothesis in enumerate(self.prev_best_hypotheses):
            for i, track_id in enumerate(prev_hypothesis):
                if t in self.tracks[track_id].observations.keys():
                    if i not in list(result.keys()):
                        result[i] = []
                    # Add the measurement for this time step and track to the results for said track
                    result[i].append(self.measurements[t][self.tracks[track_id].observations[t]])
        """
        return result

    def get_false_alarms(self):
        """
        Gets a list of false alarms at each time step in the data format required by the Simulation class
        """
        # OLD; NOT DONE

        possible_measurements = list(range(len(self.measurements[-1])))
        for track in self.cur_best_tracks:
            # Remove observation assigned most recently to track
            possible_measurements.remove(
                    track.observations[
                        max(track.observations.keys())
                ]
            )

        """
        for t, prev_hypothesis in enumerate(self.prev_best_hypotheses):
            # Start by setting all measurements as potential false alarms
            all_measurements = []
            possible = list(range(len(self.measurements[t])))
            for i, track_id in enumerate(prev_hypothesis):
                # Check to ensure the track exists at the time step and it contains the observation
                if t in self.tracks[track_id].observations.keys() and self.tracks[track_id].observations[t] in possible:
                    # Remove from the list of false alarms any measurement that was actually used
                    possible.remove(self.tracks[track_id].observations[t])
            # Add the false alarms at this time step to results
            for p in possible:
                all_measurements.append(self.measurements[t][p])
            result[t] = all_measurements
        """
        result = []
        for p in possible_measurements:
            result.append(self.measurements[-1][p])
        return result

    def clear_tracks(self, lam=None, miss_p=None):
        self.tracks = []
        self.measurements = []  # 2D array of state vectors - each row is a time step
        self.ts = 0
        self.cur_best_hypothesis = []
        self.prev_best_hypotheses = []
        self.track_maintenance.num_objects = 0

        if lam is not None:
            self.track_maintenance.lambda_fa = lam
        if miss_p is not None:
            self.track_maintenance.pd = miss_p

