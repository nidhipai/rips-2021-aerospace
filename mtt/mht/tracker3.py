"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""
from itertools import repeat
from copy import deepcopy

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
        print("Measurements:\n", measurements)

        # 1) assign all measurements to all tracks in all children of tree, AND...
        # 2) calculate the expected next position for each track using the time update equation

        for track in self.tracks:
            track.possible_observations = list(range(0, len(measurements)))
            track.time_update(self.kalman, self.ts)
            print("A priori estimate:\n", track.x_hat_minus)

        # 3) call each method's predict to process measurements through the MHT pipeline

        # First, remove possible observations from each track that are determined to be outliers by the gating
        self.gating.predict(measurements, self.tracks)

        for i, track in enumerate(self.tracks):
            print("Possible Observations {}:".format(i), track.possible_observations)

        # Next, calculate track scores and create new potential tracks
        self.tracks = self.track_maintenance.predict(self.ts, self.tracks, measurements)

        # Calculate the maximum weighted clique
        best_tracks_indexes = self.hypothesis_comp.predict(self.tracks)
        print("Best tracks:", best_tracks_indexes)

        # Save the current best hypothesis to output
        self.cur_best_hypothesis = best_tracks_indexes
        if len(best_tracks_indexes) > 0:
            self.prev_best_hypotheses.append(best_tracks_indexes)

        # Remove tracks that do not lead to the best hypothesis within a certain number of time steps
        #self.pruning.predict(self.tracks, best_tracks_indexes)

        # Run the Kalman Filter measurement update for each track
        for track in self.tracks:
            track.measurement_update(self.kalman, measurements, self.ts)
            print("A posteriori estimate:\n", track.x_hat)

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

    def get_best_trajectory(self):
        """
        Outputs hypothesized trajectory from current best hypothesis
        in format used by the Simulation class
        """
        result = []
        for t in range(self.ts):
            step = dict()
            for i, track_id in enumerate(self.cur_best_hypothesis):
                step[i] = self.tracks[track_id].aposteriori_estimates[t]
            result.append(step)
        return result

    def get_trajectories(self):
        """
        Outputs hypothesized trajectory prediction from best hypothesis at
        each respective time step in format used by the Simulation class.

        This is used to obtain predictions over time, so we can analyze how well
        the algorithm performs.
        """
        result = []
        for t, prev_hypothesis in enumerate(self.prev_best_hypotheses):
            step = dict()
            for i, track_id in enumerate(prev_hypothesis):
                if t in self.tracks[track_id].aposteriori_estimates.keys():
                    step[i] = self.tracks[track_id].aposteriori_estimates[t]
            result.append(step)
        return result

    def get_apriori_traj(self):
        """
        Outputs hypothesized a priori estimates from best hypothesis at
        each respective time step in format used by the Simulation class.

        This is used to obtain predictions over time, so we can analyze how well
        the algorithm performs.
        """

        # NOTE: This outputs a blank dictionary in the first step
        result = []
        for t, prev_hypothesis in enumerate(self.prev_best_hypotheses):
            step = dict()
            for i, track_id in enumerate(prev_hypothesis):
                if t in self.tracks[track_id].apriori_estimates.keys():
                    step[i] = self.tracks[track_id].apriori_estimates[t]
            result.append(step)
        return result

    def get_ellipses(self, mode="apriori"):
        """
		Returns: a dict with keys: tracks and values: array of ellipse params
		"""
        ellipses = dict()

        # Iterate through the hypothesis at each time step
        for t, prev_hypothesis in enumerate(self.prev_best_hypotheses):
            for i, track_id in enumerate(prev_hypothesis):
                if t in self.tracks[track_id].apriori_estimates.keys() and \
                    t in self.tracks[track_id].aposteriori_estimates.keys():

                    # Create new key value pair if we have found a new object
                    if i not in list(ellipses.keys()):
                        ellipses[i] = []

                    # Add ellipses using attributes stored previously in Track object
                    if mode == "apriori":
                        ellipses[i].append([self.tracks[track_id].apriori_estimates[t], self.tracks[track_id].apriori_P[t]])
                    else:
                        ellipses[i].append([self.tracks[track_id].aposteriori_estimates[t], self.tracks[track_id].aposteriori_P[t]])
        return ellipses

    def get_sorted_measurements(self):
        result = dict()
        for t, prev_hypothesis in enumerate(self.prev_best_hypotheses):
            for i, track_id in enumerate(prev_hypothesis):
                if t in self.tracks[track_id].observations.keys():
                    if i not in list(result.keys()):
                        result[i] = []
                    # Add the measurement for this time step and track to the results for said track
                    result[i].append(self.measurements[t][self.tracks[track_id].observations[t]])
        return result

    def get_false_alarms(self):
        """
        Gets a list of false alarms at each time step in the data format required by the Simulation class
        """
        result = dict()
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
        return result
