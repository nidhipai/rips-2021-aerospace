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
        #print("Measurements:\n", measurements)

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
        #print("Best tracks:", best_tracks_indexes)

        # Save the current best hypothesis to output
        self.cur_best_hypothesis = best_tracks_indexes
        self.cur_best_tracks = np.array(self.tracks)[self.cur_best_hypothesis]
        #print("Length of best hypothesis: ", len(self.cur_best_hypothesis))


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

    def get_trajectories(self, startup_time=None):
        """
        Outputs hypothesized trajectory prediction from best hypothesis at
        current time step in format used by the Simulation class.

        This is used to obtain predictions over time, so we can analyze how well
        the algorithm performs.

        Args:
            startup_time: We want to discard predictions from unconfirmed objects, but if ts < startup_time,
            then all objects are treated as confirmed and reported. Unconfirmed means less than pruning.n
            observations.

        Returns: Dict of (obj_id, prediction) for the current timestep
        """
        #startup_time = startup_time if startup_time is not None else self.pruning.n
        startup_time = 0
        result = dict()
        for track in self.cur_best_tracks:
            # this is a bit redundant later because all of the tracks in the best_tracks should be confirmed
            if self.ts <= startup_time or track.confirmed():
                result[track.obj_id] = track.x_hat
        return result

    # def get_trajectories(self):
    #     """
    #     Outputs hypothesized trajectory prediction from best hypothesis at
    #     current time step in format used by the Simulation class.
    #
    #     This is used to obtain predictions over time, so we can analyze how well
    #     the algorithm performs.
    #     """
    #     result = dict()
    #     for track in self.cur_best_tracks:
    #         result[track.obj_id] = track.x_hat
    #     return result

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
        result = dict()
        time = self.ts - 1 # since ts is incremented at the end of the predict method
        for track in self.cur_best_tracks:
            if track.confirmed() and time in track.observations.keys():
                result[track.obj_id] = self.measurements[-1][track.observations[time]]
            # TODO in the missed measurement case, should this be None or just don't add anything??
        return result

    def get_false_alarms(self):
        """
        Gets a list of false alarms at each time step in the data format required by the Simulation class
        """
        # false alarms are measurements that do not belong to any track in the best global hypothesis
        # the best global hypothesis should not contain tracks that may be false alarms, but that hasn't been done yet

        time = self.ts - 1 # since ts is incrememented at the end of predict
        possible_measurements = list(range(len(self.measurements[-1]))) # these are indexes
        for track in self.cur_best_tracks:
            if track.confirmed(): # this is redundant later because cur_best_tracks should all be confirmed
                if time in track.observations.keys() and track.observations[time] is not None:
                    possible_measurements[track.observations[time]] = None
        # any measurement that is not in a "good" (confirmed and in best hyp) track is a false alarm
        result = [self.measurements[-1][p] for p in possible_measurements if p is not None]
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

