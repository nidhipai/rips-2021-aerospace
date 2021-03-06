"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""


from itertools import repeat
from copy import deepcopy
import numpy as np
from .track import Track


class MHTTracker:
    def __init__(self, global_kalman, gating, track_maintenance, hypothesis_comp, pruning, starting_pos=None):

        # all the methods
        self.starting_pos = starting_pos
        self.gating = gating # holds the Gating object
        self.track_maintenance = track_maintenance # holds the track maintenance object
        self.hypothesis_comp = hypothesis_comp # holds the hypothesis comp object
        self.pruning = pruning # holds the pruning object
        self.gating.kalman = global_kalman # set the gating object's kalman to the global kalman

        self.tracks = []  # list of tracks
        self.cur_best_hypothesis = []  # holds the current best hypothesis
        self.prev_best_hypotheses = []  # holds the previous best hypothesis

        self.kalman = global_kalman  # holds the global kalman for all tracks
        self.measurements = []  # 2D list of state vectors - each row is a time step
        self.ts = 0  # time steps

        if self.starting_pos is not None:
            self.measurements.append(list(self.starting_pos.values()))
            for i, pos in self.starting_pos.items():
                self.tracks.append(Track({0: i}, 1, pos, i, self.pruning.n, P=self.track_maintenance.P))
            self.cur_best_hypothesis.append(list(range(len(self.starting_pos.values()))))
            self.ts = 1
            self.cur_best_tracks = np.array(self.tracks)[tuple(self.cur_best_hypothesis)]

        # for testing
        self.num_tracks_at_each_timestep = []

    def predict(self, measurements):
        """
        Takes in a list of measurements and performs gating, track maintenance (adding and deleting tracks), hypothesis
        computation, and pruning on the current set of tracks

        Args:
            measurements (list): A list of ndarray representing state vectors of all measurements at the current time
        """
        self.measurements.append(measurements)
        print("___________ Time step: {} _________________________________".format(self.ts))
        # print("Number of Tracks: {}".format(len(self.tracks)))
        #
        # print("Measurements")
        # print(measurements)


        #print("_MEASUREMENTS_")
        #print(measurements)
        #print("______________")

        # 1) assign all measurements to all tracks in all children of tree, AND...
        # 2) calculate the expected next position for each track using the time update equation
        for track in self.tracks:
            track.possible_observations = list(range(0, len(measurements)))
            track.time_update(self.kalman, self.ts)
            # print("A priori estimate:\n", track.x_hat_minus)

        # Remove possible observations from each track that are determined to be outliers by the gating
        self.gating.predict(measurements, self.tracks)

        # Next, calculate track scores and create new potential tracks
        self.tracks = self.track_maintenance.predict(self.ts, self.tracks, measurements)

        if self.x_lim is None:
            self.set_bound(30, 30)
        lims = [self.x_lim, self.y_lim]

        # Calculate the maximum weighted clique
        # If new tracks can be born, tracks must be confirmed before being added to max weight clique.
        best_tracks_indexes = self.hypothesis_comp.predict(self.tracks, self.track_maintenance.born_p == 0)

        # Save the current best hypothesis to output
        self.cur_best_hypothesis = best_tracks_indexes
        self.cur_best_tracks = np.array(self.tracks)[self.cur_best_hypothesis]

        if len(best_tracks_indexes) > 0:
            self.prev_best_hypotheses.append(best_tracks_indexes)


        print("==========")
        print("BEST HYP: ")
        for track in self.cur_best_tracks:
            print("TRACK ID: ", track.obj_id, "OBS: ", track.observations, "SCORE: ", track.score)
        print("==========")

        # Remove tracks that do not lead to the best hypothesis within a certain number of time steps

        if self.ts > self.pruning.n: # might be >=
            self.pruning.predict(self.tracks, best_tracks_indexes)

        # Run the Kalman Filter measurement update for each track
        i = 0
        for track in self.tracks:
            #print("A posteriori estimate:\n", track.x_hat)
            # Printing track index
            # print("Track {} Score:".format(i), track.score)
            # Printing track object id
            print("Track {} Score:".format(track.obj_id), track.score)
            track.measurement_update(self.kalman, measurements, self.ts)
            i += 1

        # for testing, save how many tracks there are at each time step
        self.num_tracks_at_each_timestep.append(len(self.tracks))

        # Indicate that one time step has passed
        self.ts += 1

    def set_bound(self, x_lim, y_lim):
        self.x_lim = x_lim
        self.y_lim = y_lim

    def get_all_trajectories(self):
        """
        returns the list of tracks along with their a posteriori estimates over the length of that
        tracjectory's lifetime.

        Returns:
            result (list): list of trajectories
        """

        result = []
        for ts in range(0, self.ts):  # iterate over timesteps
            for i, track in enumerate(self.tracks):
                if ts in track.observations:
                    result[ts][i] = self.measurements[ts][track.observations[ts]]
                else:
                    result[ts][i] = list(repeat([np.nan], 4))
        return result

    def get_best_trajectory(self):
        """
        Outputs hypothesized trajectory from current best hypothesis
        in format used by the Simulation class

        Returns:
            result (list): a list containing the best trajectory.
        """

        result = []
        result.append(self.starting_pos)
        for t in range(1, self.ts):
            step = dict()
            for i, track in enumerate(self.cur_best_tracks[::-1]):
                step[i] = track.aposteriori_estimates[t]
            result.append(step)
        return result

    def get_trajectories(self):
        """
        Outputs hypothesized trajectory prediction from best hypothesis at
        current time step in format used by the Simulation class.

        This is used to obtain predictions over time, so we can analyze how well
        the algorithm performs.

        Returns:
            result (dict): A dictionary containing time steps as the keys and the trajectories as the values.
        """

        result = dict()
        print(self.cur_best_tracks)
        for track in self.cur_best_tracks:
            if track.confirmed() or self.track_maintenance.born_p == 0:
                result[track.obj_id] = track.x_hat
        return result

    def get_apriori_traj(self):
        """
        Outputs hypothesized a priori estimates from best hypothesis at
        current time step in format used by the Simulation class.

        This is used to obtain predictions over time, so we can analyze how well
        the algorithm performs.

        Returns:
            result (dict): A dictionary with the time steps as the keys and the a priori estimates as the values.
        """

        result = dict()
        for track in self.cur_best_tracks:
            if track.confirmed() or self.track_maintenance.born_p == 0:
                result[track.obj_id] = track.x_hat_minus
        return result

    def get_ellipses(self, mode="apriori"):
        """
        Returns either the a priori ellipses or the a posteriori ellipses.

        Args:
            mode (str): either "apriori" or "a posteori", this specifies the ellipses the user
            wants to access.

        Returns:
            ellipses (dict): a dictionary with the track id as the key and a list containing either
            the a priori estimate and the a priori error covariance or the a posteriori estimate and
            the a posterori error covariance.
        """

        ellipses = dict()

        for track in self.cur_best_tracks:
            if track.confirmed() or self.track_maintenance.born_p == 0:
                if mode == "apriori":
                    ellipses[track.obj_id] = [track.x_hat_minus, track.P_minus]
                else:
                    ellipses[track.obj_id] = [track.x_hat, track.P]
        return ellipses

    def get_sorted_measurements(self):
        """
        Returns the sorted measurements with their respective time step.
        Dictionary entries represent the measurements for an object at a given time step.
        Each key represents the object id of the track as it is stored (no correspondence is calculated)

        Returns:
            result (dict): A dictionary with the object id as the key and the sorted measurements as the
            values.
        """
        result = dict()

        # Test whether each track in the best hypothesis is confirmed.
        # If it is, then add it to the output
        for track in self.cur_best_tracks:
            if track.confirmed() or self.track_maintenance.born_p == 0:
                obs = track.observations[max(track.observations.keys())]
                if obs is not None:
                    result[track.obj_id] = self.measurements[-1][obs]

        return result

    def get_best_measurements(self):
        """
        Returns the best measurements for the most recent time step.
        Dictionary entries represent the measurements for an object at a given time step.
        Each key represents the object id of the track as it is stored (no correspondence is calculated)

        Returns:
            result (dict): A dictionary with the object id as the key and the sorted measurements as the
            values.
        """
        result = dict()

        # Test whether each track in the best hypothesis is confirmed.
        # If it is, then add all measurements to the output

        for track in self.cur_best_tracks:
            if track.confirmed() or self.track_maintenance.born_p == 0:
                result[track.obj_id] = list()
                for time, obs in track.observations.items():
                    if obs is not None:
                        result[track.obj_id].append(self.measurements[time][obs])

        return result

    def get_false_alarms(self):
        """
        Gets a list of false alarms at each time step in the data format required by the Simulation class

        Returns:
            result (list): list of all false alarms for the current time step.
        """
        # false alarms are measurements that do not belong to any track in the best global hypothesis
        # the best global hypothesis should not contain tracks that may be false alarms, but that hasn't been done yet

        time = self.ts - 1  # since ts is incrememented at the end of predict
        possible_measurements = list(range(len(self.measurements[-1])))  # get indices of all possible measurements
        # Iterate through our best hypothesis to find which measurements should not be included as false alarms
        for track in self.cur_best_tracks:
            # Test if the track is confirmed yet; if not, it is considered a false alarm
            if track.confirmed() or self.track_maintenance.born_p == 0:  # this is redundant later because cur_best_tracks should all be confirmed
                # If the track records the time step, set the observation in the track as not a false alarm
                if time in track.observations.keys() and track.observations[time] is not None:
                    possible_measurements.remove(track.observations[time])
        # any measurement that is not in a "good" (confirmed and in best hyp) track is a false alarm
        result = [self.measurements[-1][p] for p in possible_measurements if p is not None]
        return result


    def clear_tracks(self, lam=None, miss_p=None):
        """
        Resets the entire tracker object so that there are no tracks, measurements or anything of the sort.
        Args:
            lam (float): the frequency of the false alarms
            miss_p (float): the frequency of the missed measurements
        """
        self.tracks = []  # list of tracks
        self.cur_best_hypothesis = []  # holds the current best hypothesis
        self.prev_best_hypotheses = []  # holds the previous best hypothesis

        self.measurements = []  # 2D list of state vectors - each row is a time step
        self.ts = 0  # time steps

        if self.starting_pos is not None:
            self.measurements.append(list(self.starting_pos.values()))
            for i, pos in self.starting_pos.items():
                self.tracks.append(Track({0: i}, 1, pos, i, self.pruning.n, P=self.track_maintenance.P))
            self.cur_best_hypothesis.append(list(range(len(self.starting_pos.values()))))
            self.ts = 1
            self.cur_best_tracks = np.array(self.tracks)[tuple(self.cur_best_hypothesis)]

        # for testing
        self.num_tracks_at_each_timestep = []

        if lam is not None:
            self.track_maintenance.lambda_fa = lam
        if miss_p is not None:
            self.track_maintenance.pd = 1 - miss_p
