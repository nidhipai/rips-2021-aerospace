"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import numpy as np
from scipy.stats import chi2
from .track import Track
from copy import deepcopy
from mtt.mht.distances_mht import DistancesMHT
from scipy.stats import binom

class TrackMaintenanceMHT:

    def __init__(self, threshold_old_track, threshold_miss_measurement, threshold_new_track, prob_detection, obs_dim, lambda_fa, R, P, kFilter_model, pruning_n, scoring_method, born_p):
        """
        Args:
            threshold_old_track (float): score threshold for creating a new track from an existing object
            threshold_miss_measurement (float): score threshold for when a track misses a measurement
            threshold_new_track (float): score threshold for creating a new track from a single measurement
            prob_detection (float in [0,1]): probability that object will be detected, 1 - P(missed measurement), [0,1]
            obs_dim (int): dimension of observations, usually 4
            lambda_fa (float): false alarm density, [0,1]
            R (ndarray): observation residual covariance matrix
            P (ndarray): P matrix from Kalman filter, starting value for a new track
            kFilter_model (object): global kalman filter model #TODO - we may not need
            pruning_n (int): the n in N-scan pruning
            scoring_method (string): chi2, distance, log likelihood
        """
        self.threshold_old_track = threshold_old_track
        self.threshold_miss_measurement = threshold_miss_measurement
        self.threshold_new_track = threshold_new_track
        self.M = obs_dim
        self.pd = prob_detection
        self.lambda_fa = lambda_fa
        self.R = R
        self.kFilter_model = kFilter_model
        self.num_objects = 0
        self.pruning_n = pruning_n
        self.scoring_method = scoring_method
        self.born_p = born_p
        if P is None:
            self.P = np.eye(4)
        else:
            self.P = P

    def predict(self, ts, tracks, measurements):
        """
        Scores potential tracks, scores them, immediately deletes tracks with too low a score
        Args:
            ts (int) : current timestep
            tracks (list): list of tracks from Tracker
            measurements (list) : array of measurements, the values, from Tracker

        Returns:
            new_tracks (list): list of new tracks for this ts
        """
        new_tracks = []
        self.missed_measurement(ts, tracks, new_tracks)
        self.add_measurements(ts, tracks, measurements, new_tracks)
        self.new_object(ts, tracks, measurements, new_tracks)
        return new_tracks

    def missed_measurement(self, ts, tracks,  new_tracks):
        """
        Scores a track in the event it misses a measurement.

        Args:
            ts (int) : current timestep
            tracks (list): list of tracks from Tracker
            new_tracks (list): new tracks created at this time step, to return to tracker
        """
        for j, track in enumerate(tracks):

            # consider the case of missed measurement, replicate each of these tracks as if they missed a measurement
            missed_measurement_score = self.score_no_measurement(track, method=self.scoring_method)
            if missed_measurement_score >= self.threshold_miss_measurement:
                mm_track = deepcopy(track)  # So that the original track can be used in add_measurements
                mm_track.score = missed_measurement_score
                mm_track.observations[ts] = None
                mm_track.possible_observations = []  # Reset for the next time step
                mm_track.diff = {}
                new_tracks.append(mm_track)

    def add_measurements(self, ts, tracks, measurements, new_tracks):
        """
        Creates a new track for each track for each possible measurement

        Args:
            ts (int): current timestep
            tracks (list): tracks from tracker
            measurements (list): list of measurements
            new_tracks (list): list of new tracks for this timestep
        """
        for j, track in enumerate(tracks):
            # Now, for every possible observation in a track, create a new track
            # This new tracks should be a copy of the old track, with the new possible
            # observation added to the observations
            for possible_observation in track.possible_observations:
                score, test_stat, diff = self.score_measurement(measurements[possible_observation], track, self.scoring_method)
                track.diff[possible_observation] = diff
                if score >= self.threshold_old_track:
                    # Create a new track with the new observations and score
                    po_track = deepcopy(track)
                    po_track.score = score
                    po_track.test_stats[ts] = test_stat
                    po_track.observations[ts] = possible_observation
                    po_track.possible_observations = []
                    po_track.diff = {}
                    new_tracks.append(po_track)

    def new_object(self, ts, tracks, measurements, new_tracks):
        """
        For every measurement that is NOT in the gate of an existing track, we make a new track.
        Args:
            ts (int): current timestep
            tracks (list): tracks from tracker
            measurements (list): list of measurements
            new_tracks (list): list of new tracks for this timestep
        """
        for i, measurement in enumerate(measurements):

            #measurement_used = False
            #for track in tracks:
                #if i in track.possible_observations:
                    #measurement_used = True
                    #break
            #if not measurement_used:
            p_not_fa = 1 - (self.lambda_fa / (1 + self.lambda_fa))
            if self.scoring_method == "distance":
                if len(new_tracks) > 0:
                    score = min([track.score for track in new_tracks]) - 1
                else:
                    score = -1
            else:
                # dists = [DistancesMHT.mahalanobis(measurement, track, self.kFilter_model) for track in tracks]
                # nearest_track = tracks[dists.index(min(dists))]
                # score = 1 - self.score_measurement(measurements, nearest_track)
                #score = .00001
                p = self.closest_track(i, tracks)
                if p is not None:
                    score = p_not_fa * (1 - p) * self.born_p
                elif self.lambda_fa > 0:
                    score = p_not_fa * self.born_p
                else:
                    score = p_not_fa

            if score > p_not_fa*self.born_p*self.threshold_new_track:
                print("New Track Created")
                starting_observations = {ts: i}
                new_track = Track(starting_observations, score, measurement, self.num_objects, self.pruning_n, P=self.P)
                new_tracks.append(new_track)
                self.num_objects += 1

    def closest_track(self, measurement_index, tracks):
        min_p = None
        for track in tracks:
            if measurement_index in track.possible_observations:
                diff = track.diff[measurement_index] # a little unclean right now but that's fine
                test_stat = diff.T @ np.linalg.inv(self.R + track.P_minus) @ diff
                test_stat = test_stat[0, 0]
                p = chi2.cdf(test_stat, 3)
                if min_p is None or p < min_p:
                    min_p = p
        return min_p

    def score_measurement(self, measurement, track, method = "chi2"):
        """
        Scores a track given a particular measurement.
        Args:
            measurement (ndarray): A measurement vector, the values not index
            track (Track): A track object.
            method (str): A string which tells the function which scoring method we prefer.
        Returns:
            (float): a score using the chi square values.
        """

        if method == "loglikelihood":
            m_dis_sq = DistancesMHT.mahalanobis(measurement, track, self.kFilter_model) ** 2 # TODO fix
            norm_S = np.linalg.norm(self.R, ord=2) # TODO this may not be the right norm
            score = np.log(self.pd / ((2 * np.pi) ** (self.M / 2) * self.lambda_fa * np.sqrt(norm_S))) - m_dis_sq / 2
            return track.score + score

        elif method == "distance":
            m_dis_sq = DistancesMHT.mahalanobis(measurement, track, self.kFilter_model) ** 2 # TODO fix
            return ((track.score*len(track.observations)) - (m_dis_sq / 2)) / (len(track.observations)+1)

        else:
            # Calculate the test statistic by adding the sum of squared differences between the measurement and the
            # predicted value weighted the expected measurement noise variance to the old test stat
            diff = measurement - track.x_hat_minus
            aug_diff = diff.T @ np.linalg.inv(self.R + track.P_minus) @ diff
            test_stat = track.test_stat() + aug_diff
            test_stat = test_stat[0,0]  # Remove numpy array wrapping
            # Convert the test stat to a probability from the chi 2 distribution
            # We multiply by 4 because there are four independent components of the measurements, so
            # we add four random variables at each time step
            score = 1 - chi2.cdf(test_stat, 4*track.num_observations() + 4 - 1)
            return score, aug_diff, diff

    def score_no_measurement(self, track, method="distance"):
        """
        Scores a track given that there is no particular measurement.
        Args:
            track (Track): A track object.
            method (str): A string which tells the function which scoring method we prefer.
        Returns:
            (float): a score using the chi square values.
        """
        if method == "loglikelihood":
            return track.score + np.log(1 - self.pd)
        # elif method == "distance":  # this makes no sense - why would you increase the score?
        #     return track.score * (1 + self.pd)
        else:  # chi2 method - decrease the previous score
            test_stat = track.test_stat()  #if track.num_observations() == 1 else 20
            score = 1 - chi2.cdf(test_stat, 4 * track.num_observations())
            score = self.bi_factor(score, track)
            return score

    def bi_factor(self, score, track): #ONLY USE FOR MM RN
        #binom_factor = binom.pmf(track.num_consecutive_mm() + 1, len(track.observations.values())+1, 1 - self.pd)
        binom_factor = binom.pmf(track.num_mm_latest() + 1, self.pruning_n + 1, 1 - self.pd)
        return score * binom_factor #* np.log(len(track.observations))
