"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import numpy as np
from scipy.stats import chi2
from .track import Track
from copy import deepcopy
import matplotlib.pyplot as plt
from mtt.mht.distances_mht import DistancesMHT

class TrackMaintenanceMHT:

    def __init__(self, threshold_old_track, threshold_miss_measurement, threshold_new_track, prob_detection, obs_dim, lambda_fa, R, kFilter_model, pruning_n):
        """
        Args:
            threshold_old_track (numeric): score threshold for creating a new track from an existing object
            threshold_miss_measurement (numeric): score threshold for when a track misses a measurement
            threshold_new_track (numeric): score threshold for creating a new track from a single measurement
            prob_detection (numeric in [0,1]: probability that an object will be detected, 1 - P(missed measurement)
            obs_dim: dimension of observations
            lambda_fa: false alarm density
            R: observation residual covariance matrix
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
        self.record_scores = False

    def predict(self, ts, tracks, measurements):


        """
        Scores potential tracks, scores them, immediately deletes tracks with too low a score
        Args:
            ts (int) : current timestep
            tracks (list): list of tracks from Tracker
            measurements (list) : array of measurements, the values, from Tracker
            num_obj (int) : number of objects we've been keeping track of, used for creating object IDs

        Returns: 
            new_tracks (list): list of new tracks for this ts, number of objects

        """
        score_method = "chi"

        new_tracks = []
        for j, track in enumerate(tracks):
            # consider the case of missed measurement, replicate each of these tracks as if they missed a measurement

            missed_measurement_score = self.score_no_measurement(track, method=score_method)
            #print("TEST: ", missed_measurement_score)
            if missed_measurement_score >= self.threshold_miss_measurement:
                mm_track = deepcopy(track)
                mm_track.score = missed_measurement_score
                if self.record_scores:
                    track.all_scores[ts] = []
                    track.all_scores[ts].append(self.score_no_measurement(track, method= "loglikelihood"))
                    track.all_scores[ts].append(self.score_no_measurement(track, method= "distance"))
                    track.all_scores[ts].append(self.score_no_measurement(track, method= "chi"))
                mm_track.observations[ts] = None
                mm_track.possible_observations = []
                new_tracks.append(mm_track)
                print("MM", track.obj_id, "OBS: ", track.observations)



            # consider the case of missed measurement, replicate each of these tracks as if they missed a measurement
            # Now, for every possible observation in a track, create a new track
            # This new tracks should be a copy of the old track, with the new possible
            # observation added to the observations
            for possible_observation in track.possible_observations:
                score = self.score_measurement(measurements[possible_observation], track, method=score_method)

                if score >= self.threshold_old_track:
                    # Create a new track with the new observations and score

                    po_track = deepcopy(track)
                    po_track.score = score
                    if self.record_scores:
                        track.all_scores[ts] = []
                        track.all_scores[ts].append(self.score_measurement(measurements[possible_observation], track, method= "loglikelihood"))
                        track.all_scores[ts].append(self.score_measurement(measurements[possible_observation], track, method= "distance"))
                        track.all_scores[ts].append(self.score_measurement(measurements[possible_observation], track, method= "chi"))
                    po_track.observations[ts] = possible_observation
                    po_track.possible_observations = []
                    new_tracks.append(po_track)
                    print("ID, OBS: ", track.obj_id, possible_observation, "OBS: ", track.observations, "SCORE: ", track.score)

        # finally, for every measurement, make a new track (assume it is a new object)
        for i, measurement in enumerate(measurements):
            new_scores = [0,0,0]
            if score_method == "distance":
                if len(new_tracks) > 0:
                    score = min([track.score for track in new_tracks]) - 1
                else:
                    score = -1
            else:
                score = 0.0001
            new_scores[0] = 0.001
            new_scores[2] = 0.001

            if len(new_tracks) > 0:
                new_scores[1]= min([track.score for track in new_tracks]) - 1
            else:
                new_scores[1] = -1
            # TODO: The below line is completely pointless as of right now.
            # Need to replace with the actual probability of a new track appearing
            # This is where the chi-square test could come in...
            # Is this parameter necessary?
            if score >= self.threshold_new_track:
                starting_observations = {ts: i}
                new_track = Track(starting_observations, score, measurement, self.num_objects, self.pruning_n)
                if self.record_scores:
                    new_track.all_scores[ts] = new_scores
                new_tracks.append(new_track)
                self.num_objects += 1

        return new_tracks

    def score_measurement(self, measurement, track, method="distance"):
        """
        Scores a track given a particular measurement. 
        Args:
            measurement (ndarray): A measurement vector. 
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

        # New method: Chi2
        else:
            # First, convert the track score, which is a probability, into a chi2 test statistic
            # We multiply by 4 because there are four independent components of the measurements, so
            # we add four random variables at each time step
            test_stat = chi2.ppf(1 - track.score, 4*len(track.observations))

            # Next, calculate the sum of squared differences between the measurement and the predicted value,
            # weighted by the expected meausurement noise variance
            diff = measurement - track.x_hat_minus
            vel = track.x_hat_minus
            ang = np.arctan2(vel[3][0], vel[2][0])
            vel = np.sqrt(vel[2][0] ** 2 + vel[3][0] ** 2)
            c = np.cos(ang)
            s = np.sin(ang)
            W = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, c, -s], [0, 0, s, c]])
            #test_stat += diff.T @ np.linalg.inv(track.P_minus * (1 + vel)) @ diff
            #test_stat += diff.T @ np.linalg.inv(self.R * (1 + vel)) @ diff
            Q = self.kFilter_model.Q
            test_stat += diff.T @ np.linalg.inv((self.R + W @ Q @ W.T) * (1 + vel)) @ diff
            test_stat = test_stat[0,0] # Remove numpy array wrapping

            # Finally, convert back to a p-value, but with an additional degree of freedom
            # representing the additional time step which has been added
            #print("Test Stat:",test_stat)
            #print("Deg. of free:", 4*len(track.observations) + 4)
            return 1 - chi2.cdf(test_stat, 4*len(track.observations) + 4)

    def score_no_measurement(self, track, method="distance"):

        """
        Scores a track given that there is no particular measurement. 
        Args:
            track (Track): A track object. 
            method (str): A string which tells the function which scoring method we prefer.
        Returns:
            (float): a score using the chi square values. 
        """


        # scoring without measurement occurs here
        if method == "loglikelihood":
            return track.score + np.log(1 - self.pd)
        elif method == "distance":
            return track.score * (1 + self.pd)
        # New method: Chi2
        else:
            # Here we simply recalculate the p-value, but with an additional degree of freedom
            # which represents the time step that passed without a new measurement
            test_stat = chi2.ppf(track.score, 4*len(track.observations))
            return chi2.cdf(test_stat, 4*len(track.observations) + 4)

    # def graph_scores(self):
    #     x_vals =  list(range(0, len(self.scores["distance"])))
    #     # print(self.scores["distance"])
    #     plt.plot(x_vals,self.scores["distance"], color = "red")
    #     plt.plot(x_vals, self.scores["chi"], color = "blue")
    #     plt.show()

