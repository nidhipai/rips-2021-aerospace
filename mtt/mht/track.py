"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import numpy as np

class Track:
    def __init__(self, starting_observations, score, x_hat, obj_id, pruning_n, P = None):
        self.obj_id = obj_id
        self.score = score
        self.x_hat = x_hat
        self.n = self.x_hat.shape[0]
        self.x_hat_minus = self.x_hat
        self.observations = starting_observations  # list of (ts, k), where ts is the timestep and k is the number of the measurement
        self.all_scores = dict()

        # Storage for plotting output
        # Each key is a time step
        self.apriori_estimates = dict()
        self.aposteriori_estimates = dict()
        self.apriori_P = dict()
        self.aposteriori_P = dict()

        # essentially this is the index in tracker.observations
        self.possible_observations = []  # lists possible observations for this timestep, indexes
        self.status = 0
        self.pruning_n = pruning_n

        # set a priori and a posteriori estimate error covariances to all ones (not all zeros)
        if P is None:
            self.P = np.eye(self.n) # posteriori estimate error covariance initialized to the identity matrix
        else:
            self.P = P # posteriori estimate error covariance initialized to the identity matrix
        self.P_minus = self.P

        #testing
        self.test_stats = {}
        self.diff = {}

    def __str__(self):
        return "[OBJ ID: " + str(self.obj_id) + "   OBSERVATIONS: " + str(self.observations) + "   SCORE: " + str(self.score) + "]"

    def time_update(self, kalman_filter, ts):
        """
        We use the kalman filter to do the time update.

        Args:
            kalman_filter (KalmanFilter): A kalman filter object. 
            ts (int): the current time step. 
        """
        self.x_hat_minus, self.P_minus = kalman_filter.time_update(self.x_hat, self.P)
        self.apriori_estimates[ts] = self.x_hat_minus
        self.apriori_P[ts] = self.P_minus

    def measurement_update(self, kalman_filter, measurements, ts):
        """
        We use the kalman filter to do the measurement update.

        Args:
            kalman_filter (KalmanFilter): A kalman filter object. 
            ts (int): the current time step. 
            measurements (list): list of measurements to be used in the measurement update. 
        """
        self.x_hat_minus = np.array(self.x_hat_minus)
        obs = self.observations[max(list(self.observations.keys()))]
        if obs is not None:
            observation = measurements[obs]
        else:
            observation = None

        # Calculate the Kalman measurement update
        self.x_hat, self.P = kalman_filter.measurement_update(self.x_hat_minus, self.P_minus, observation)

        # Store the new values for plotting
        self.aposteriori_estimates[ts] = self.x_hat
        self.aposteriori_P[ts] = self.P

    # def confirmed(self):
    #     num_observation = len(self.observations.values())
    #     return num_observation > self.pruning_n

    def confirmed(self):
        return self.num_observations() > 1

    def num_observations(self):
        return sum(x is not None for x in list(self.observations.values()))

    def num_missed_measurements(self):
        return sum(x is None for x in list(self.observations.values()))

    def num_mm_latest(self):
        earliest_index = max(max(self.observations.keys()) - self.pruning_n + 1, min(self.observations.keys()))
        max_index = max(self.observations.keys())
        return sum(self.observations[ts] is None for ts in range(earliest_index, max_index))

    def test_stat(self):
        if len(self.test_stats) == 0:
            return 0
        earliest_index = max(max(self.test_stats.keys()) - self.pruning_n + 1, min(self.test_stats.keys()))
        max_index = max(self.observations.keys())
        test_stats = [self.test_stats[ts] if ts in self.test_stats.keys() else 0 for ts in range(earliest_index, max_index)]
        return sum(test_stats)
