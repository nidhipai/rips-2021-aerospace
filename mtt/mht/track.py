"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import numpy as np

class Track:
    def __init__(self, starting_observations, score, x_hat, obj_id, pruning_n, P=None):
        self.obj_id = obj_id
        self.score = score
        self.x_hat = x_hat
        self.n = self.x_hat.shape[0]
        self.x_hat_minus = self.x_hat
        self.observations = starting_observations  # dict of (ts, k), where ts is the timestep and k is the number of the measurement
        # TODO if a measurement is missed, then there is a None entry for the ts, but should we make it so there is no entry?

        # we need this to check if a track is confirmed
        self.pruning_n = pruning_n

        # Storage for plotting output
        # Each key is a time step
        self.apriori_estimates = dict()
        self.aposteriori_estimates = dict()
        self.apriori_P = dict()
        self.aposteriori_P = dict()

        # essentially this is the index in tracker.observations
        self.possible_observations = []  # lists possible observations for this timestep, indexes
        #self.status = 0

        # set a priori and a posteriori estimate error covariances to all ones (not all zeros)
        if P is None:
            self.P = np.eye(self.n) # posteriori estimate error covariance initialized to the identity matrix
        else:
            self.P = P # posteriori estimate error covariance initialized to the identity matrix
        self.P_minus = self.P
        self.missed_measurements = 0

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
        if self.observations[max(list(self.observations.keys()))] is not None:
            observation = self.observations[max(list(self.observations.keys()))]
        else:
            observation = None


        # Calculate the Kalman measurement update
        self.x_hat, self.P = kalman_filter.measurement_update(self.x_hat_minus, self.P_minus, observation)

        # Store the new values for plotting
        self.aposteriori_estimates[ts] = self.x_hat
        self.aposteriori_P[ts] = self.P

    def confirmed(self):
        num_observations = sum(x is not None for x in list(self.observations.values()))
        return num_observations > self.pruning_n
