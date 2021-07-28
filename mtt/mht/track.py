"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import numpy as np

class Track:
    def __init__(self, starting_observations, score, x_hat, P = None):
        self.score = score
        self.x_hat = x_hat
        self.n = self.x_hat[0].shape[0]
        self.x_hat_minus = self.x_hat
        self.observations = starting_observations  # list of (ts, k), where ts is the timestep and k is the number of the measurement

        # Storage for plotting output
        self.apriori_estimates = []
        self.aposteriori_estimates = []
        self.apriori_ellipses = []
        self.aposteriori_ellipses = []

        # essentially this is the index in tracker.observations
        self.possible_observations = []  # lists possible observations for this timestep, indexes
        self.status = 0

        # set a priori and a posteriori estimate error covariances to all ones (not all zeros)
        if P is None:
            self.P = np.eye(self.n) # posteriori estimate error covariance initialized to the identity matrix
        else:
            self.P = P # posteriori estimate error covariance initialized to the identity matrix
        self.P_minus = self.P
        self.missed_measurements = 0

    def time_update(self, kalman_filter):
        # Run the time update of the kalman filter and store estimates for plotting
        self.x_hat_minus, self.P_minus = kalman_filter.time_update(self.x_hat, self.P)
        self.apriori_estimates.append(self.x_hat_minus)
        self.apriori_ellipses.append(self.P_minus)

    def measurement_update(self, kalman_filter, measurements):
        self.x_hat_minus = np.array(self.x_hat_minus)
        # Select the measurement to incorporation based on the next observation
        observation = measurements[self.observations[max(list(self.observations.keys()))]]

        # Calculate the Kalman measurement update
        self.x_hat, self.P = kalman_filter.measurement_update(self.x_hat_minus, self.P_minus, observation)

        # Store the new values for plotting
        self.aposteriori_estimates.append(self.x_hat)
        self.aposteriori_ellipses.append(self.P)
