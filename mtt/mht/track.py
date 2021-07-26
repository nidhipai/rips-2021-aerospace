"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

class Track:
    def __init__(self, starting_observations, score, x_hat, P = None):
        self.score = score
        self.x_hat = x_hat
        self.observations = starting_observations  # list of (ts, k), where ts is the timestep and k is the number of the measurement
        # essentially this is the index in tracker.observations
        self.possible_observations = []  # lists possible observations for this timestep, indexes
        self.status = 0

        # set a priori and a posteriori estimate error covariances to all ones (not all zeros)
        if P is None:
            self.P = np.eye(self.n) # posteriori estimate error covariance initialized to the identity matrix
        else:
            self.P = P # posteriori estimate error covariance initialized to the identity matrix
    def run_kalman(self, kalman_filter, measurements):
        self.x_hat_minus, self.P_minus = kalman_filter.time_update(self.x_hat, self.P)
        self.x_hat, self.P = kalman_filter.measurement_update(self.x_hat_minus, self.P_minus, measurements[possible_observations[0]])
        self.possible_observations = []
