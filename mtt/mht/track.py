"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import numpy as np

class Track:
    def __init__(self, starting_observations, score, object_id, x_hat, P = None):
        self.score = score
        self.x_hat = []
        self.P = []
        self.observations = starting_observations  # list of (ts, k), where ts is the timestep and k is the number of the measurement
        # essentially this is the index in tracker.observations
        self.possible_observations = []  # lists possible observations for this timestep, indexes
        self.status = 0
        self.object_id = object_id

        # set a priori and a posteriori estimate error covariances to all ones (not all zeros)
        if P is None:
            P = np.eye(4)

        self.P.append(P)  # posteriori estimate error covariance initialized to the identity matrix
        self.x_hat.append(x_hat)
