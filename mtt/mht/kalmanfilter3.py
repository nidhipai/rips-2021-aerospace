"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
Kalman Filter - Discrete
"""

import numpy as np
import numpy.linalg as linalg

class KalmanFilter:
    def __init__(self, f, A, h, Q, W, R, H=None, u=0, dt=1):
        self.Q = Q * dt  # process noise covariance
        self.W = W # rotation matrix
        self.R = R  # measurement noise covariance
        self.f = f  # process function
        self.h = h  # measurement function
        self.A = A  # jacobian of the process function

        # set default H if it is not defined
        if H is None:
            self.H = np.eye(self.n, self.n)
        else:
            self.H = H  # jacobian of the measurement function

        self.u = u # optional control input

    def time_update(self, x_hat, P):
        """
        Constructs a simulation environment for one-line plotting data
        Args:
            x_hat (ndarray): The a posteriori estimate
            P (ndarray): The a posteriori error covariance
            
        Returns:
            x_hat_minus (ndarray): The a priori estimate
            P_minus (ndarray): The a priori error covariance matrix

        """

        x_hat_minus = self.f(x_hat, self.u).reshape((4,1))
        P_minus = self.A(x_hat_minus, self.u) @ P @ self.A(x_hat_minus, self.u).T + \
                  (self.W(x_hat_minus) @ self.Q @ self.W(x_hat_minus).T)
        return x_hat_minus, P_minus

    def measurement_update(self, x_hat_minus, P_minus, measurement=None):
        """
        Update a posteriori estimate based on a priori estimate and measurement
        In case measurements are missing, we can handle this by not accounting for the missed measurement and only the process.

        Args:
            x_hat_minus (ndarray): Current state vector a priori estimate
            P_minus (ndarray): Current error covariance matrix a priori estimate
            measurement (ndarray): the measurement vector

        Returns:
            x_hat (ndarray): The a posteriori estimate
            P (ndarray): The a posteriori error covariance matrix

        """

        if measurement is None:
            return x_hat_minus, P_minus
        else:
            K = P_minus @ self.H.T @ linalg.inv(self.H @ P_minus @ self.H.T + self.R)
            x_hat = x_hat_minus + K @ (measurement - self.h(x_hat_minus))
            P = (np.eye(x_hat_minus.shape[0]) - K @ self.H) @ P_minus

            return x_hat, P
