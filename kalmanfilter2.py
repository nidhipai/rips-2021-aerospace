"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
Kalman Filter - Discrete
"""

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math

class KalmanFilter:
    def __init__(self, x_hat0, f, A, h, Q, W, R, H=None, u=0, dt=1):
        """
        Initialize the Extended Kalman Filter object
        :param x_hat0: initial state vector
        :param f: function representing the process
        :param A: Jacobian matrix of the process function
        :param h: function representing the measurement
        :param Q: Covariance matrix of the process noise
        :param R: Covariance matrix of the measurement noise
        :param H: Matrix transforming the process state vector into a measurement state vector
        :param u: Control vector
        """
        self.Q = Q * dt  # process noise covariance
        self.W = W # rotation matrix
        self.R = R  # measurement noise covariance
        self.f = f  # process function
        self.h = h  # measurement function
        self.A = A  # jacobian of the process function

        # calculate dimension of x
        self.n = x_hat0.shape[0]

        #Set default H if it is not defined
        if H is None:
            self.H = np.eye(self.n, self.n)
        else:
            self.H = H  # jacobian of the measurement function

        self.u = u

        # set a priori and a posteriori estimate error covariances to all ones (not all zeros)
        self.P = np.eye(self.n)
        self.P_minus = np.eye(self.n)
        self.x_hat = x_hat0  # set a priori estimate to initial guess
        self.x_hat_minus = x_hat0  # set a posteriori estimate to initial guess
        self.error_array = []

    # Update a posteriori estimate based on a priori estimate and measurement
    def predict(self, measurement=None, measurement_array=None):
        # The extended Kalman Filter
        if measurement is None:
            self.x_hat_minus = self.f(self.x_hat, self.u)
            self.P_minus = self.A(self.x_hat_minus, self.u) @ self.P @ self.A(self.x_hat_minus, self.u).T + (self.W(self.x_hat_minus) @ self.Q @ self.W(self.x_hat_minus).T)
            self.x_hat = self.x_hat_minus
        else:
            self.x_hat_minus = self.f(self.x_hat, self.u)
            self.P_minus = self.A(self.x_hat_minus, self.u) @ self.P @ self.A(self.x_hat_minus, self.u).T + (self.W(self.x_hat_minus) @ self.Q @ self.W(self.x_hat_minus).T)
            self.K = self.P_minus @ self.H.T @ linalg.inv(self.H @ self.P_minus @ self.H.T + self.R)
            self.x_hat = self.x_hat_minus + self.K @ (measurement - self.h(self.x_hat_minus))
            self.P = (np.eye(self.n) - self.K @ self.H) @ self.P_minus

    # Return current a posteriori estimate
    def get_current_guess(self):
        return self.x_hat



