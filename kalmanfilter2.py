"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
Kalman Filter - Discrete
"""

import numpy as np
import numpy.linalg as linalg


class KalmanFilter:
    def __init__(self, x_hat0, A, Q, R, H=None, B=0, u=0, method = "standard"):
        self.A = A  # state-transition matrix
        self.Q = Q  # process noise covariance
        self.R = R  # measurement noise covariance

        # calculate dimension of x
        self.n = x_hat0.shape[0]

        #Set default H if it is not defined
        if H is None:
            self.H = np.eye(self.n, self.n)
        else:
            self.H = H

        if B is not None:
            self.B = B
            self.u = u

        # set a priori and a posteriori estimate error covariances to all ones (not all zeros)
        self.P = np.ones((self.n, self.n))
        self.P_minus = np.ones((self.n, self.n))
        self.x_hat = x_hat0  # set a priori estimate to initial guess
        self.x_hat_minus = x_hat0  # set a posteriori estimate to initial guess

        self.method = method

    #Update a posteriori estimate based on a priori estimate and measurement
    def predict(self, measurement, f = None, h = None):
        #The standard Kalman Filter
        if self.method == "standard":
            self.x_hat_minus = self.A @self.x_hat
            self.P_minus = self.A @self.P@self.A.T+ self.Q
            self.K = self.P_minus @ self.H.T @ linalg.inv(self.H @ self.P_minus @ self.H.T + self.R)
            self.x_hat = self.x_hat_minus + self.K@(measurement - (self.H@self.x_hat_minus))
            self.P = (np.eye(self.n,self.n) - self.K @ self.H) @ self.P_minus

        #The extended Kalman Filter
        elif self.method == "extended":
            self.x_hat_minus = f(self.x_hat)
            self.P_minus = self.A @self.P@self.A.T+ self.Q
            self.K = self.P_minus @ self.H.T @ linalg.inv(self.H @ self.P_minus @ self.H.T + self.R)
            self.x_hat = self.x_hat_minus + self.K@(measurement - h(x_hat_minus))
            self.P = (np.eye(self.n,self.n) - self.K @ self.H) @ self.P_minus

    #Return current a posteriori estimate
    def get_current_guess(self):
        return self.x_hat


