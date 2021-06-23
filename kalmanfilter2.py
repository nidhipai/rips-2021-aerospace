#Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
#Aerospace Team
#Kalman Filter - Discrete

import numpy as np
import numpy.linalg as linalg

#Kalman Filter class
class KalmanFilter:

    def __init__(self, x_hat0, A, R, Q, H = None, B = None, u = None):
        self.n = x_hat0.shape[0] #dimension of x

        #state-transition matrix
        self.A = A

        if H is None:
            self.H = np.eye(n, n)
        else:
            self.H = H                           #observation model (identity for now)
        self.Q = Q #process noise covariance
        self.R = R #measurement noise covariance

        self.x_hat = x_hat0         #set a priori estimate to initial guess
        self.x_hat_minus = x_hat0   #set a posteriori estimate to initial guess

        #set a priori and a posteriori estimate error covariances to all ones (not all zeros)
        self.P, self.P_minus = np.ones((n,n))

        if B is not None:
            self.B = B
            self.u = u

    #Update a posteriori estimate based on a priori estimate and measurement
    def predict(self, measurement):
        self.x_hat_minus = np.dot(self.A,self.x_hat)
        self.P_minus = np.dot(np.dot(self.A,self.P),self.A.T) + self.Q
        self.K = np.dot(self.P_minus,linalg.inv(self.P_minus + self.R))
        self.x_hat = self.x_hat_minus + np.dot(self.K,(measurement - self.x_hat_minus))
        self.P = np.dot((np.eye(self.n,self.n)- self.K),self.P_minus)

    #Return current a posteriori estimate
    def get_current_guess(self):
        return self.x_hat


