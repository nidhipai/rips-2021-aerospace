#Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
#Aerospace Team
#Kalman Filter - Discrete

import numpy as np
import numpy.linalg as linalg

class KalmanFilter:
    def __init__(self, x_hat0, delta, drag, epsilon, d1, d2):

        self.delta = delta
        self.drag = drag
        self.epsilon = epsilon
        self.d1 = d1
        self.d2 = d2

        self.A = np.eye(2, 2)+ np.array(([0, self.delta],[0, -1*self.drag * self.delta]))
        self.H = np.eye(2, 2)
        self.Q = np.array(([0, 0], [0, self.epsilon]))
        self.R = np.array(([self.d1, 0], [0, self.d2]))

        self.x_hat = x_hat0
        self.x_hat_minus = x_hat0

        self.P, self.P_minus = np.ones((2,2))

    def predict(self, measurement):
        self.x_hat_minus = np.dot(self.A,self.x_hat)
        self.P_minus = np.dot(np.dot(self.A,self.P),self.A.T) + self.Q
        self.K = np.dot(self.P_minus,linalg.inv(self.P_minus + self.R))
        self.x_hat = self.x_hat_minus + np.dot(self.K,(measurement - self.x_hat_minus))
        self.P = np.dot((np.eye(2,2)- self.K),self.P_minus)

    def get_current_guess(self):
        return self.x_hat




