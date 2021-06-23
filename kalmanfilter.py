#Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
#Aerospace Team
#Kalman Filter - Discrete

import numpy as np
import numpy.linalg as linalg

#Kalman Filter class
class KalmanFilter:

    def __init__(self, x_hat0, delta, drag, epsilon, d1, d2):
        self.delta = delta      #forward euler time-step
        self.drag = drag        #coefficient of drag
        self.epsilon = epsilon  #estimated process noise (variance)
        self.d1 = d1            #measurement noise position (variance)
        self.d2 = d2            #measurement noise velocity (variance)

        #state-transition matrix
        self.A = np.eye(2, 2)+ np.array(([0, self.delta],[0, -1*self.drag * self.delta]))

        self.H = np.eye(2, 2)                           #observation model (identity for now)
        self.Q = np.array(([0, 0], [0, self.epsilon]))  #process noise covariance
        self.R = np.array(([self.d1, 0], [0, self.d2])) #measurement noise covariance

        self.x_hat = x_hat0         #set a priori estimate to initial guess
        self.x_hat_minus = x_hat0   #set a posteriori estimate to initial guess

        #set a priori and a posteriori estimate error covariances to all ones (not all zeros)
        self.P, self.P_minus = np.ones((2,2))

    #Update a posteriori estimate based on a priori estimate and measurement
    def predict(self, measurement):
        self.x_hat_minus = np.dot(self.A,self.x_hat)
        self.P_minus = np.dot(np.dot(self.A,self.P),self.A.T) + self.Q
        self.K = np.dot(self.P_minus,linalg.inv(self.P_minus + self.R))
        self.x_hat = self.x_hat_minus + np.dot(self.K,(measurement - self.x_hat_minus))
        self.P = np.dot((np.eye(2,2)- self.K),self.P_minus)

    #Return current a posteriori estimate
    def get_current_guess(self):
        return self.x_hat

