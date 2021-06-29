import numpy as np
from data_generator import DataGenerator


class TwoDObject(DataGenerator):
    def __init__(self, xt0, dt, ep_normal, ep_tangent, nu):
        """
        Constructor for the 2DObject Data Generator.
        :param xt0: Initial state vector
        :param dt: Length of one single time step
        :param ep_mag: Variance of the magnitude of the change in velocity
        :param ep_dir: Variance of the direction of the change in velocity. Specify as n-1 dimensional vector for n dimensional simulation.
        :param nu: Variance of the measurement noise
        """
        self.dim = 2
        self.n = 4

        if(xt0.length != 4):
            raise Exception("Length of initial state vector does not equal 4")

        self.Q = np.diag(np.append(np.zeros(self.dim), np.append(np.array([ep_normal]), np.array(ep_tangent))))
        self.R = np.eye(self.dim) * nu

        super().__init__(xt0, dt, self.Q, self.R)

        self.H = np.append(np.eye(self.dim), np.zeros((self.dim, self.dim)), axis=1)
        self.A = np.append(np.append(np.eye(self.dim), np.eye(self.dim) * self.dt, axis=1),
                           np.append(np.zeros((self.dim, self.dim)), np.eye(self.dim), axis=1), axis=0)
        self.nu = nu



