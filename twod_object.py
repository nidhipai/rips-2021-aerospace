import numpy as np
import math
from data_generator import DataGenerator


class TwoDObject(DataGenerator):
    def __init__(self, xt0, dt, ep_normal, ep_tangent, nu):
        """
        Constructor for the 2DObject Data Generator.
        :param xt0: Initial state vector
        :param dt: Length of one single time step
        :param ep_normal: Variance of the change in velocity vector in the normal direction.
        :param ep_tangent: Variance of the change in velocity vector in the tangent direction.
        :param nu: Variance of the measurement noise
        """
        self.dim = 2
        self.n = 4

        if(xt0.size != 4):
            raise Exception("Length of initial state vector does not equal 4")

        self.Q = np.diag(np.append(np.zeros(self.dim), np.append(np.array([ep_normal]), np.array(ep_tangent))))
        self.R = np.eye(self.dim) * nu

        super().__init__(xt0, dt, self.Q, self.R)

        self.H = np.append(np.eye(self.dim), np.zeros((self.dim, self.dim)), axis=1)
        self.A = np.append(np.append(np.eye(self.dim), np.eye(self.dim) * self.dt, axis=1),
                           np.append(np.zeros((self.dim, self.dim)), np.eye(self.dim), axis=1), axis=0)
        self.nu = nu

    def process_step(self, xt_prev):
        """
        Generate the next process state from the previous
        :param xt_prev: Previous process state
        :return: State vector of next step in the process
        """
        return self.A @ xt_prev + self.process_noise(xt_prev)

    def measure_step(self, xt):
        """
        Generate the next measure from the current process state vector
        :param xt: Current state vector
        :return: State vector representing measure at the current process state
        """
        return self.H @ xt + self.measure_noise()

    def measure_noise(self):
        """
        Generate measure noise
        """
        return np.random.normal(scale=self.nu, size=(self.dim, 1))

    #TODO: CURRENTLY HARD-CODED FOR 2D
    def process_noise(self, xt):
        """
        Generate process noise
        :param xt: current state vector
        :return: vector of noise for each parameter in the state vector
        """
        ang = math.atan2(xt[3, 0], xt[2, 0])
        c = math.cos(ang)
        s = math.sin(ang)
        rotation = np.array([[c, -s], [s, c]])
        rotated_cov = rotation @ self.Q[2:4, 2:4] @ rotation.T
        pad = np.array([0, 0])
        noise = np.random.multivariate_normal((0, 0), rotated_cov)
        output = np.append(pad, noise)
        output.shape = (4,1)
        return output

    def process_function(self, xt, u):
        return self.A @ xt

    def process_jacobian(self, xt, u):
        return self.A

    def measurement_function(self, xt):
        return self.process_function(xt, 0)[:self.dim]

    def measurement_jacobian(self, xt):
        return self.H



