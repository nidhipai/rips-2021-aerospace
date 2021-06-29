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

    def process_step(self, xt_prev):
        """
        Generate the next process state from the previous
        :param xt_prev: Previous process state
        :return: State vector of next step in the process
        """
        return self.A @ xt_prev + self.process_noise()

    def measure_step(self, xt):
        """
        Generate the next measure from the current process state vector
        :param xt: Current state vector
        :return: State vector representing measure at the current process state
        """
        return self.H @ xt + self.measure_noise()

    def process_noise(self):
        rand = np.random.normal(np.zeros(self.n), np.append(np.zeros(self.dim), np.array(ep_normal, ep_tangent)))
        R = np.array([])
        return

    def measure_noise(self):
        return np.random.normal(scale=self.nu, size=(self.dim, 1))

    def process_function(self, xt, u):
        return self.A @ xt

    def process_jacobian(self, xt, u):
        return self.A

    def measurement_function(self, xt):
        return self.process_function(xt, 0)[:self.dim]

    def measurement_jacobian(self, xt):
        return self.H



