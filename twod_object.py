import numpy as np
from copy import copy
from data_generator import DataGenerator


class TwoDObject(DataGenerator):
    def __init__(self, xt0, dt, ep_tangent, ep_normal, nu, miss_p=0):
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

        self.ep_tangent = ep_tangent
        self.ep_normal = ep_normal
        self.nu = nu
        self.miss_p = miss_p

        if xt0[0].size != 4:
            raise Exception("Length of initial state vector does not equal 4")

        self.Q = np.diag(np.append(np.zeros(self.dim), np.append(np.array([ep_tangent]), np.array(ep_normal))))
        self.R = np.eye(self.dim) * nu

        super().__init__(xt0, dt, self.Q, self.R)

        self.H = np.append(np.eye(self.dim), np.zeros((self.dim, self.dim)), axis=1)
        self.A = np.append(np.append(np.eye(self.dim), np.eye(self.dim) * self.dt, axis=1),
                           np.append(np.zeros((self.dim, self.dim)), np.eye(self.dim), axis=1), axis=0)
        self.nu = nu

    def process_step(self, xt_prevs):
        """
        Generate the next process state from the previous
        :param xt_prev: Previous process state
        :return: State vector of next step in the process
        """
        output = []
        # Iterate through each state in the list of previous object states
        for xt_prev in xt_prevs:
            # calculate the next state and add to output
            xt_output = self.A @ xt_prev + self.process_noise(xt_prev)
            output.append(xt_output)
        return output

    def measure_step(self, xts):
        """
        Generate the next measure from the current process state vector
        :param xt: Current state vector
        :return: State vector representing measure at the current process state
        """

        # Iterate through each object state in the input
        output = []
        for xt in xts:
            #Calculate whether the measurement is missed
            if np.random.rand() > self.miss_p:
                output.append(self.H @ xt + self.measure_noise())
        return output

    def measure_noise(self):
        """
        Generate measure noise
        """
        return np.random.normal(scale=self.nu, size=(self.dim, 1))

    def process_noise(self, xt):

        """
        Generate process noise
        :param xt: current state vector
        :return: vector of noise for each parameter in the state vector
        """

        #NOTE: if the angle is 90 degrees then 0 is returned
        #Also this uses radians
        ang = np.arctan2(xt[3, 0], xt[2, 0])
        c = np.cos(ang)
        s = np.sin(ang)
        rotation = np.array([[c, -s], [s, c]])
        rotated_cov = rotation @ self.Q[2:4, 2:4] @ rotation.T
        pad = np.array([0, 0])
        noise = np.random.multivariate_normal((0, 0), rotated_cov)
        output = np.append(pad, noise)
        output.shape = (4, 1)
        return output

    def process_function(self, xt, u):
        return self.A @ xt

    def process_jacobian(self, xt, u):
        return self.A

    def measurement_function(self, xt):
        return self.H @ xt

    def measurement_jacobian(self, xt):
        return self.H

    def mutate(self, **kwargs):
        clone = copy(self)
        for arg in kwargs.items():
            setattr(clone, arg[0], arg[1])
        return TwoDObject(clone.xt0, clone.dt, clone.ep_tangent, clone.ep_normal, clone.nu, clone.miss_p)



