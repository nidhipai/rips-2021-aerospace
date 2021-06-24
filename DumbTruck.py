import numpy as np
import DataGenerator


class DumbTruck(DataGenerator.DataGenerator):
    def __init__(self, xt0, ts, dt, ep, nu, k):
        Q = np.eye(2) * ep
        R = np.eye(2) * nu
        super().__init__(xt0, ts, dt, Q, R)
        self.k = k
        self.ep = ep
        self.nu = nu

    def process_step(self, xt_prev):
        """
        Generate the next process state from the previous
        :param xt_prev: Previous process state
        :return: State vector of next step in the process
        """
        return xt_prev + self.dt*(np.matmul(np.array([[0, 1], [0, -self.k]]), xt_prev) + self.process_noise())

    def measure_step(self, xt):
        """
        Generate the next measure from the current process state vector
        :param xt: Current state vector
        :return: State vector representing measure at the current process state
        """
        return xt + self.measure_noise()

    def process_noise(self):
        """
        Generate process noise
        """
        return np.array([[0], [np.random.normal(scale=np.sqrt(self.ep))]])

    def measure_noise(self):
        """
        Generate measurement noise
        """
        return np.array([[np.random.normal(scale=self.nu)], [np.random.normal(scale=np.sqrt(self.nu))]])
