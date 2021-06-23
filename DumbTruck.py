import numpy as np
import DataGenerator


class DumbTruck(DataGenerator.DataGenerator):
    def __init__(self):
        super().__init__()

    def process_step(self, xt_prev, dt, k, ep):
        """
        Generate the next process state from the previous
        :param xt_prev: Previous process state
        :param dt: Time step
        :param k: Coefficient of friction
        :param ep: Variance of process error
        :return: State vector of next step in the process
        """
        return xt_prev + dt*(np.matmul(np.array([[0, 1], [0, -k]]), xt_prev) + self.process_noise(ep))

    def measure_step(self, xt, nu):
        """
        Generate the next measure from the current process state vector
        :param xt: Current state vector
        :param nu: Measurement variance
        :return: State vector representing measure at the current process state
        """
        return xt + self.measure_noise(nu)

    def process_noise(self, ep):
        """
        Generate process noise
        """
        return np.array([[0], [np.random.normal(scale=np.sqrt(ep))]])

    def measure_noise(self, nu):
        """
        Generate measurement noise
        """
        return np.array([[np.random.normal(scale=nu)], [np.random.normal(scale=np.sqrt(nu))]])
