import numpy as np
import DataGenerator


class SmartTruck(DataGenerator.DataGenerator):
    def __init__(self, xt0, ts, dt, ep, nu):
        super().__init__(xt0, ts, dt, ep, nu)
        self.H = np.append(np.eye(self.n//2), np.zeros((self.n//2,self.n//2)), axis = 1)
        self.A = np.append(np.append(np.eye(self.n//2), np.eye(self.n//2)*self.dt, axis=1), np.append(np.zeros((self.n//2,self.n//2)), np.eye(self.n//2), axis=1), axis=0)


    def process_step(self, xt_prev):
        """
        Generate the next process state from the previous
        :param xt_prev: Previous process state
        :return: State vector of next step in the process
        """
        return np.matmul(self.A,xt_prev) + self.process_noise()

    def measure_step(self, xt):
        """
        Generate the next measure from the current process state vector
        :param xt: Current state vector
        :return: State vector representing measure at the current process state
        """
        return np.matmul(self.H, xt) + self.measure_noise()

    def process_noise(self):
        """
        Generate process noise
        """
        return np.append(np.zeros((self.n // 2, 1)), np.random.normal(scale=self.ep, size=(self.n // 2, 1)), axis=0)

    def measure_noise(self):
        """
        Generate measurement noise
        """
        return np.random.normal(scale=self.nu, size=(self.n // 2, 1))