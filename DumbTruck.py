import numpy as np
import DataGenerator


class DumbTruck(DataGenerator.DataGenerator):
    def __init__(self):
        super().__init__()

    def process_step(self, xt_prev, dt, k, ep):
        return xt_prev + np.matmul(np.array([[0, 1], [0, k]]), xt_prev) + self.process_noise(ep)

    def measure_step(self, xt, dt, k, ep, nu):
        return xt + self.measure_noise(nu)

    def process_noise(self, ep):
        return np.array([[0], [np.random.normal(scale=ep)]])

    def measure_noise(self, nu):
        return np.array([[0], [np.random.normal(scale=nu)]])
