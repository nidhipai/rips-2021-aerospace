import numpy as np


class DataGenerator:
    def __init__(self):
        pass

    def process(self, xt_initial, t, dt, k, ep):
        output = xt_initial
        xt = xt_initial
        for i in range(t):
            xt = self.process_step(xt, dt, k, ep)
            output = np.append(output, xt, axis=1)
        return output

    def measure(self, xt_initial, t, dt, k, nu):
        output = xt_initial
        xt = xt_initial
        for i in range(t):
            xt = self.process_step(xt, dt, k, nu)
            output = np.append(output, xt, axis=1)
        return output

    def process_step(self, xt_prev, dt, k, ep):
        pass

    def measure_step(self, xt_prev, dt, k, ep, nu):
        pass

    def process_noise(self, ep):
        pass

    def measure_noise(self, nu):
        pass
