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

    def measure(self, process_result, nu):
        output = np.array([[], []])
        for i in range(process_result.shape[1]):
            proc = process_result[:,i]
            proc.shape = (2,1)
            xt_measure = self.measure_step(proc, nu)
            output = np.append(output, xt_measure, axis=1)
        return output

    def process_measure(self, xt_initial, t, dt, k, ep, nu):
        output = self.process(xt_initial, t, dt, k, ep)
        output = self.measure(output, nu)
        return output

    def process_step(self, xt_prev, dt, k, ep):
        pass

    def measure_step(self, xt_prev, nu):
        pass

    def process_noise(self, ep):
        pass

    def measure_noise(self, nu):
        pass
