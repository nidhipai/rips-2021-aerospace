import numpy as np


class DataGenerator:
    def __init__(self):
        pass

    def process(self, xt_initial, t, dt, k, ep):
        """
        Generate the process data over the specified number of time points

        :param xt_initial: Initial values of the state
        :param t: Number of time points to simulate
        :param dt: Number of seconds between time points
        :param k: Coefficient of friction
        :param ep: Variance of process noise
        :return: a matrix with each column a state vector at the nth point in time
        """
        output = xt_initial
        xt = xt_initial
        for i in range(t):
            xt = self.process_step(xt, dt, k, ep)
            output = np.append(output, xt, axis=1)
        return output

    def measure(self, process_result, nu):
        """
        Generate a dataset of measurements given an underlying process dataset

        :param process_result: A matrix with each column a state vector at each time step
        :param nu: Variance of the measurement error
        :return: A matrix with each column a state vector measurement at each time step
        """
        output = np.array([[], []])
        for i in range(process_result.shape[1]):
            proc = process_result[:,i]
            proc.shape = (2,1)
            xt_measure = self.measure_step(proc, nu)
            output = np.append(output, xt_measure, axis=1)
        return output

    def process_measure(self, xt_initial, t, dt, k, ep, nu):
        """
        First generate a process, and then generate a dataset of measurements given an underlying process dataset

        :param xt_initial: Initial values of the state
        :param t: Number of time points to simulate
        :param dt: Number of seconds between time points
        :param k: Coefficient of friction
        :param ep: Variance of process noise
        :param nu: Variance of the measurement error
        :return: A matrix with each column a state vector measurement at each time step
        """
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
