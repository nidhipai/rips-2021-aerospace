import numpy as np


class DataGenerator:
    def __init__(self, xt0, n, dt, ep, nu):
        """
        Create an object to store the different aspects of the system being generated
        :param xt0:  Initial values of the state
        :param n: Number of time points to simulate
        :param dt: Number of seconds between time points
        :param ep: Variance of process noise
        :param nu: Variance of measure noise
        """
        self.xt0 = xt0
        self.n = n
        self.dt = dt
        self.ep = ep
        self.nu = nu

    def process(self):
        """
        Generate the process data over the specified number of time points
        :return: A matrix with each column a state vector representing the process at each time step
        """
        output = self.xt0
        xt = self.xt0
        for i in range(self.n):
            xt = self.process_step(xt)
            output = np.append(output, xt, axis=1)
        return output

    def measure(self, proc_result):
        """
        Generate a dataset of measurements given an underlying process dataset
        :param proc_result: A matrix with each column a state vector at each time step
        :return: A matrix with each column a state vector measurement at each time step
        """
        output = np.array([[], []])
        for i in range(proc_result.shape[1]):
            proc = proc_result[:, i]
            proc.shape = (2, 1)
            xt_measure = self.measure_step(proc)
            output = np.append(output, xt_measure, axis=1)
        return output

    def process_measure(self):
        """
        First generate a process, and then generate a dataset of measurements given an underlying process dataset
        :return: A matrix with each column a state vector measurement at each time step
        """
        output = self.process()
        output = self.measure(output)
        return output

    def process_step(self, xt_prev):
        pass

    def measure_step(self, xt_prev):
        pass

    def process_noise(self):
        pass

    def measure_noise(self):
        pass
