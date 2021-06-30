import numpy as np


class DataGenerator:
    def __init__(self, xt0, dt, Q, R):
        """
        Create an object to store the different aspects of the system being generated
        :param xt0:  Initial values of the state
        :param dt: Number of seconds between time points
        :param Q: Covariance matrix of process noise
        :param R: Covariance matrix of measure noise
        """
        self.xt0 = xt0
        self.dt = dt
        self.Q = Q
        self.R = R
        self.n = xt0.shape[0]

    def process(self, ts):
        """
        Generate the process data over the specified number of time points
        :return: A matrix with each column a state vector representing the process at each time step
        """
        output = self.xt0
        xt = self.xt0
        for i in range(ts):
            xt = self.process_step(xt)
            output = np.append(output, xt, axis=1)
        return output

    def measure(self, proc_result):
        """
        Generate a dataset of measurements given an underlying process dataset
        :param proc_result: A matrix with each column a state vector at each time step
        :return: A matrix with each column a state vector measurement at each time step
        """
        output = np.empty((self.n//2, 1))
        for i in range(proc_result.shape[1]):
            proc = proc_result[:, i]
            proc.shape = (self.n, 1)
            xt_measure = self.measure_step(proc)
            output = np.append(output, xt_measure, axis=1)
        return output[:, 1:]

    def process_measure(self, ts):
        """
        First generate a process, and then generate a dataset of measurements given an underlying process dataset
        :return: A matrix with each column a state vector measurement at each time step
        """
        output = self.process(ts)
        output = self.measure(output)
        return output

    def process_step(self, xt_prev):
        pass

    def measure_step(self, xt_prev):
        pass

    def process_noise(self, xt):
        pass

    def measure_noise(self):
        pass

    def process_function(self, xt, u):
        pass

    def process_jacobian(self, xt, u):
        pass

    def measurement_function(self, xt):
        pass

    def measurement_jacobian(self, xt):
        pass
