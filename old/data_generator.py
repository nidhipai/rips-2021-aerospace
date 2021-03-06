
"""
Aerospace Team - Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
"""
import numpy as np

class DataGenerator:
        """Super class where we can generate data. We process and measure some generated randomized data using the covariance matrices for process
    and measurement noise. """
    def __init__(self, xt0, dt, Q, R):
        """
        Create an object to store the different aspects of the system being generated
        :param xt0: a list of initial values of the state for each point
        :param dt: Number of seconds between time points
        :param Q: Covariance matrix of process noise
        :param R: Covariance matrix of measure noise
        """
        self.xt0 = xt0
        self.dt = dt
        self.Q = Q
        self.R = R
        self.n = xt0[0].shape[0]

    def process(self, ts, rng):
        """
        Generate the process data over the specified number of time points.

        Args:
            ts (int): the number of time steps.
            rng: --

        Returns:
            output (ndarray): A matrix with each column a state vector representing the process at each time step
        """

        # Store the list of initial values to start
        xts = self.xt0
        output = [xts]

        # Run the simulation for the specified time steps
        for i in range(ts):
            # Produce a new state for each object and store
            xts = self.process_step(xts, rng)
            output.append(xts)
        return output

    def measure(self, process_result, rng):
        """
        Generate a dataset of measurements given an underlying process dataset

        Args:
            proc_result (ndarray): A matrix with each column a state vector at each time step.
            rng: --

        Returns:
            output (ndarray): A matrix with each column a state vector measurement at each time step
        """
        output = []
        for process in process_result:
            xt_measures = self.measure_step(process, rng)
            output.append(xt_measures)
        return output

    def process_measure(self, ts, rng):
        """
        First generate a process, and then generate a dataset of measurements given an underlying process dataset.

        Args:
            ts (int): the number of time steps.
            rng: --

        Returns:
            output (ndarray): A matrix with each column a state vector measurement at each time step
        """
        output = self.process(ts, rng)
        output = self.measure(output, rng)
        return output

    def process_step(self, xt_prev, rng):
        pass

    def measure_step(self, xt_prev, rng):
        pass

    def process_noise(self, xt, rng):
        pass

    def measure_noise(self, rng):
        pass

    def W(self,xt):
        return np.eye(self.n, self.n)

    def process_function(self, xt, u):
        pass

    def process_jacobian(self, xt, u):
        pass

    def measurement_function(self, xt):
        pass

    def measurement_jacobian(self, xt):
        pass


