
"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
Simulation
"""
import numpy as np
from abc import ABC, abstractmethod


class DataGenerator(ABC):
    """Super class where we can generate data. We process and measure some generated randomized data using the covariance matrices for process
        and measurement noise. """
    def __init__(self, xt0, dt, Q, R):
        """
        Generate the process data over the specified number of time points.

        Args:
            xt0 (ndarray): Initial value state vector
            dt (numeric): Time step
            Q (ndarray): Process noise matrix
            R (ndarray): Measurement noise matrix
            rng (numpy.random.Generator): Random number generator object from numpy

        Returns:
            output (ndarray): A matrix with each column a state vector representing the process at each time step
        """
        self.xt0 = xt0
        self.dt = dt
        self.Q = Q
        self.R = R
        self.n = xt0[0].shape[0]

    def process(self, ts, rng):
        """
        Generate the process data over the specified number of time points

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
        Generates a dataset of measurements given underlying process data

        Args:
            process_result (ndarray): A matrix with each column a state vector at each time step.
            rng (numpy.random.Generator): Random number generator object from numpy

        Returns:
            output (ndarray): A list of state vector measurements at each time step
            colors (ndarray): A list of colors corresponding to each state vector

        """
        output = []
        colors = []
        for process in process_result:
            xt_measures, xt_colors = self.measure_step(process, rng)
            output.append(xt_measures)
            colors.append(xt_colors)
        return output, colors

    def process_measure(self, ts, rng):
        """
        First generates a process, and then generate a dataset of measurements given the underlying process data.

        Args:
            ts (int): the number of time steps.
            rng (numpy.random.Generator): Random number generator object from numpy

        Returns:
            output (ndarray): A matrix with each column a state vector measurement at each time step
        """
        output = self.process(ts, rng)
        output = self.measure(output, rng)
        return output

    def W(self, xt):
        return np.eye(self.n, self.n)

    @abstractmethod
    def process_step(self, xt_prev, rng):
        pass

    @abstractmethod
    def measure_step(self, xt_prev, rng):
        pass

    @abstractmethod
    def process_noise(self, xt, rng):
        pass

    @abstractmethod
    def measure_noise(self, rng):
        pass

    @abstractmethod
    def process_function(self, xt, u):
        pass

    @abstractmethod
    def process_jacobian(self, xt, u):
        pass

    @abstractmethod
    def measurement_function(self, xt):
        pass

    @abstractmethod
    def measurement_jacobian(self, xt):
        pass
