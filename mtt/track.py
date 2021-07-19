"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
"""

import numpy as np


class Track:
    """ Track object encapsulating the kalman filter, associated measurements, and predictions
    """
    def __init__(self, kfilter, filter_params, init_measure, init_time):
        """
        Creates a Track object

        Args:
            kfilter: a constructor for a Kalman filter (not an already initalized filter)
            filter_params: a dictionary of parameters for the filter
            init_measure: the initial measurement that creates the track
            init_time: what timestep init_measure was recorded
            init_velocity: initial velocity of object, np array
        """
        # initial state for filter, uses intial measure and 0 velocity
        self.filter_model = kfilter(**filter_params, xt0=init_measure)  # instantiate a filter,
        # assuming first measure is the correct starting location
        self.measurements = {init_time: init_measure}  # keys are timesteps, value may be none
        self.predictions = dict() # keys are timesteps, doesn't need to start at 0
        self.possible_observations = dict()  # used to pass around the possible measurements for this track for this ts
        self.apriori_ellipses = dict() # keys are timesteps, values are x_hat tuples - arguments for cov_ellipse in sim
        self.aposteriori_ellipses = dict() # keys are timesteps, values are x_hat tuples - arguments for cov_ellipse in sim
        self.stage = 0  # 0 is not confirmed yet, 1 is confirmed, 2 is deleted (done in track maintenance)
        self.missed_measurements = 0  # used to expand the gate and delete object if too many measurements are missed

    def get_current_guess(self):
        """
        Returns: the current prediction of the state from the Kalman filter
        """
        return self.filter_model.get_current_guess()

    def get_measurement_cov(self):
        """
        Returns: Returns the covariance matrix from the Kalman filter
        """
        return self.filter_model.R

    def add_measurement(self, index, measurement):
        """
        Add a measurement that is predicted to be associated with this track
        Args:
            index: the timestep for the measurement
            measurement: the observation
        """
        self.measurements[index] = measurement

    def add_all_possible_observations(self, observations):
        """
        Wrapper function for add_possible_observations
        Args:
            observations: an array of column vectors
        """
        for i, obs in enumerate(observations):
            self.add_possible_observation(i, obs)

    def add_possible_observation(self, index, observation):
        """
        Add a measurement as being possible associated with the track, clears every timestep
        Args:
            index: timestep for the measurement
            observation: the observation, a column vector
        """
        self.possible_observations[index] = observation
