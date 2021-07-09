import numpy as np

class Track:
    def __init__(self, kfilter, filter_params, init_measure):
        self.filter_model = kfilter(**filter_params, x_hat0=init_measure)
        self.measurements = [init_measure]
        self.predictions = []
        self.possible_observations = dict() # just for using to pass from gating to data association
        self.stage = 0 # 0 is not confirmed yet, 1 is confirmed, 2 is deletion

    def get_current_guess(self):
        return self.filter_model.get_current_guess()

    def get_measurement_cov(self):
        return self.filter_model.R

    def add_measurement(self, measurement):
        self.measurements.append(measurement)

    def add_all_possible_observations(self, observations):
        for i, obs in enumerate(observations):
            self.add_possible_observation(i, obs)

    def add_possible_observation(self, index, observation):
        self.possible_observations[index] = observation
