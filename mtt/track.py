import numpy as np

class Track:
    def __init__(self, filter_model):
        self.measurements = []
        self.predictions = []
        self.kfilter = filter_model.__init__()
        self.possible_observations = dict() # just for using to pass from gating to data association
        self.stage = 0 # 0 is not confirmed yet, 1 is confirmed, 2 is on the way to deletion, 3 is deletion
        self.initiate_count = 0
        self.delete_count = 0

    def get_current_guess(self):
        return self.kfilter.get_current_guess()

    def get_measurement_cov(self):
        return self.kfilter.R

    def add_measurement(self, measurement):
        self.measurements.append(measurement)

    def add_all_possible_observations(self, observations):
        for i, obs in enumerate(observations):
            self.add_possible_observation(i, obs)

    def add_possible_observation(self, index, observation):
        self.possible_observations[index] = observation
