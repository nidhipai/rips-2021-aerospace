import numpy as np

class Track:
    def __init__(self):
        self.measurements = [] # List of measurements associated with this track
        self.predictions = [] # One prediction per time step
        self.possible_observations = dict() # just for using to pass from gating to data association, resets every time step
        self.stage = 0 # labels stage of the object; 0 is not confirmed yet, 1 is confirmed, 2 is deletion
        #self.initiate_count = 1
        #self.delete_count = 0

    def set_filter(self, filter_model, filter_params):
        # THIS IS DEF NOT RIGHT WAY OF PASSING THE PARAMS AS A DICTIONARY
        self.kfilter = filter_model.__init__(filter_params)

    def get_current_guess(self):
        """"
        Returns current guess of the Kalman filter
        """"
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
