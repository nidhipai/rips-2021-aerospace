import numpy as np

class Track:
    def __init__(self, filter_model):
        self.measurements = []
        self.predictions = []
        self.kfilter = filter_model.__init__()
        self.possible_observations = [] # just for using to pass from gating to data association

    def get_latest_prediction(self):
        return self.kfilter.get_current_guess()

    def add_measurement(self, measurement):
        self.measurements.append(measurement)

    def add_possible_observations(self, observations):
        self.possible_observations.append(observations)
