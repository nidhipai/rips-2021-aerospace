import numpy as np

class Track:
    def __init__(self, kfilter, filter_params, init_measure, init_time):
        initial_state = np.row_stack((init_measure, np.array([[0], [0]]))) #TO DO - how do we know initial velocity
        self.filter_model = kfilter(**filter_params, x_hat0=initial_state)
        self.measurements = {init_time: init_measure} #keys are timesteps
        self.predictions = dict() # keys are timesteps
        self.possible_observations = dict() # just for using to pass from gating to data association
        self.stage = 0 # 0 is not confirmed yet, 1 is confirmed, 2 is deletion

    def get_current_guess(self):
        return self.filter_model.get_current_guess()

    def get_measurement_cov(self):
        return self.filter_model.R

    def add_measurement(self, index, measurement):
        #self.measurements.append(measurement)
        self.measurements[index] = measurement

    def add_all_possible_observations(self, observations):
        for i, obs in enumerate(observations):
            self.add_possible_observation(i, obs)

    def add_possible_observation(self, index, observation):
        self.possible_observations[index] = observation
