import numpy as np

class Tracker:
    def __init__(self, kFilter_model):
        self.kFilter_model = kFilter_model

        #Store the previous measures for future use
        self.measures = []
        self.current_guess = []

    def predict(self, measure_t):
        """

        :param measure_t: Either an empty list or a list of numpy arrays representing the measurements at the current time step
        :return: A list containing the predicted location of each object
        """

        if len(measure_t) > 0:
            # Process the point using the filter
            self.measures.append(measure_t)
            measure_t_new = self.remove_fas(measure_t)
            self.kFilter_model.predict(measure_t_new, np.array(self.measures))
            self.current_guess = [self.kFilter_model.get_current_guess()[0:2]]
        else:
            # If we don't have any measurements we need to guess for each object
            self.kFilter_model.predict(None, np.array(self.measures))
            self.current_guess = [self.kFilter_model.get_current_guess()[0:2]]

    def get_current_guess(self):
        return self.current_guess

    def remove_fas(self, measure_t):
        dists = []
        for point in measure_t:
            dists.append(self.mahalanobis_dist(point))
        return measure_t[np.argmin(dists)]


    def mahalanobis_dist(self, y):
        error = y - self.kFilter_model.h(self.kFilter_model.x_hat_minus)
        self.kFilter_model.error_array.append(error)
        # test = self.H@self.P_minus@self.H.T + self.R
        return np.sqrt(error.T @ np.linalg.inv(self.kFilter_model.R) @ error)
