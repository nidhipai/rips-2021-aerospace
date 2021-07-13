import numpy as np
from scipy.stats import chi2

class Tracker:
    def __init__(self, kFilter_model):
        """
        Constructor for a Tracker object where we pass in a Kalman Filter object and initiate a measurement list instance
        and a estimated state vector list instance.

        Args:
            kFilter_model (KalmanFilter): A kalman filter to be used throughout.
        """

        self.kFilter_model = kFilter_model

        #Store the previous measures for future use
        self.measures = []
        self.current_guess = []
        # self.mhlb_dists = []

    def predict(self, measure_t):
        """
        Approximate trajectory/state vector of our simulated object.

        Args:
            measure_t (list) : Either an empty list or a list of numpy arrays representing the measurements at the current time step.

        Returns:
            list: A list containing the predicted location of each object
        """

        if len(measure_t) > 0:
            # Process the point using the filter
            measure_t_new = self.remove_fas(measure_t)
            self.measures.append(measure_t_new)
            self.kFilter_model.predict(measure_t_new, np.array(self.measures))
            self.current_guess = {0: self.kFilter_model.get_current_guess()[0:2]}
        else:
            # If we don't have any measurements we need to guess for each object
            self.kFilter_model.predict(None, np.array(self.measures))
            self.current_guess = {0: self.kFilter_model.get_current_guess()[0:2]}

    def get_current_guess(self):
        """
        Returns:
            ndarray: current state vector
        """
        return self.current_guess

    def remove_fas(self, measure_t):
        self.dists = []
        for point in measure_t:
            self.dists.append(self.mahalanobis_dist(point))
        measure = measure_t[np.argmin(self.dists)]
        # self.mhlb_dists.append(np.min(self.dists))
        return measure

    # def get_dists(self):
    #     return self.mhlb_dists


    def mahalanobis_dist(self, y):
        """
        Computes the mahalanobis distance between the measurement error to the predicted error to test
        measurement is an outlier.

        Args:
            y (ndarray): the current measurement state vector

        Returns:
            float: the calculated mahalanobis distance.
        """

        error = y - self.kFilter_model.h(self.kFilter_model.x_hat_minus)
        # error = error.reshape((2,1))
        self.kFilter_model.error_array.append(error)
        K = self.kFilter_model.H@self.kFilter_model.P_minus@self.kFilter_model.H.T + self.kFilter_model.R
        mhlb_dist = np.sqrt(error.T @ np.linalg.inv(K) @ error)
        # if mhlb_dist > chi2.ppf(0.95,2):
        return mhlb_dist
