"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
"""

import numpy as np
from scipy.stats import chi2

class Distances:
    """
    Just calculates various distances

    Methods tagged with threshold means that it takes in a thresold and returns True if the dis is < threshold
    Measurement is a column vector and kfilter is a filter
    """
    @staticmethod
    def euclidean(measurement, kfilter):
        return np.linalg.norm(measurement - kfilter.get_current_guess())

    @staticmethod
    def mahalanobis(measurement, kfilter):
        innovation = measurement - kfilter.h(kfilter.x_hat_minus)
        K = kfilter.H @ kfilter.P_minus @ kfilter.H.T + kfilter.R
        dis = np.sqrt(innovation.T @ np.linalg.inv(K) @ innovation)
        dis = dis[0][0]  # this is kinda hacky and the fact that I have to do this may signal that something is wrong
        return dis

    @staticmethod
    def euclidean_threshold(measurement, kfilter, error_threshold):
        dis = Distances.euclidean(measurement, kfilter)
        return dis < error_threshold

    @staticmethod
    def mahalanobis_threshold(measurement, kfilter, error_threshold):
        dis = Distances.mahalanobis(measurement, kfilter)
        cutoff = chi2.ppf(error_threshold, 2)
        return dis < cutoff

    # consider adding a switcher to this class
