"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
"""

import numpy as np
from scipy.stats import chi2

class DistancesMHT:
    """
    Calculates various distances

    Methods tagged with threshold means that it takes in a thresold and returns True if the dis is < threshold
    Measurement is a column vector and kfilter is a filter
    """
    @staticmethod
    def euclidean(measurement, track):
        return np.linalg.norm(measurement - track.x_hat)

    @staticmethod
    def mahalanobis(measurement, track, kfilter):
        innovation = measurement - kfilter.h(track.x_hat_minus).squeeze()
        K = kfilter.H @ track.P_minus @ kfilter.H.T + kfilter.R
        dis = np.sqrt(innovation.T @ np.linalg.inv(K) @ innovation)
        print(dis)
        # dis = dis[0][0]  # this is kinda hacky and the fact that I have to do this may signal that something is wrong
        return dis

    @staticmethod
    def euclidean_threshold(measurement, track, kfilter, error_threshold):
        dis = DistancesMHT.euclidean(measurement, track, kfilter)
        return dis < error_threshold

    @staticmethod
    def mahalanobis_threshold(measurement, track, kfilter, error_threshold):
        dis = DistancesMHT.mahalanobis(measurement, track, kfilter)
        cutoff = chi2.ppf(error_threshold, 2)
        return dis < cutoff

    # consider adding a switcher to this class
