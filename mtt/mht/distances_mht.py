"""
Sal Balkus, Nidhi Pai, Eduardo Sosa, Tony Zeng
RIPS 2021 Aerospace Team
"""

import numpy as np
from scipy.stats import chi2


class DistancesMHT:
    """
    Utility class for calculating various distances.

    Methods tagged with threshold means that it takes in a threshold and returns True if the dis is < threshold.
    Measurement is a column vector and kfilter is a filter throughout.
    """
    @staticmethod
    def euclidean(measurement, track):
        """
        Computes the euclidean distance between a measurement and the a posteriori estimate of a track.

        Args:
            measurement (ndarray): The measurement.
            track (Track): The track which contains the current a posteriori estimate.
        Returns:
            (float): the distance between the measurement and the a posteriori estimate. 
        """

        return np.linalg.norm(measurement - track.x_hat)

    @staticmethod
    def mahalanobis(measurement, track, kfilter):
        """
        Computes the Mahalanobis distance between a measurement and a track.

        Args:
            measurement (ndarray): The measurement. 
            track (Track): The track which contains the current a posteriori estimate.
            kfilter (KalmanFilter): The global kalman filter. 
        Returns:
            dis (float): The distance between the measurement and the a posteriori estimate.
        """

        innovation = (measurement - kfilter.h(track.x_hat_minus)).reshape((4,1))
        K = kfilter.H @ track.P @ kfilter.H.T + kfilter.R
        dis = np.sqrt(innovation.T @ np.linalg.inv(K) @ innovation)
        dis = dis[0][0]  # remove the numpy array wrapping
        return dis

    @staticmethod
    def euclidean_threshold(measurement, track, error_threshold, kfilter=None):
        """
        Returns whether the Euclidean distance between a measurement and a track is below a certain error
        threshold. 

        Args:
            measurement (ndarray): The measurement. 
            track (Track): The track which contains the current a posteriori estimate.
            error_threshold (float): The error threshold to be under to return True.
            kfilter (KalmanFilter, optional): The global Kalman filter used for all the tracks, unused for this method.
        Returns:
            True if the distance between the measurement and track prediction is below the threshold.
        """

        dis = DistancesMHT.euclidean(measurement, track)
        return dis < error_threshold

    @staticmethod
    def mahalanobis_threshold(measurement, track, error_threshold, kfilter=None):
        """
        Returns whether the Mahalanobis distance between a measurement and a track is below a certain error
        threshold. 

        Args:
            measurement (ndarray): The measurement. 
            track (Track): The track which contains the current a posteriori estimate.
            kfilter (KalmanFilter): The global Kalman filter used for all the tracks.
            error_threshold (float): The error threshold to be under to return True.
        Returns:
            True if the distance between the measurement and track prediction is below the threshold.
        """
        if kfilter is None:
            raise Exception("KalmanFilter is None in Distances")
        dis = DistancesMHT.mahalanobis(measurement, track, kfilter)
        cutoff = chi2.ppf(error_threshold, 2)
        return dis < cutoff
