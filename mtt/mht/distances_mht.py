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
        Computes the mahalanobis distance between a measurement and a track. 

        Args:
            measurement (ndarray): The measurement. 
            track (Track): The track which contains the current a posteriori estimate.
            kfilter (KalmanFilter): The global kalman filter. 
        Returns:
            dis (float): the distance between the measurement and the a posteriori estimate. 
        """

        innovation = (measurement - kfilter.h(track.x_hat_minus)).reshape((4,1))
        K = kfilter.H @ track.P @ kfilter.H.T + kfilter.R
        dis = np.sqrt(innovation.T @ np.linalg.inv(K) @ innovation)
        dis = dis[0][0]  # this is kinda hacky and the fact that I have to do this may signal that something is wrong
        return dis


    @staticmethod
    def euclidean_threshold(measurement, track, kfilter, error_threshold):
        """
        returns whether the euclidean distance between a measurement and a track is below a certain error
        threshold. 

        Args:
            measurement (ndarray): The measurement. 
            track (Track): The track which contains the current a posteriori estimate.
            kfilter (KalmanFilter): The global kalman filter. 
            error_threshold (float): the error threshold 
        """

        dis = DistancesMHT.euclidean(measurement, track, kfilter)
        return dis < error_threshold

    @staticmethod
    def mahalanobis_threshold(measurement, track, kfilter, error_threshold):
        """
        returns whether the euclidean distance between a measurement and a track is below a certain error
        threshold. 

        Args:
            measurement (ndarray): The measurement. 
            track (Track): The track which contains the current a posteriori estimate.
            kfilter (KalmanFilter): The global kalman filter.
            error_threshold (float): the error threshold 
        """
        
        dis = DistancesMHT.mahalanobis(measurement, track, kfilter)
        cutoff = chi2.ppf(error_threshold, 2)
        return dis < cutoff

    # consider adding a switcher to this class
