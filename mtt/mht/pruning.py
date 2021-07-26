"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""
import numpy as np

class Pruning:
    def __init__(self, n):
        """


        """

        self.n = n # The number of time steps from which to prune back

    def predict(self, tracks, best_tracks):
        """
        The root node of each tree is changed to the node at T = n
        which feeds into the hypothesized track for that tree.

        Args:
            tracks (list): a list of all possible tracks at the current time step.
            best_tracks (list): list of indices of the best tracks selected at the current time step.
        """
        # Extract and store the sequences of measurements that correspond to valid tracks

        required_obs = []
        for index in best_tracks:
            prev_obs = np.array(list(tracks[index].observations.values))
            required_obs.append(prev_obs[:(prev_obs.size - 1 - self.n)])
        required_obs = np.array(required_obs)

        # Test each track to see whether its initial sequence leads to a valid part of the tree
        # OLD METHOD
        for track in tracks:
            keep = False
            # Extract the first part of the sequence of measurements, up to n
            prev_ob = np.array(list(track.observations.values))
            prev_ob = prev_ob[:(prev_ob.size - 1 - self.n)]

            # Test each possibility
            for required_ob in required_obs:
                if required_ob.size == prev_ob.size and prev_ob(required_ob == prev_ob).all():
                    keep = True

            # Remove the current track if its initial sequence of measurements does not match the current best hypothesis up to n
            if not keep:
                tracks.remove(track)








