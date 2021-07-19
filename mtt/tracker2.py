"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team

Todo:
    Eventually add an evaluation metric/class to the pipeline
"""
from itertools import repeat

class MTTTracker:
    """Pipeline of processes for multi-target tracking
    """
    def __init__(self, methods):
        """
        Keeps track of series of objects that manipulate incoming measurements to predict trajectories

        Args:
            methods: list of objects implementing predict that process the data
            MUST have data association, track maintenance, and filter predict
        """
        self.methods = methods
        self.tracks = dict()  # dictionary of all the tracks
        self.time = 0  # a counter for the timestep
        self.false_alarms = dict()  # each key is a timestep, the value is an array of false alarms for that timestep
        self.mtt = mtt

    def predict(self, measurements):
        """
        Runs all the elements of the pipeline once every iteration

        Args:
            measurements: array of 2D position vectors (column vectors) representing observations
        """

        # first add measurements to all tracks, and then we'll narrow it down
        for key, track in self.tracks.items():
            if track.stage != 2:
                track.add_all_possible_observations(measurements)

        # Apply each method to the set of tracks
        for method in self.methods:
            method.predict(tracks=self.tracks, measurements=measurements, time=self.time, false_alarms=self.false_alarms)

        self.time += 1
        self.clear_possible_observations()  # reset this for the next round

    def get_trajectories(self):
        """
        Gets the list of trajectories from all the tracks

        Returns: A list. Each index of the list represents a timestep; an index contains a dictionary. A key of the dictionary is the object keys (from MTTTracker.tracks) and a value is a column vector (2D vector)
        """
        result = []
        for ts in range(0, len(self.tracks[0].predictions.values())): # iterate over timesteps
            result.append(dict())
            for j, track in self.tracks.items():
                if ts in track.predictions.keys():
                    result[ts][j] = track.predictions[ts]
                else:
                    # Note that this assumes the state vector is of length 4
                    result[ts][j] = list(repeat([None], 4))
        return result

    def get_ellipses(self):
        """
        Returns: a dict with keys: tracks and values: array of ellipse params
        """
        ellipses = dict()
        for i, track in self.tracks.items():
            ellipses[i] = list(track.ellipses.values())
        return ellipses

    def get_sorted_measurements(self):
        """
        Returns: A dict with key: track and value: array of measurements for that track
        """
        measures = dict()
        for i, track in self.tracks.items():
            measures[i] = list(track.measurements.values())
        return measures

    def clear_possible_observations(self):
        """
        Resets the list of possible observations for the timestep (for all the tracks)
        """
        for key, track in self.tracks.items():
            track.possible_observations = dict()
