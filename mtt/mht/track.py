"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

class Track:
    def __init__(self, starting_observations, score):
        self.score = score
        self.x_hat = None
        self.P = None
        self.observations = starting_observations  # list/dict of (ts, k), where ts is the timestep and k is the number of the measurement
        # essentially this is the index in tracker.observations
        self.possible_observations = []  # lists possible observations for this timestep, indexes
        self.status = 0
