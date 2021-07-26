"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

class Track:
    def __init__(self, starting_observations, score, object_id):
        self.score = score
        self.x_hat = []
        self.P = []
        self.observations = starting_observations  # list of (ts, k), where ts is the timestep and k is the number of the measurement
        # essentially this is the index in tracker.observations
        self.possible_observations = []  # lists possible observations for this timestep, indexes
        self.status = 0
        self.object_id = object_id
