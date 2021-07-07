
class Tracker:
    def __init__(self, methods):
        self.methods = methods
        self.tracks = [] # list of track objects

    def predict(self, measurements):
        # measurements is an array of 2D position vectors

        # first add measurements to all tracks, and then we'll narrow it down
        for track in self.tracks:
            track.add_possible_observations(measurements)

        for method in self.methods:
            method.predict(tracks=self.tracks)

        self.clear_possible_observations()  # reset this for the next round

    def clear_possible_observations(self):
        for track in self.tracks:
            track.possible_observations = []
