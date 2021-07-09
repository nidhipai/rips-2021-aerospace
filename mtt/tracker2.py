class MTTTracker:
    def __init__(self, methods):
        self.methods = methods
        self.tracks = dict()  # list of track objects

    def predict(self, measurements):
        print("tracks " + str(self.tracks))
        print("measurements " + str(measurements))
        # measurements is an array of 2D position vectors

        # first add measurements to all tracks, and then we'll narrow it down
        for key, track in self.tracks.items():
            track.add_all_possible_observations(measurements)

        for method in self.methods:
            method.predict(tracks=self.tracks, measurements=measurements)

        print(self.tracks)

        self.clear_possible_observations()  # reset this for the next round


    # current guess is the entire list of trajectories so far
    def get_trajectories(self):
        # get a dictionary of tracks
        current_guess = dict()
        for key, track in self.tracks.items():
            current_guess[len(current_guess)] = track.predictions
        return current_guess

    def clear_possible_observations(self):
        for key, track in self.tracks.items():
            track.possible_observations = dict()
