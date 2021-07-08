class MTTTracker:
    def __init__(self, methods):
        self.methods = methods
        self.tracks = dict()  # list of track objects

    def predict(self, measurements):
        # measurements is an array of 2D position vectors

        # first add measurements to all tracks, and then we'll narrow it down
        for key, track in self.tracks:
            track.add_all_possible_observations(measurements)

        # Apply each method to the set of tracks
        for method in self.methods:
            method.predict(tracks=self.tracks, measurements=measurements)

        print(self.tracks)

        self.clear_possible_observations()  # reset this for the next round


    # current guess is the entire list of trajectories so far
    def get_trajectories(self):
        # get a dictionary of tracks
        current_guess = dict()
        for key, track in self.tracks:
            current_guess[len(current_guess)] = track.predictions
        return current_guess

    def clear_possible_observations(self):
        for key, track in self.tracks:
            track.possible_observations = dict()
