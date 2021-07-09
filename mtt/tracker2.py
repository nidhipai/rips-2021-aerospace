class MTTTracker:
    def __init__(self, methods):
        self.methods = methods
        self.tracks = dict()  # list of track objects
        self.time = 0

    def predict(self, measurements):
        #print("tracks " + str(self.tracks))
        #print("measurements " + str(measurements))
        # measurements is an array of 2D position vectors

        # first add measurements to all tracks, and then we'll narrow it down
        for key, track in self.tracks.items():
            track.add_all_possible_observations(measurements)

        # Apply each method to the set of tracks
        for method in self.methods:
            method.predict(tracks=self.tracks, measurements=measurements, time=self.time)

        #print(self.tracks)
        self.time += 1

        self.clear_possible_observations()  # reset this for the next round


    # current guess is the entire list of trajectories so far
    def get_trajectories(self):
        # get a dictionary of tracks
        result = []
        for ts in range(0, len(self.tracks[0].predictions.values())): # iterate over timesteps
            result.append(dict())
            for j, track in self.tracks.items():
                if ts in track.predictions.keys():
                    result[ts][j] = track.predictions[ts]
                else:
                    result[ts][j] = [[None], [None]]
                #print("measurements" + str(self.tracks[0].measurements))
        #print("result: " + str(result))

        return result

    def clear_possible_observations(self):
        for key, track in self.tracks.items():
            track.possible_observations = dict()
