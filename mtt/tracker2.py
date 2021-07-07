
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

        print(self.tracks)

        self.clear_possible_observations()  # reset this for the next round

    def set_kalman_params(self, f, jac, h, Q, W, R, H, u):
        self.kalman_params = {
            'f': f,
            'jac': jac,
            'h': h,
            'Q': Q,
            'W': W,
            'R': R,
            'H': H,
            'u': u
        }


    # current guess is the entire list of trajectories so far
    def get_trajectories(self):
        # get a dictionary of tracks
        current_guess = dict()
        for track in self.tracks:
            current_guess[len(current_guess)] = track.predictions
        return current_guess

    def clear_possible_observations(self):
        for track in self.tracks:
            track.possible_observations = []
