"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
"""


class FilterPredict:
    """
    Wrapper around Kalman predict
    """
    def predict(self, tracks=None, measurements=None, time=0):
        """
        Runs the Kalman filters in each of the tracks
        Args:
            tracks: dictionary of tracks from MTTTracker
            measurements: not used
            time: the current timestep
        """
        for key, track in tracks.items():
            if track.stage == 0 or track.stage == 1:
                # measurement could be none but that's cool
                track.filter_model.predict(measurement=track.measurements[time])
                track.predictions[time] = track.get_current_guess()
