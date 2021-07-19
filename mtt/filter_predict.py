"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
"""


class FilterPredict:
    """
    Wrapper around Kalman predict
    """
    def predict(self, tracks=None, measurements=None, time=0, false_alarms=None, hypotheses = None):
        """
        Runs the Kalman filters in each of the tracks
        Args:
            tracks: dictionary of tracks from MTTTracker
            measurements: not used
            time: the current timestep
            false_alarms: not used
        """
        for key, track in tracks.items():
            if track.stage == 0 or track.stage == 1:
                # measurement could be none but that's cool
                track.filter_model.predict(measurement=track.measurements[time])
                track.predictions[time] = track.get_current_guess()

                mean = (track.filter_model.x_hat[0, 0], track.filter_model.x_hat[1, 0])
                cov = track.filter_model.P[:2, :2]
                track.ellipses[time] = [mean, cov]  # these are the params needed to make the ellipses
            # don't need to take care of dead objects here beause it's  taken care of in get_traj in tracker2