"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
"""

from mtt import Track

class TrackMaintenance:
    """
    Updates the status for tracks and creates new tracks as necessary
    """
    def __init__(self, kfilter, generator_params, num_init, num_init_frames, num_delete, predict_params=None):
        """
        Sets the rules for how tracks are created and deleted
        Args:
            kfilter: constructor for the filter
            generator_params: dictionary of parameters for the filter, from the generator
            num_init: number of observations needed to confirm object in num_init_frames timesteps, <= num_init_frames
            num_init_frames: number of frames in which num_init observations will confirm an object
            num_delete: number of consecutive missing observations need to delete track
            predict_params: dict of params that should be used for the filter, overrides generator params
        """
        self.kfilter = kfilter

        # ex) if num_init = 3 and num_init_frames = 4, then 3 obs in 4 timesteps will confirm the object
        self.num_init = num_init
        self.num_init_frames = num_init_frames
        self.num_delete = num_delete

        # pull the filter params from filter_params first, then if they aren't specificed, ask generator_params
        self.filter_params = generator_params.update(predict_params) if predict_params is not None else generator_params

    def predict(self, tracks = None, measurements=None, time=0):
        """
        Updates the status for tracks and creates new tracks as necessary
        Args:
            tracks: dictionary of tracks
            measurements: list of column vector measurements
            time: current timestep
        """
        # create new tracks for the measurements without a track - eventually we should check if they are false alarms
        for measurement in measurements:
            if measurement is not None:
                tracks[len(tracks)] = Track(self.kfilter, self.filter_params, measurement, time)

        for track_key, track in tracks.items():
            # check what we need to confirm the target
            if track.stage == 0:
                last_init_frame_obs = list(track.measurements.values())[len(track.measurements) - self.num_init_frames:]
                appearances = sum(x is not None for x in last_init_frame_obs)
                if appearances >= self.num_init:
                    track.stage = 1
            # check what we need to in order to delete the object

            if (track.stage == 0 or track.stage == 1) and list(track.measurements.values())[-1] is None:
                last_delete_obs = list(track.measurements.values())[len(track.measurements) - self.num_delete:]
                missing_observations = sum(x is None for x in last_delete_obs)
                if missing_observations == self.num_delete:
                    track.stage = 2
            # we don't currently bring objects back from the dead
