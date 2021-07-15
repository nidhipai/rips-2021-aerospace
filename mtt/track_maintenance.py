"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
"""

from mtt import Track
import numpy as np

class TrackMaintenance:
    """
    Updates the status for tracks and creates new tracks as necessary
    """
    def __init__(self, kfilter, generator_params, num_init, num_init_frames, num_delete, predict_params=None, num_obj=None):
        """
        Sets the rules for how tracks are created and deleted
        Args:
            kfilter: constructor for the filter
            generator_params: dictionary of parameters for the filter, from the generator
            num_init: number of observations needed to confirm object in num_init_frames timesteps, <= num_init_frames
            num_init_frames: number of frames in which num_init observations will confirm an object
            num_delete: number of consecutive missing observations need to delete track
            predict_params: dict of params that should be used for the filter, overrides generator params, not necessary
            num_obj: number of objects (present at all times), if known, limits creation of new tracks
        """
        self.kfilter = kfilter

        # ex) if num_init = 3 and num_init_frames = 4, then 3 obs in 4 timesteps will confirm the object
        self.num_init = num_init
        self.num_init_frames = num_init_frames
        self.num_delete = num_delete

        self.num_obj = num_obj

        # pull the filter params from filter_params first, then if they aren't specificed, ask generator_params
        self.filter_params = generator_params.update(predict_params) if predict_params is not None else generator_params

    def predict(self, tracks=None, measurements=None, time=0, false_alarms=None):
        """
        Updates the status for tracks and creates new tracks as necessary
        Args:
            tracks: dictionary of tracks
            measurements: list of column vector measurements
            time: current timestep
            false_alarms: dict of false_alarms from tracker, where unmatched measurements go
        """
        # create new tracks for the measurements without a track - eventually we should check if they are false alarms
        false_alarms_time = []
        for measurement in measurements:
            new_objects_q = self.num_obj is None or len(tracks) < self.num_obj  # true if we can make more objects
            if measurement is not None:
                if new_objects_q:
                    tracks[len(tracks)] = Track(self.kfilter, self.filter_params, measurement, time)
                else:
                    false_alarms_time.append(measurement)
        false_alarms[time] = false_alarms_time

        # Confirm the track
        for track_key, track in tracks.items():
            # check what we need to confirm the target
            if track.stage == 0:
                last_init_frame_obs = list(track.measurements.values())[len(track.measurements) - self.num_init_frames:]
                appearances = sum(x is not None for x in last_init_frame_obs)
                if appearances >= self.num_init:
                    track.stage = 1

            # check what we need to in order to delete the object
            if (track.stage == 0 or track.stage == 1):
                if list(track.measurements.values())[-1] is None:
                    track.missed_measurements += 1
                    if track.missed_measurements == self.num_delete and self.num_obj is None:
                        # we don't actually kill the object/consider deaths if we know there are a fixed number of objects
                        track.stage = 2
                    # we don't currently bring objects back from the dead
                else:
                    track.missed_measurements = 0
