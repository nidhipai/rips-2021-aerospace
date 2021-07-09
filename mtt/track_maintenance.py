from mtt import Track

class TrackMaintenance:
    def __init__(self, kfilter, filter_params, num_init, num_init_frames, num_delete):
        self.kfilter = kfilter
        self.filter_params = filter_params
        self.num_init = num_init
        self.num_init_frames = num_init_frames
        self.num_delete = num_delete

    def predict(self, tracks = None, measurements=None):
        for measurement in measurements:
            if measurement is not None:
                tracks[len(tracks)] = Track(self.kfilter, self.filter_params, measurement)
        print("tracks in maintain " + str(tracks))
        for track_key, track in tracks.items():
            if track.stage == 0: # check to confirm
                last_init_frame_obs = track.measurements[len(track.measurements) - self.num_init_frames:]
                appearances = sum(x is not None for x in last_init_frame_obs)
                if appearances >= self.num_init:
                    # confirm object
                    track.stage = 1
            elif track.measurements[-1] is None:
                last_delete_obs = track.measurements[len(track.measurements) - self.num_delete:]
                appearances = sum(x is not None for x in last_delete_obs)
                if appearances >= self.num_delete:
                    # delete track
                    track.stage = 3
            # we don't currently bring objects back from the dead
