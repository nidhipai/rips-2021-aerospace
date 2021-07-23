"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import numpy as np
from mtt import Track

"""
- start with the results of gating
- create new tracks, adding in the observations as necessary
"""

class MHTDataAssociation:
    def __init__(self):
        pass

    def predict(self, tracks=None, measurements=None, time=None, false_alarms=None, hypotheses=None):
        for hyp_key, hyp in hypotheses.items():
            possible_next_tracks = []
            for track_key, track in hyp.tracks.items():
                possible_next_tracks.append(track)
                for i, obs in enumerate(measurements):
                    for poss_obs in track.possible_observations.values():
                        if np.array_equiv(obs, poss_obs): # check if obs is in poss_obs
                            # TODO get kfilter and params from somewhere
                            new_track = Track(kfilter=kfilter, filter_params=filter_params, init_measure=obs, init_time=time)
                            tracks[len((tracks))] = new_track  # add track to the master list of tracks
                            possible_next_tracks.append(new_track)
