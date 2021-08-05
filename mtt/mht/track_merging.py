"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import numpy as np

class TrackMerging:
    def __init__(self, rmse_threshold=0):
        self.rmse_threshold = rmse_threshold

    def predict(self, ts, tracks):
        # case 1 and case 2
        print("skipping track merging")
        #self.track_merge_rmse(ts, tracks)
        # case 3 - consecutive trajectories

    @staticmethod
    def rmse(track1, track2):
        """
        Assumes they are the same length and the time steps are synched.
        """
        error = np.sqrt(np.square(track1 - track2).sum(axis=1))
        return np.sqrt(1 / len(track1) * np.sum(error, axis=1))

    def track_merge_rmse(self, ts, tracks):
        # track merging - case 1 & 2 (non consecutive)
        # synchronize time steps
        indexes_to_remove = set()
        for i in range(len(tracks)):
            for j in range(i):
                if i == j or not tracks[i].confirmed() or not tracks[j].confirmed():
                    continue
                pred1 = tracks[i].aposteriori_estimates
                pred2 = tracks[j].aposteriori_estimates
                max_time = max(list(pred1.keys()) + list(pred2.keys()))
                seg1 = np.array(list(pred1.values())[max_time:])
                seg2 = np.array(list(pred2.values())[max_time:])
                if len(seg1) < 3 or len(seg2) < 3:
                    continue
                rmse = TrackMerging.rmse(seg1, seg2)
                threshold = 2
                if rmse < threshold:
                    # actually merge the tracks
                    # for now just pick the one with the higher score
                    max_score = max(tracks[i].score, tracks[j].score)
                    if max_score == tracks[j].score:
                        print("THROWN in TM: ", tracks[i].obj_id, "OBS: ", tracks[i].observations)
                        indexes_to_remove.add(i)
                    else:  # max_score == tracks[i].score:
                        print("THROWN in TM: ", tracks[j].obj_id, "OBS: ", tracks[j].observations)
                        indexes_to_remove.add(j)
                    continue
        indexes_to_remove = list(indexes_to_remove)
        indexes_to_remove.sort(reverse=True)
        for s in indexes_to_remove:
            tracks.pop(s)

# track merging - case 2 and 3
# # case 2
#
# # case 3
# obs1 = tracks[i].observations
# obs2 = tracks[j].observations
# # Check that the last ts is a missed measurement and the second object is new
# if obs1[ts] is None and len(pred2.values()) < 3: #TODO change value to pruning.n
# 	dist = np.linalg.norm(pred1[ts], obs2)
