import mtt
import numpy as np

class Presets:

	# # TODO: Change default parameters to be empirically-tested heuristics
	# @staticmethod
	# def standardSHT(num_objects, params, gate_size=0.95, gate_expand_size=0, gate_method="mahalanobis",
	#                 num_init=2, num_init_frames=3, num_delete=3):
	# 	gate = mtt.DistanceGating(gate_size, expand_gating=gate_expand_size, method=gate_method)
	# 	assoc = mtt.DataAssociation()
	# 	maintain = mtt.TrackMaintenance(mtt.KalmanFilter, params, num_obj=num_objects,
	# 	                                num_init=num_init, num_init_frames=num_init_frames, num_delete=num_delete)
	# 	filter_ = mtt.FilterPredict()
	# 	return mtt.MTTTracker([gate, assoc, maintain, filter_])

	@staticmethod
	def standardMHT(params, miss_p, lam, gate_size=0.95, gate_expand_size=0, gate_method="mahalanobis",
	                tot=0.00001, tmm=0.1, tnt=1, born_p = 0.05, prune_time=4, scoring_method="chi2", P=None, starting_pos=None):
		if "P" in params.keys():
			params.pop("P")
		if P is None:
			P = np.eye(4)

		k = mtt.KalmanFilter(**params)
		gate = mtt.DistanceGatingMHT(gate_size, gate_expand_size, gate_method)
		if starting_pos is not None:
			main = mtt.TrackMaintenanceMHT(tot, tmm, tnt, 1 - miss_p, 4, lam, params['R'], P, k, prune_time, scoring_method, born_p, len(starting_pos))
		else:
			main = mtt.TrackMaintenanceMHT(tot, tmm, tnt, 1 - miss_p, 4, lam, params['R'], P, k, prune_time, scoring_method, born_p, 0)
		hypo = mtt.HypothesisComp()
		prune = mtt.Pruning(prune_time)

		return mtt.MHTTracker(k, gate, main, hypo, prune, starting_pos)



