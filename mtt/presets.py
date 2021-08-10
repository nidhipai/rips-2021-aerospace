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
	                tot=0.00001, tmm=0.1, tnt=0.7, prune_time=4, scoring_method = "chi2", P = None):
		if "P" in params.keys():
			params.pop("P")
		if P is None:
			P = np.eye(4)

		k = mtt.KalmanFilter(**params)
		gate = mtt.DistanceGatingMHT(gate_size, gate_expand_size, gate_method)
		main = mtt.TrackMaintenanceMHT(tot, tmm, tnt, 1 - miss_p, 4, lam, params['R'], P, k, prune_time, scoring_method)
		hypo = mtt.HypothesisComp()
		prune = mtt.Pruning(prune_time)

		return mtt.MHTTracker(k, gate, main, hypo, prune)



