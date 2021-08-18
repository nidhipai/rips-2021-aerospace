"""
Sal Balkus, Nidhi Pai, Eduardo Sosa, Tony Zeng
RIPS 2021 Aerospace Team
"""

import mtt
import numpy as np


class Presets:
	"""
	Creates a preset set of input parameters for the tracker class
	"""
	@staticmethod
	def standardMHT(params, miss_p, lam, gate_size=0.95, gate_expand_size=0, gate_method="mahalanobis",
	                tot=0.001, tmm=0.1, tnt=1, born_p = 0, prune_time=5, scoring_method="chi2", P=None, starting_pos=None):
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



