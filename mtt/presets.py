import mtt


class Presets:

	# TODO: Change default parameters to be empirically-tested heuristics
	@staticmethod
	def standardSHT(num_objects, params, gate_size=0.95, gate_expand_size=0, gate_method="mahalanobis",
	                num_init=2, num_init_frames=3, num_delete=3):
		gate = mtt.DistanceGating(gate_size, expand_gating=gate_expand_size, method=gate_method)
		assoc = mtt.DataAssociation()
		maintain = mtt.TrackMaintenance(mtt.KalmanFilter, params, num_obj=num_objects,
		                                num_init=num_init, num_init_frames=num_init_frames, num_delete=num_delete)
		filter_ = mtt.FilterPredict()
		return [gate, assoc, maintain, filter_]
