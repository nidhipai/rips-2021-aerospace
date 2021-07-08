import numpy as np
import matplotlib.pyplot as plt


class SingleTargetEvaluation:

	# in this class, truth/prediction are 3D arrays - for example, truth is a list of column vectors (and column vectors are 2D themselves)

	@staticmethod
	def center_error(truth, prediction):
		# returns a list of the norm
		return np.sqrt(np.square(truth - prediction).sum(axis=0))

	@staticmethod
	def average_error(truth, prediction):
		return 1 / len(truth) * np.sum(SingleTargetEvaluation.center_error(truth, prediction))

	@staticmethod
	def max_error(truth, prediction, n):
		"""
		Calculates the maximum error at a given time point after the first n time points.

		Args:
			truth (ndarray): the true process matrix output by the data generator.
			prediction (ndarray): the predicted trajectory output by the tracker or filter.

		Returns:
			numeric: the max error between the true and predicted values after n time steps
		"""
		truth_new = truth[:, n:]
		pred_new = prediction[:, n:]
		return np.max(SingleTargetEvaluation.center_error(truth_new, pred_new))

	@staticmethod
	def rmse(truth, prediction):
		norms = np.square(np.subtract(truth, prediction))
		return np.sqrt(1 / len(truth) * np.sum(norms))

	@staticmethod
	def failure_rate(truth, prediction, error_threshold=.5):
		# basically measures how many times it goes off track
		# not a very good measure for accuracy, but it tells you something about how much it relies on it's own prediction and follows a consistent path
		failures = 0
		off_track = False
		center_error = SingleTargetEvaluation.center_error(truth, prediction)
		for i in range(0, len(truth)):
			# print(center_error[i] - error_threshold)
			if center_error[i] > error_threshold:
				if not off_track:
					failures += 1
				off_track = True
			else:
				off_track = False
		return failures

	@staticmethod
	def center_error_plot(truth, prediction, ax = None):
		# this method isn't really used because it's in simulation
		if ax is None:
			fig, ax = plt.subplots()
		center_errors = SingleTargetEvaluation.center_error(truth, prediction)
		fig = plt.figure()
		plt.plot(center_errors)
