import numpy as np
import matplotlib.pyplot as plt


class SingleTargetEvaluation:

	def __init__(self, simulation, time_steps, runs = 1):
		self.simulation = simulation
		self.time_steps = time_steps
		self.runs = runs

	def run(self):
		self.truth = []
		self.prediction = []
		for i in range(self.runs):
			self.simulation.generate(self.time_steps)
			self.simulation.predict()
			truth_sim = self.simulation.clean_process(self.simulation.processes[i])[0][:2,:]
			predicted_sim = self.simulation.clean_trajectory(self.simulation.trajectories[i])[0]
			self.truth.append(truth_sim)
			self.prediction.append(predicted_sim)
		self.truth = np.array(self.truth)		
		self.prediction = np.array(self.prediction)

	def plot_error(self, x, y, title = None, x_label = None, y_label = None):
		fig,ax = plt.subplots()
		ax.plot(x, y, color = "red")
		ax.set_title(title)
		ax.set_xlabel(x_label)
		ax.set_ylabel(y_label)
		plt.show()

	def center_error(self, truth, prediction):
		# returns a list of the norm
		error = np.sqrt(np.square(truth - prediction).sum(axis=1))
		error = np.mean(error, axis = 0)
		return error

	@staticmethod
	def average_error(self, truth, prediction):
		return np.mean(np.sqrt(np.square(truth - prediction).sum(axis=1)),axis = 1)

	def max_error(self, n = 0):
		"""
		Calculates the maximum error at a given time point after the first n time points.

		Args:
			truth (ndarray): the true process matrix output by the data generator.
			prediction (ndarray): the predicted trajectory output by the tracker or filter.

		Returns:
			numeric: the max error between the true and predicted values after n time steps
		"""
		truth_new = self.truth[:, n:]
		pred_new = self.prediction[:, n:]
		return np.max(np.sqrt(np.square(self.truth - self.prediction).sum(axis=1)), axis = 1)

	def rmse(self):
		norms = np.square(self.truth - self.prediction).sum(axis = 1)
		print(norms.shape)
		return np.sqrt(1 / len(self.truth) * np.sum(norms, axis = 1))

	# def failure_rate(self,error_threshold=.5):
	# 	# basically measures how many times it goes off track
	# 	# not a very good measure for accuracy, but it tells you something about how much it relies on it's own prediction and follows a consistent path
	# 	failures = 0
	# 	off_track = False
	# 	center_error = np.sqrt(np.square(self.truth - self.prediction).sum(axis=1))
	# 	for i in range(self.truth.shape[0]):
	# 		for j in range(self.truth.shape[1]):
	# 		# print(center_error[i] - error_threshold)
	# 		if center_error[i] > error_threshold:
	# 			if not off_track:
	# 				failures += 1
	# 			off_track = True
	# 		else:
	# 			off_track = False
	# 	return failures
