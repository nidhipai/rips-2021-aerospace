import numpy as np

class Metrics:
	# Returns AME of Euclidean distances between trajectory and actual process for each object
	# Access AME of ith object with errors[i]
	@staticmethod
	def AME_euclidean(processes, trajectos, cut = 0):
		index = 0
		errors = []
		for process in processes:
			diff_x = process[0] - trajectos[index][0]
			diff_y = process[1] - trajectos[index][1]
			diff_x = diff_x[cut:]
			diff_y = diff_y[cut:]
			errors.append(sum(np.sqrt(np.square(diff_x) + np.square(diff_y))) / len(diff_x))
			index += 1
		return errors

	# Returns RMSE of Euclidean distances between trajectory and actual process for each object
	# Access RMSE of ith object with errors[i]
	@staticmethod
	def RMSE_euclidean(processes, trajectos, cut = 0):
		index = 0
		errors = []
		for process in processes:
			diff_x = process[0] - trajectos[index][0]
			diff_y = process[1] - trajectos[index][1]
			diff_x = diff_x[cut:]
			diff_y = diff_y[cut:]
			errors.append(np.sqrt(sum(np.square(diff_x) + np.square(diff_y) / len(diff_x))))
			index += 1
		return errors

	# Returns lists of AT and CT errors for each object
	# Access AT errors of ith object with errors[i][0]
	# Access CT errors of ith object with errors[i][1]
	@staticmethod
	def atct_signed(processes, trajectos, cut = 0):
		i = 0
		errors = []
		for process in processes:
			diff_x = process[0] - trajectos[i][0]
			diff_y = process[1] - trajectos[i][1]
			vx = process[2]
			vy = process[3]
			angles = np.arctan2(vy, vx)
			j = 0
			diff_at = []
			diff_ct = []
			for angle in angles:
				c = np.cos(angle)
				s = np.sin(angle)
				diff_at.append(c * diff_x[j] - s * diff_y[j])
				diff_ct.append(s * diff_x[j] + s * diff_y[j])
				j += 1
			diff_at = diff_at[cut:]
			diff_ct = diff_ct[cut:]
			errors.append([diff_at, diff_ct])
			i += 1
		return errors
