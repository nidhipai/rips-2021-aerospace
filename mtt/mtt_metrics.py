import numpy as np


class MTTMetrics:
	# Returns AME of Euclidean distances between trajectory and actual process for each object
	# Access AME of ith object with errors[i]
	@staticmethod
	def AME_euclidean(processes, trajectos, cut=0):
		index = 0
		errors = []
		starts = MTTMetrics.find_first_value(trajectos)

		for i, process in enumerate(processes):
			start = starts[i]
			# If the whole trajectory is Nones, just set error to flag
			if len(trajectos[i][0]) == 0:
				errors.append(-1)
				continue
			diff_x = process[0][start:] - trajectos[index][0][start:]
			diff_y = process[1][start:] - trajectos[index][1][start:]
			diff_x = diff_x[cut:]
			diff_y = diff_y[cut:]
			errors.append(sum(np.sqrt(np.square(diff_x) + np.square(diff_y))) / len(diff_x))
			index += 1
		return errors

	# Returns RMSE of Euclidean distances between trajectory and actual process for each object
	# Access RMSE of ith object with errors[i]
	@staticmethod
	def RMSE_euclidean(processes, trajectos, cut = 0):
		errors = []
		starts = MTTMetrics.find_first_value(trajectos)

		for i, process in enumerate(processes):
			start = starts[i]
			# If the whole trajectory is Nones, just set error to flag
			if len(trajectos[i][0]) == 0:
				errors.append(-1)
				continue
			diff_x = process[0][start:] - trajectos[i][0][start:]
			diff_y = process[1][start:] - trajectos[i][1][start:]
			diff_x = diff_x[cut:]
			diff_y = diff_y[cut:]
			errors.append(np.sqrt(sum(np.square(diff_x) + np.square(diff_y)) / len(diff_x)))
		return errors


	# Returns lists of AT and CT errors for each object
	# Access AT errors of ith object with errors[i][0]
	# Access CT errors of ith object with errors[i][1]
	@staticmethod
	def atct_signed(processes, trajectos, cut = 0):
		errors = []

		# Check to make sure the trajectories are not None
		# If so, return flag value
		if trajectos[-1] is None:
			return [[[-1],[-1],[-1],[-1]]]

		# If trajectory starts with None values, need to find where to start the error calculation
		starts = MTTMetrics.find_first_value(trajectos)


		# This makes no sense
		for i, process in enumerate(processes):
			start = starts[i]
			diff_x = process[0][start:] - trajectos[i][0][start:]
			diff_y = process[1][start:] - trajectos[i][1][start:]
			vx = process[2][start:]
			vy = process[3][start:]
			diff_vx = vx - trajectos[i][2][start:]
			diff_vy = vy - trajectos[i][3][start:]
			angles = np.arctan2(vy, vx)
			diff_at = []
			diff_ct = []
			diff_atv = []
			diff_ctv = []
			for j, angle in enumerate(angles):
				c = np.cos(angle)
				s = np.sin(angle)
				diff_at.append(c * diff_x[j] - s * diff_y[j])
				diff_ct.append(s * diff_x[j] + s * diff_y[j])
				diff_atv.append(c * diff_vx[j] - s * diff_vy[j])
				diff_ctv.append(s * diff_vx[j] + s * diff_vy[j])
			diff_at = diff_at
			diff_ct = diff_ct
			diff_atv = diff_atv
			diff_ctv = diff_ctv
			errors.append([diff_at, diff_ct, diff_atv, diff_ctv])
		return errors

	# Returns 4 ratios based on TP, FP, TN, FN
	# Access recall with errors[0]
	# Access specificity with errors[1]
	# Access precision with errors[2]
	# Access last with errors[3]
	@staticmethod
	def false_id_rate(tfa_and_count, fa, cut = 0):
		errors = []
		pfa = []
		for index in range(len(fa[0])):
			curr = [fa[0][index] + fa[1][index]* 1j]
			pfa.append(curr)
		tfa = np.array(tfa_and_count[0])
		pfa = np.array(pfa)
		num_measures = tfa_and_count[1]
		TP = sum(sum(tfa.T == pfa))
		FP = len(pfa) - TP
		TN = num_measures - len(pfa) - (len(tfa) - TP)
		FN = len(tfa) - TP
		errors.append(TP / (TP + FN))
		errors.append(TN / (TN + FP))
		errors.append(TP / (TP + FP))
		errors.append(TN / (TN + FN))
		return errors

	# Simple method to calculate where the trajectories start; searches for first non-None value
	@staticmethod
	def find_first_value(trajectos):
		# If trajectory starts with None values, need to find where to start the error calculation
		cuts = []
		for traj in trajectos:
			cut = 0
			for k in range(len(traj[0])):
				if traj[0][k] is None:
					cut += 1
				else:
					cut += 1
					break
			cuts.append(cut)
		return cuts

