import numpy as np


class MTTMetrics:
	# Returns AME of Euclidean distances between trajectory and actual process for each object
	# Access AME of ith object with errors[i]
	@staticmethod
	def AME_euclidean(processes, trajectos, cut=0):
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
			#errors.append(np.sqrt(sum(np.square(diff_x) + np.square(diff_y) / len(diff_x))))
			errors.append(np.sqrt(sum(np.square(diff_x) + np.square(diff_y)) / len(diff_x)))
			#errors.append(np.sqrt(sum(np.square(diff_x) + np.square(diff_y))) / len(diff_x))
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
			diff_vx = vx - trajectos[i][2]
			diff_vy = vy - trajectos[i][3]
			angles = np.arctan2(vy, vx)
			j = 0
			diff_at = []
			diff_ct = []
			diff_atv = []
			diff_ctv = []
			for angle in angles:
				c = np.cos(angle)
				s = np.sin(angle)
				diff_at.append(c * diff_x[j] - s * diff_y[j])
				diff_ct.append(s * diff_x[j] + s * diff_y[j])
				diff_atv.append(c * diff_vx[j] - s * diff_vy[j])
				diff_ctv.append(s * diff_vx[j] + s * diff_vy[j])
				j += 1
			diff_at = diff_at[cut:]
			diff_ct = diff_ct[cut:]
			diff_atv = diff_atv[cut:]
			diff_ctv = diff_ctv[cut:]
			errors.append([diff_at, diff_ct, diff_atv, diff_ctv])
			i += 1
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

	@staticmethod
	def mota_motp(processes, trajectories, traj_keys):
		"""
		Calculates the Multi-Object Tracking Precision and Accuracy
		"""
		motp = 0
		mota = 0

		true_keys = []
		false_keys = []
		for key in traj_keys:
			if type(key) is int:
				true_keys.append(key)
			else:
				false_keys.append(key)

		for key in true_keys:
			motp += np.sqrt(np.nan_to_num(np.power(processes[key] - trajectories[key],2)).sum(axis=0)).sum()

		motp = motp / len(true_keys)
		return motp, mota



