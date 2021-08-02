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
		Calculates the Multi-Object Tracking Precision and Accuracy using formulas from
		Bernardin, Keni and Stiefelhagen, Rainer. "Evaluating Multiple Object Tracking Performance:
		The CLEAR MOT Metrics," Hindawi, 2008. doi:10.1155/2008/246309

		Note: This method assumes processes and trajectories have been cleaned by their
		respective methods in the Simulation class
		"""
		motp = 0
		mota = 0

		#TODO: Simplify use of traj_keys in this algorithm

		true_keys = []
		false_keys = []
		# Record the number of true and false keys
		for key in traj_keys:
			if type(key) is int:
				true_keys.append(key)
			else:
				false_keys.append(key)
		true_keys = np.array(true_keys)
		# Determine which time steps are marked correctly and calculate error based on RMSE
		print(true_keys)
		for i in range(len(true_keys)):
			marked_dist = np.linalg.norm(trajectories[i] - processes[i], axis=0)
			print("_________")
			print(marked_dist)
			# Test each process to see if a point at a given time step is closer
			# If a point from a different process is closer, mark this as a swap by setting to NaN
			for j in range(len(true_keys)):
				if i != j:
					cur_dist = np.linalg.norm(trajectories[i] - processes[j], axis=0)
					print(cur_dist)
					better = cur_dist < marked_dist
					marked_dist[better] = np.nan
			# Calculate the error for each time step when object is correctly identified (NaN)
			motp += marked_dist[~np.isnan(marked_dist)].sum()

			# Add the number of swaps and misses to the MOTA (misses start at NaN, swaps are added as NaN previously)
			mota += np.isnan(marked_dist).sum()

		# Count all of the times that the algorithm detected a false object and add to the MOTA
		for i in range(len(false_keys)):
			traj = trajectories[len(true_keys) + i]
			# Count how many time steps an entry is actually added
			mota += traj[~np.isnan(traj)].size

		# Divide by number of matches
		motp = motp / true_keys.size

		# Divide by the sum of the number of objects present over all time steps
		total_obj = sum([proc[~np.isnan(proc)].size for proc in processes])
		mota = 1 - (mota / total_obj)
		return motp, mota



