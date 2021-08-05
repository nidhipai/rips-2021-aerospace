import numpy as np


class MTTMetrics:
	# Returns AME of Euclidean distances between trajectory and actual process for each object
	# Access AME of ith object with errors[i]
	# TODO: Obsolte and no longer maintained
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
			if i >= len(trajectos):
				break
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

		# Pseudocode:
		# For each matching object-hypothesis pair:
		# Calculate the distance between the object and hypothesized track
		# Calculate the distance between the object and all other tracks
		# For each time step where the hypothesized track is closest, calculate RMSE (L2 norm) and add to MOTP
		# For each time step where the hypothesized track is NOT closest, add 1 to the MOTA (count swaps)
		# For each hypothesis without a matching object, add time steps where this hypothesis appears to MOTA
		# For each object without a matching hypothesis, add time steps where this object appears to MOTA
		# Divide MOTP by number of object-hypothesis matches
		# Divide MOTA by (number of objects + number of hypotheses)* time steps (max is no hypotheses overlap with any objects)

		# MOTP simply measures the error at the "correct" matches
		motp = 0

		# MOTA measures how frequently an object is misidentified or missed, including false alarms
		mota = 0
		total_possibilities = 0

		# Handle case for when there are no trajectories or processes
		if len(trajectories) == 0 or len(processes) == 0:
			return 0, 0

		# Record the number of true and false keys
		true_keys = [key for key in traj_keys if type(key) is int]
		true_keys.sort()
		false_keys = [key for key in traj_keys if type(key) is not int]
		# Determine which time steps are marked correctly and calculate error based on RMSE
		for key in true_keys:
			# Filter out observations before or after the process begins
			proc = processes[key]
			traj = trajectories[key]
			# NOTE: This throws an error when the process is shorter than the trajectory
			# We need both the proc and the traj to be NaN-passed on time steps in which they do not exist

			# Calculate the errors
			marked_dist = np.linalg.norm(traj - proc, axis=0)

			# Test each process to see if a point at a given time step is closer
			# If a point from a different process is closer, mark this as a swap by setting to NaN
			for key2 in true_keys:
				if key != key2:
					proc2 = processes[key2]
					cur_dist = np.linalg.norm(traj - proc2, axis=0)
					better = cur_dist < marked_dist
					marked_dist[better] = np.nan
			# Calculate the error for each time step when object is correctly identified (NaN)
			# Note that NaNs at the beginning and end represent areas where the object was only partially tracked correctly
			# This incorporates misses for objects we do identify at some point, but not perfectly
			motp += marked_dist[~np.isnan(marked_dist)].sum()

			# Calculate number of times object swaps

			mota += np.isnan(marked_dist).sum()

			# Add to the tally of total objects and hypotheses at each time step
			total_possibilities += proc.shape[1] + traj.shape[1]


		# Count all of the times that the algorithm detected a false object and add to the MOTA
		for i in range(len(false_keys)):
			traj = trajectories[len(true_keys) + i]

			# Count how many time steps an entry is actually added
			false_objs = np.sum(~np.isnan(traj)[0])

			# Add to MOTA and to the tally of total objects and hypotheses at each time step
			total_possibilities += false_objs

		# Count all of the times that the algorithm failed to detect a true process (not counting swaps) and add to the MOTA
		for proc in range(len(true_keys), len(processes)):
			undetected_objs = np.sum(~np.isnan(processes[proc])[0])
			total_possibilities += undetected_objs


		# Add all the measurements that could be potentially identified incorrectly as objects instead of false alarms,
		# minus the ones associated with objects

		# Divide by number of matches
		if len(true_keys) > 0:
			motp = motp / len(true_keys)
		else:
			motp = 0

		# Tally number of objects and hypotheses at each time step
		mota = 1 - (mota / total_possibilities)
		return motp, mota



