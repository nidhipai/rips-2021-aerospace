import numpy as np


class MTTMetrics:
	# in this whole class, processes and trajectos are the ones that have been cleaned in sim

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



	@staticmethod
	def atct_signed(processes, trajectos, cut = 0):
		"""
		Outputs the along-track and cross-track error of the prediction.
		Asssumes processes and trajectories have been cleaned with sim.clean_process
		and sim.clean_trajectory methods.

		Args:
			processes (list): a list of ndarray representing true state vector over time for each object
			trajectos (list): a list of ndarray representing predicted state vector over time for each object
		"""


		# Returns lists of AT and CT errors for each object
		# Access AT errors of ith object with errors[i][0]
		# Access CT errors of ith object with errors[i][1]
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
			period_alive = (~np.isnan(proc))[0]
			proc = proc[:, period_alive]
			traj = trajectories[key][:,period_alive]
			# Calculate the errors
			marked_dist = np.linalg.norm(traj - proc, axis=0)

			# Test each process to see if a point at a given time step is closer
			# If a point from a different process is closer, mark this as a swap by setting to NaN
			for key2 in true_keys:
				if key != key2:
					proc2 = processes[key2][:,period_alive]
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
		motp = motp / len(true_keys)

		# Tally number of objects and hypotheses at each time step
		mota = 1 - (mota / total_possibilities)
		return motp, mota



