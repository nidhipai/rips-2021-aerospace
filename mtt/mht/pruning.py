"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import numpy as np

class Pruning:
	def __init__(self, n):

		self.n = n # The number of time steps from which to prune back

	def predict(self, tracks, best_tracks):
		"""

		Extracts and stores the sequences of measurements that correspond to valid tracks.
		Gathers the required observations that feed into the best tracks
		before the specified time step and then gets rid of any tracks that do not feed into 
		those tracks. 

		Args:
			tracks (list): a list of all possible tracks at the current time step.
			best_tracks (list): list of indices of the best tracks selected at the current time step.
		"""

		required_obs = []
		for index in best_tracks:
			prev_obs = np.array(list(tracks[index].observations.values()))
			#print("P", index, prev_obs)
			required_obs.append(prev_obs[:(prev_obs.size - 0 - self.n)])
		#required_obs = np.array(required_obs)
		#print("R", required_obs)

		# Test each track to see whether its initial sequence leads to a valid part of the tree
		for i in reversed(range(len(tracks))):
			track = tracks[i]
			keep = False
			# Extract the first part of the sequence of measurements, up to n
			prev_ob = np.array(list(track.observations.values()))
			# if prev_ob.size - 0 - self.n <= 0:
			# 	continue
			if prev_ob.size - 0 - self.n <= 0:
				continue
			prev_ob = prev_ob[:(prev_ob.size - 0 - self.n)]
			# Test each possibility
			for required_ob in required_obs:
				if required_ob.size == prev_ob.size and (required_ob == prev_ob).all():
					keep = True

			# Remove the current track if its initial sequence of measurements does not match the current best hypothesis up to n
			if not keep:
				#print("THROWN: ", track.obj_id, "OBS: ", track.observations)
				tracks.remove(track)

