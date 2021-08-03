"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import networkx as nx
import networkx.algorithms.clique as nxac
import numpy as np

class HypothesisComp:

	def predict(self, ts, tracks):
		"""
		Uses maximum weight clique of a graph, where compatible tracks are connected by an edge and every
		node is a track with a specific score assigned in track maintenance to create the best hypothesis.

		Args:
			tracks (list): The list of all confirmed tracks.

		Returns:
			clique (list): the list of the best tracks.
		"""
		self.G = nx.Graph()

		index = 0

		# Calculate values needed to normalize the score
		# Only use confirmed tracks
		# scores = [track.score for track in tracks if track.confirmed()]

		confirmed_tracks = []
		for track in tracks:
			if track.confirmed():
				confirmed_tracks.append(track)
		tracks = confirmed_tracks

		if len(tracks) > 0:
			# Calculate values needed to normalize the score
			#scores = [track.score for track in tracks if track.confirmed()]
			scores = [track.score for track in tracks]
			minimum = min(scores)
			maximum = max(scores)
			if max(scores) != min(scores):
				dif = maximum - minimum
			else:
				dif = 1

			self.track_merge_rmse(ts, tracks) #TODO - move this to its own class at some point

			for track in tracks:
				self.G.add_node(index, weight = 1 + int(((track.score - minimum) / dif)*1000))
				index += 1

			for i in range(len(tracks)):
				for j in range(i):
					if i == j:
						continue
					# print(self.are_compatible(tracks[i], tracks[j])
					if self.are_compatible(tracks[i], tracks[j]):
						# track merging - case 2 and 3
						# # case 2
						#
						# # case 3
						# obs1 = tracks[i].observations
						# obs2 = tracks[j].observations
						# # Check that the last ts is a missed measurement and the second object is new
						# if obs1[ts] is None and len(pred2.values()) < 3: #TODO change value to pruning.n
						# 	dist = np.linalg.norm(pred1[ts], obs2)

						self.G.add_edge(i, j)
			result = nxac.max_weight_clique(self.G)
			clique = result[0]
		else:
			clique = []
		return clique

	def are_compatible(self, track1, track2):
		"""
		Checks whether two given tracks are compatible with each other (share an observation or root node)

		Args:
			track1 (Track): first track.
			track2 (Track): second track.

		Returns:
			(bool): Whether they are compatible or not.
		"""

		if len(track1.observations) > len(track2.observations):
			return self.are_compatible(track2, track1)
		for ts, obs in track1.observations.items():
			if ts in track2.observations.keys():
				if obs == track2.observations[ts]:
					return False
			else:
				continue
		return True

	def draw_graph(self):
		"""
		Draws the current graph

		"""

		nx.draw(self.G)

	@staticmethod
	def rmse(track1, track2):
		"""
		Assumes they are the same length and the time steps are synched.
		"""
		error = np.sqrt(np.square(track1 - track2).sum(axis=1))
		return np.sqrt(1 / len(track1) * np.sum(error, axis=1))

	def track_merge_rmse(self, ts, tracks):
		# track merging - case 1 & 2 (non consecutive)
		# synchronize time steps
		indexes_to_remove = set()
		for i in range(len(tracks)):
			for j in range(i):
				if i == j:
					continue
				pred1 = tracks[i].aposteriori_estimates
				pred2 = tracks[j].aposteriori_estimates
				max_time = max(list(pred1.keys()) + list(pred2.keys()))
				seg1 = np.array(list(pred1.values())[max_time:])
				seg2 = np.array(list(pred2.values())[max_time:])
				if len(seg1) < 3 or len(seg2) < 3:
					continue
				rmse = HypothesisComp.rmse(seg1, seg2)
				threshold = 1
				if rmse < threshold:
					# actually merge the tracks
					# for now just pick the one with the higher score
					max_score = max(tracks[i].score, tracks[j].score)
					if max_score == tracks[j].score:
						print("THROWN in HC: ", tracks[i].obj_id, "OBS: ", tracks[i].observations)
						indexes_to_remove.add(i)
					else:  # max_score == tracks[i].score:
						print("THROWN in HC: ", tracks[j].obj_id, "OBS: ", tracks[j].observations)
						indexes_to_remove.add(j)
					continue
		indexes_to_remove = list(indexes_to_remove)
		indexes_to_remove.sort(reverse=True)
		for s in indexes_to_remove:
			tracks.pop(s)
