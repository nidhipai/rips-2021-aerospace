"""
Sal Balkus, Nidhi Pai, Eduardo Sosa, Tony Zeng
RIPS 2021 Aerospace Team
"""

import numpy as np
import networkx as nx
import networkx.algorithms.clique as nxac


class HypothesisComp:
	"""
	Determines the current predictions by picking the best tracks using a max clique algorithm.
	"""
	def predict(self, tracks, no_new_obj=False):
		"""
		Create a graph where the nodes are tracks, with weight equal to track score, and two nodes have an edge
		if they are compatible. The max weight clique tracks are in the best hypothesis.

		Args:
			tracks (list): The list of all tracks.
		Returns:
			clique (list): Indexes of the best tracks.
		"""
		graph = nx.Graph()

		# Only use confirmed tracks, "confirmed" is defined in track
		# We only want to consider tracks that have had enough time to be pruned off
		confirmed_tracks = []
		for track in tracks:
			if track.confirmed() or no_new_obj:
				confirmed_tracks.append(track)

		if len(confirmed_tracks) > 0:
			scores = [np.log(confirmed_track.score) for confirmed_track in confirmed_tracks]
			# Normalize the scores
			minimum = min(scores)
			maximum = max(scores)
			if max(scores) != min(scores):
				dif = maximum - minimum
			else:
				dif = 1

			# Add track nodes to the graph with weights
			for i, track in enumerate(tracks):
				if track in confirmed_tracks:
					graph.add_node(i, weight=1 + int(((track.score - minimum) / dif) * 1000))

			# Add edges between compatible tracks
			for i in range(len(tracks)):
				if tracks[i] not in confirmed_tracks:
					continue
				for j in range(i):
					if tracks[j] not in confirmed_tracks:
						continue
					if self.are_compatible(tracks[i], tracks[j]):
						graph.add_edge(i, j)
			result = nxac.max_weight_clique(graph)
			clique = result[0]
		else:
			clique = []
		return clique

	def are_compatible(self, track1, track2):
		"""
		Checks whether two given tracks are compatible with each other (share an observation at any time).

		Args:
			track1 (Track): One of the tracks.
			track2 (Track): The other track.

		Returns:
			(bool): True if they are compatible.
		"""

		# Track 2 should be the longer one
		if len(track1.observations) > len(track2.observations):
			return self.are_compatible(track2, track1)

		for ts, obs in track1.observations.items():
			if ts in track2.observations.keys():
				if obs == track2.observations[ts]:
					return False
		return True
