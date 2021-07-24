"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import numpy as np
import networkx as nx
import networkx.algorithms.clique as nxac

class HypothesisComp:
	def __init__(self):
		pass

	def predict(self, tracks):
		G = nx.Graph()
		index = 0
		for track in tracks:
			G.add_node(index, weight = track.score)
			index += 1
		for i in range(len(tracks)):
			for j in range(i):
				if self.are_compatible(tracks[i], tracks[j]):
					G.add_edge(i, j)
		result = nxac.max_weight_clique(G)
		clique = result[0]
		return clique

	def are_compatible(self, track1, track2):
		if len(track1.observations) > len(track2.observations):
			return self.are_compatible(track2, track1)
		for ts, obs in track1.observations.items():
			if ts in track2.observations.keys():
				if obs == track2.observations[ts]:
					return false
			else:
				continue
		return true
