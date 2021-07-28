"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import networkx as nx
import networkx.algorithms.clique as nxac

class HypothesisComp:

	def predict(self, tracks):
		print("number of tracks: ", len(tracks))
		self.G = nx.Graph()

		index = 0

		# Calculate values needed to normalize the score
		scores = [track.score for track in tracks]
		minimum = min(scores)
		maximum = max(scores)
		if max(scores) != min(scores):
			dif = max(scores) - min(scores)
		else:
			dif = 1

		for track in tracks:
			# NOTE: hacky way to turn track scores into integers.
			# May want a better way to do this

			self.G.add_node(index, weight = int(((track.score - minimum) / dif)*1000))
			index += 1
		for i in range(len(tracks)):
			for j in range(i):
				# print(self.are_compatible(tracks[i], tracks[j]))
				if self.are_compatible(tracks[i], tracks[j]):
					self.G.add_edge(i, j)
		result = nxac.max_weight_clique(self.G)
		#print("NODES: ", len(G.nodes))
		clique = result[0]
		return clique

	def are_compatible(self, track1, track2):
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
		nx.draw(self.G)
