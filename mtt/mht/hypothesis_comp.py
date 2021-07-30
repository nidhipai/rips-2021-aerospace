"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import networkx as nx
import networkx.algorithms.clique as nxac

class HypothesisComp:

	def predict(self, tracks):
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
		scores = [track.score for track in tracks]
		minimum = min(scores)
		maximum = max(scores)
		if max(scores) != min(scores):
			dif = maximum - minimum
		else:
			dif = 1

		for track in tracks:

			self.G.add_node(index, weight = int(((track.score - minimum) / dif)*1000))
			index += 1
		for i in range(len(tracks)):
			for j in range(i):
				# print(self.are_compatible(tracks[i], tracks[j]))
				if self.are_compatible(tracks[i], tracks[j]):
					self.G.add_edge(i, j)
		result = nxac.max_weight_clique(self.G)
		clique = result[0]
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
