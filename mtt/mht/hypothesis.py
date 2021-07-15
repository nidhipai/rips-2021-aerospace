"""Aerospace Team - Eduardo Sosa, Nidhi Pai, Sal Balkus, Tony Zeng"""

import numpy as np

class Hypothesis:
	def __init__(self, tracks, ancestor):
		self.tracks = tracks
		self.ancestor = ancestor
		self.children = None

	
		