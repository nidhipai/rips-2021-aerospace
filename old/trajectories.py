import unittest
import numpy as np
from twod_object import TwoDObject
from simulation import Simulation
from kalmanfilter2 import KalmanFilter


def main():
	initial_kal = [np.array([[1], [1], [1], [0]])]
	dt = 0.1
	ep_normal = 1
	ep_tangent = 0
	nu = 0.1
	ts = 500
	miss_p = 0.2
	gen = TwoDObject(initial_kal, dt, ep_tangent, ep_normal, nu, miss_p)
	sim = Simulation(gen, KalmanFilter, 1)

	sim.generate(ts)
	p1s = []
	p2s = []
	for i in range(ts + 1):
		p1s.append(sim.processes[0][i][0][0][0])
		p2s.append(sim.processes[0][i][0][1][0])

	for i in range(ts + 1):
		print("(", round(p1s[i], 3), ", ", round(p2s[i], 3), ")")

main()
