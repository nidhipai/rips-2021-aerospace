import unittest
import numpy as np
from twod_object import TwoDObject
from simulation import Simulation
from kalmanfilter2 import KalmanFilter

class Test(unittest.TestCase):

	# test TwoDObject construction
	def test_gen_constructor(self):#{{{
		initial_kal = [np.array([[1], [1], [1], [0]])]
		dt = 0.1
		ep_normal = 0.01
		ep_tangent = 0
		nu = 0.1
		ts = 20
		miss_p = 0.2
		gen = TwoDObject(initial_kal, dt, ep_tangent, ep_normal, nu, miss_p)

		self.assertTrue(type(gen) == TwoDObject)#}}}

	# test Simulation construction
	def test_sim_constructor(self):#{{{
		initial_kal = [np.array([[1], [1], [1], [0]])]
		dt = 0.1
		ep_normal = 0.01
		ep_tangent = 0
		nu = 0.1
		ts = 20
		miss_p = 0.2
		gen = TwoDObject(initial_kal, dt, ep_tangent, ep_normal, nu, miss_p)
		sim = Simulation(gen, KalmanFilter, 1)

		self.assertTrue(type(sim) == Simulation)#}}}

	# test Simulation process length
	def test_sim_process_size(self):#{{{
		initial_kal = [np.array([[1], [1], [1], [0]])]
		dt = 0.1
		ep_normal = 0.01
		ep_tangent = 0
		nu = 0.1
		ts = [10, 21, 101, 200, 1000]
		miss_p = 0.2
		gen = TwoDObject(initial_kal, dt, ep_tangent, ep_normal, nu, miss_p)
		sim = Simulation(gen, KalmanFilter, 1)

		trials = len(ts)
		for i in range(trials):
			sim.generate(ts[i])

		status = True
		for i in range(trials):
			if len(sim.processes[i]) != ts[i] + 1:
				status = False

		self.assertTrue(status)#}}}

	# test measurement generation
	# In particular, check stdev of the measurement noise
	def test_sim_measurements(self):#{{{
		initial_kal = [np.array([[1], [1], [1], [0]])]
		dt = 0.1
		ep_normal = 0.01
		ep_tangent = 0
		nu = 0.1
		ts = 1000
		miss_p = 0.2
		gen = TwoDObject(initial_kal, dt, ep_tangent, ep_normal, nu, miss_p)
		sim = Simulation(gen, KalmanFilter, 1)

		sim.generate(ts)
		sim.predict()

		total_pos_1 = 0
		total_pos_2 = 0
		count = 0
		for i in range(ts + 1):
			if len(sim.measures[0][i]) > 0:
				x = np.array(sim.measures[0][i][0]) - np.array(sim.processes[0][i][0][0:2])
				total_pos_1 += x[0][0]
				total_pos_2 += x[1][0]
				count += 1
		mean_1 = total_pos_1 / count
		mean_2 = total_pos_2 / count
		usv_1 = 0
		usv_2 = 0
		for i in range(ts + 1):
			if len(sim.measures[0][i]) > 0:
				x = np.array(sim.measures[0][i][0]) - np.array(sim.processes[0][i][0][0:2])
				temp_1 = x[0][0] - mean_1
				temp_2 = x[1][0] - mean_2
				usv_1 += temp_1 * temp_1
				usv_2 += temp_2 * temp_2
		print("ACTUAL: ", nu, "EMPIRICAL: ", round(np.sqrt(usv_1 / (count - 1)), 3))
		print("ACTUAL: ", nu, "EMPIRICAL: ", round(np.sqrt(usv_2 / (count - 1)), 3))#}}}

	# NOTE: bug, flipped tangent and normal
	def test_sim_processes(self):#{{{
		initial_kal = [np.array([[1], [1], [1], [0]])]
		dt = 0.1
		ep_normal = 0.01
		ep_tangent = 0
		nu = 0.1
		ts = 500
		miss_p = 0.2
		gen = TwoDObject(initial_kal, dt, ep_tangent, ep_normal, nu, miss_p)
		sim = Simulation(gen, KalmanFilter, 1)

		sim.generate(ts)
		v1s = []
		v2s = []
		for i in range(ts + 1):
			v1s.append(sim.processes[0][i][0][2][0])
			v2s.append(sim.processes[0][i][0][3][0])

		def rot_mat_from_ang(x, y):
			ang = np.arctan2(y, x)
			c = np.cos(-ang)
			s = np.sin(-ang)
			return np.array([[c, -s], [s, c]])

		vel_noise_recon_1 = []
		vel_noise_recon_2 = []
		for i in range(ts):
			v1 = v1s[i + 1] - v1s[i]
			v2 = v2s[i + 1] - v2s[i]
			rot = rot_mat_from_ang(v1s[i], v2s[i])
			vel_noise = np.array([v1, v2])
			vel_noise.shape = (2, 1)
			vel_noise_rot = rot @ vel_noise
			vel_noise_recon_1.append(vel_noise_rot[0][0]) #tangent
			vel_noise_recon_2.append(vel_noise_rot[1][0]) #normal
		#print(vel_noise_recon_1)
		#print(vel_noise_recon_2)
		mean_1 = sum(vel_noise_recon_1) / len(vel_noise_recon_1)
		mean_2 = sum(vel_noise_recon_2) / len(vel_noise_recon_2)

		usv_1 = 0
		usv_2 = 0
		for i in range(ts):
			temp_1 = vel_noise_recon_1[i] - mean_1
			temp_2 = vel_noise_recon_2[i] - mean_2
			usv_1 += temp_1 * temp_1
			usv_2 += temp_2 * temp_2
		print("ACTUAL (tangent): ", ep_tangent, "EMPIRICAL: ", round((usv_1 / (ts)), 3))
		print("ACTUAL (normal): ", ep_normal, "EMPIRICAL: ", round((usv_2 / (ts)), 3))#}}}

	# test proportion of missed measurements
	def test_sim_missed_prop(self):#{{{
		initial_kal = [np.array([[1], [1], [1], [0]])]
		dt = 0.1
		ep_normal = 0.01
		ep_tangent = 0
		nu = 0.1
		ts = 1000
		miss_p = 0.2
		gen = TwoDObject(initial_kal, dt, ep_tangent, ep_normal, nu, miss_p)
		sim = Simulation(gen, KalmanFilter, 1)

		sim.generate(ts)
		sim.predict()

		count = 0
		for i in range(ts + 1):
			if len(sim.measures[0][i]) > 0:
				count += 1
		print("ACTUAL: ", miss_p, "EMPIRICAL: ", round(1 - count / (ts + 1), 3))#}}}

	# test trajectory size
	# In particular, test to ensure that trajectories have same length as processes
	def test_trajectory_size(self):#{{{
		initial_kal = [np.array([[1], [1], [1], [0]])]
		dt = 0.1
		ep_normal = 0.01
		ep_tangent = 0
		nu = 0.1
		ts = [10, 21, 101, 200, 1000]
		miss_p = 0.2
		gen = TwoDObject(initial_kal, dt, ep_tangent, ep_normal, nu, miss_p)
		sim = Simulation(gen, KalmanFilter, 1)

		status = True
		for i in range(len(ts)):
			sim.generate(ts[i])
			sim.predict()
			if not(len(sim.trajectories[i]) == 4):
				status = False
				break
			for j in range(4):
				if not(len(sim.processes[i]) == len(sim.trajectories[i][j])):
					status = False
					break
		self.assertTrue(status)#}}}

	#def next_test(self):


if __name__ == '__main__':
	unittest.main()
