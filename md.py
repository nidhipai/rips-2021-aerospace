#Eduardo Sosa
#Mahalanobis Distance

import numpy as np
import matplotlib.pyplot as plt

def mhlb_dis(u, y, R, limit = 2):
	difference = y-u
	md = np.sqrt(difference.T@np.linalg.inv(R)@difference)
	if md < limit:
		return True
	else:
		return False


