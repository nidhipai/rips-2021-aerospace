import unittest
import numpy as np
from twod_object import TwoDObject
from

class TestDataGeneration(unittest.TestCase):

    def test_constructor(self):
        initial = [np.array([[0], [0], [0], [1]])]
        initial_kal = [np.array([[1], [1], [1], [0]])]
        dt = 0.1
        ep_normal = 0.01
        ep_tangent = 0
        nu = 0.1
        ts = 20
        miss_p = 0.2
        gen = TwoDObject(initial_kal, dt, ep_tangent, ep_normal, nu, miss_p)

        self.assertTrue(type(gen) == TwoDObject)



if __name__ == '__main__':
    unittest.main()