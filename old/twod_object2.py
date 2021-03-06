"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
Simulation
"""

import numpy as np
import pandas as pd
from copy import copy
from data_generator import DataGenerator

#Child class for data_generator where we simulate a two-D object with its position and velocity as part of the state vector
class TwoDObject(DataGenerator):
    def __init__(self, xt0, dt, ep_tangent, ep_normal, nu, miss_p=0, lam=0, fa_scale=10):
        """
        Constructor for the 2DObject Data Generator.

        :param xt0: Initial state vector
        :param dt: Length of one single time step
        :param ep_normal: Variance of the change in velocity vector in the normal direction.
        :param ep_tangent: Variance of the change in velocity vector in the tangent direction.
        :param nu: Variance of the measurement noise
        :param miss_p: Probability of missing a measurement
        :param lam: Expected number of false alarms per time step
        :param fa_scale: Scaling of measurement noise on a false alarm

        """
        self.dim = 2 #We work in a two dimensional space
        self.n = 4 #dimension of the state vector


        self.ep_tangent = ep_tangent #variance of the process noise tangent to the velocity vector
        self.ep_normal = ep_normal #variance of the process noise normal to the velocity vector
        self.nu = nu #variance of the measuremet noise.
        self.miss_p = miss_p #proportion of missed measurements in the generation of the data.
        self.lam = lam
        self.fa_scale = fa_scale

        #We require our initial state vector to have all 4 needed components: x,y, velocity in the x direction, velocity in the y direction
        if xt0[0].size != 4:
            raise Exception("Length of initial state vector does not equal 4")

        #We set the process noise covariance matrix to 
        self.Q = np.diag(np.append(np.zeros(self.dim), np.append(np.array([ep_tangent]), np.array(ep_normal))))
        self.R = np.eye(self.dim) * nu

        super().__init__(xt0, dt, self.Q, self.R)

        #Jacobian matrices for the h function and the f function.
        self.H = np.append(np.eye(self.dim), np.zeros((self.dim, self.dim)), axis=1)
        self.A = np.append(np.append(np.eye(self.dim), np.eye(self.dim) * self.dt, axis=1),
                           np.append(np.zeros((self.dim, self.dim)), np.eye(self.dim), axis=1), axis=0)
        self.nu = nu #measurement noise variance
        self.dt = dt
        self.state_names = ["x","y","vx","vy"]

    def process_step(self, xt_prevs, rng):
        """
        Generate the next process state from the previous
        :param xt_prev: Previous process state
        :return: State vector of next step in the process
        """
        df = pd.DataFrame(columns = ["step","timesl","x","y","vx","vy","obj"])
        # Iterate through each state in the list of previous object states
        for obj in df.obj.unique():
            # index the next state vector with time step
            prev = df.loc[np.where((df.obj == 1) & (df.step == df.step.max()))[0]]
            xt_prev = np.array(prev[self.state_names])
            xt_prev.shape = (4,1)
            #Index the time step for this object's state vector
            dt = prev["timesl"][0]

            # calculate the next state and add to output
            xt_output= self.A @ xt_prev + dt*self.process_noise(xt_prev, rng)
            xt_output = xt_output = pd.DataFrame(xt_output.T, columns = ["x","y","vx","vy"])

        pass

    def measure_step(self, xts, rng):
        """
        Generate the next measure from the current process state vector
        :param xt: Current state vector
        :return: State vector representing measure at the current process state
        """

        # Iterate through each object state in the input
        output = []
        for xt in xts:
            #Calculate whether the measurement is missed
            if np.random.rand() > self.miss_p:
                output.append(self.H @ xt + self.measure_noise(rng))

        for i in range(rng.poisson(self.lam)):
            output.append(self.H @ xt + self.measure_noise(rng)*self.fa_scale)

        return output

    def measure_noise(self, rng):
        """
        Generate measure noise
        """
        return rng.normal(scale=self.nu, size=(self.dim, 1))

    def process_noise(self, xt, rng):

        """
        Generate process noise
        :param xt: current state vector
        :return: vector of noise for each parameter in the state vector
        """

        #NOTE: if the angle is 90 degrees then 0 is returned
        #Also this uses radians
        pad = np.array([0, 0])
        rotation = self.W(xt)[2:4, 2:4]
        noise = rng.multivariate_normal((0, 0), rotation @ self.Q[2:4, 2:4] @ rotation.T)
        output = np.append(pad, noise)
        output.shape = (4, 1)
        return output

    def W(self, xt):
        ang = np.arctan2(xt[3, 0], xt[2, 0])
        c = np.cos(ang)
        s = np.sin(ang)

        return np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, c, -s], [0, 0, s, c]])

    def process_function(self, xt, u):
        return self.A @ xt

    def process_jacobian(self, xt, u):
        return self.A

    def measurement_function(self, xt):
        return self.H @ xt

    def measurement_jacobian(self, xt):
        return self.H

    def mutate(self, **kwargs):
        clone = copy(self)
        for arg in kwargs.items():
            setattr(clone, arg[0], arg[1])
        return TwoDObject(clone.xt0, clone.dt, clone.ep_tangent, clone.ep_normal, clone.nu, clone.miss_p)



