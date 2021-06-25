import numpy as np
import DataGenerator


class SmartTruck(DataGenerator.DataGenerator):
    def __init__(self, xt0, ts, dt, ep_mag, ep_dir, nu):
        """
        Constructor for the SmartTruck Data Generator.
        :param xt0: Initial state vector
        :param ts: Time step
        :param dt: Length of one single time step
        :param ep_mag: Variance of the magnitude of the change in velocity
        :param ep_dir: Variance of the direction of the change in velocity. Specify as n-1 dimensional vector for n dimensional simulation.
        :param nu: Variance of the measurement noise
        """
        self.n = xt0.shape[0]

        Q = np.diag(np.append(np.zeros(self.n//2), np.append(np.array([ep_mag]), np.array(ep_dir))))
        R = np.eye(self.n//2)*nu

        super().__init__(xt0, ts, dt, Q, R)
        self.H = np.append(np.eye(self.n//2), np.zeros((self.n//2, self.n//2)), axis=1)
        self.A = np.append(np.append(np.eye(self.n//2), np.eye(self.n//2)*self.dt, axis=1), np.append(np.zeros((self.n//2,self.n//2)), np.eye(self.n//2), axis=1), axis=0)
        self.nu = nu


    def process_step(self, xt_prev):
        """
        Generate the next process state from the previous
        :param xt_prev: Previous process state
        :return: State vector of next step in the process
        """
        return np.matmul(self.A,xt_prev) + self.process_noise()

    def measure_step(self, xt):
        """
        Generate the next measure from the current process state vector
        :param xt: Current state vector
        :return: State vector representing measure at the current process state
        """
        return np.matmul(self.H, xt) + self.measure_noise()

    def process_noise(self):
        """
        Generate process noise
        """
        n2 = self.n // 2

        th = 2 * np.pi * np.random.normal(np.zeros(n2), np.diag(self.Q)[(n2 + 1):self.n])
        pk = np.ones((n2) - 1) * np.random.normal(np.zeros(n2), np.diag(self.Q)[n2])

        for i in range((n2) - 1):
            for j in range(i - 1):
                pk[i] *= np.sin(th[j])
            pk[i] *= np.cos(th[i])
        for j in range(n2 - 1):
            pk[(n2) - 1] *= np.sin(th[j])
        output = np.append(np.zeros(n2), pk)
        output.shape = (self.n, 1)
        return output

    def measure_noise(self):
        """
        Generate measurement noise
        """
        return np.random.normal(scale=self.nu, size=(self.n // 2, 1))

    def process_function(self, xt, u):
        return self.A @ xt

    def process_jacobian(self, xt, u):
        return self.A

    def measurement_function(self, xt):
        return self.process_function(xt, 0)[:self.n//2]

    def measurement_jacobian(self, xt):
        return self.H

