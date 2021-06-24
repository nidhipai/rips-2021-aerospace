import numpy as np
import DataGenerator


class SmartTruck(DataGenerator.DataGenerator):
    def __init__(self, xt0, ts, dt, ep_mag, ep_dir, nu):
        self.n = xt0.shape[0]
        th = 2*np.pi*np.sqrt(ep_dir)
        pk = np.ones(self.n//2)*np.sqrt(ep_mag)
        for i in range((self.n//2)-1):
            for j in range(i-1):
                pk[i] = pk[i]*np.sin(th[j])
            pk[i] = pk[i]*np.cos(th[i])
        for j in range((self.n)//2-2):
            pk[(self.n//2)-1] = pk[(self.n//2)-1] * np.sin(th[j])

        pk = np.power(pk, 2)
        Q = np.diag(np.append(np.zeros(self.n//2), pk.T))
        R = np.eye(self.n)*nu
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
        v_noise = np.random.normal(np.diag(self.Q))
        v_noise.shape = (self.n, 1)
        return v_noise

    def measure_noise(self):
        """
        Generate measurement noise
        """
        return np.random.normal(scale=self.nu, size=(self.n // 2, 1))