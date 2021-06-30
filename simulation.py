"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
Simulation
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
# plt.rcParams['text.usetex'] = True

import random as random
from mpl_toolkits import mplot3d
from kalmanfilter2 import KalmanFilter
from matplotlib.patches import Ellipse
import math


class Simulation:
    def __init__(self, generator, kFilter, seed_value=1):
        """
	    Constructs a simulation environment for one-line plotting data

		:param generator: Data Generator object
		:param kFilter: function for predicting the trajectory
		"""
        self.seed_value = seed_value
        self.generator = generator
        self.kFilter = kFilter
        self.kFilter_model = None
        self.n = generator.n
        self.processes = dict()
        self.measures = dict()
        self.trajectories = dict()
        self.ellipses = dict()

    def set_generator(self, generator):
        self.generator = generator

    def set_filter(self, kFilter):
        self.kFilter = kFilter

    def generate(self, time_steps):
        """
		Generates process and measurement data
		"""
        random.seed(self.seed_value)
        process = self.generator.process(time_steps)
        self.processes[len(self.processes.keys())] = process
        self.measures[len(self.measures.keys())] = self.generator.measure(process)

    def predict(self, index=None, x0=None, Q=None, R=None, H=None, u=None):
        output = np.empty((self.n, 1))
        # if any necessary variables for the filter have not been defined, assume we know them exactly
        if x0 is None:
            x0 = self.generator.xt0
        if Q is None:
            Q = self.generator.Q
        if R is None:
            R = self.generator.R
        if H is None:
            H = self.generator.H
        if index is None:
            index = len(self.measures.keys()) - 1

        f = self.generator.process_function
        jac = self.generator.process_jacobian
        h = self.generator.measurement_function

        self.kFilter_model = self.kFilter(x0, f, jac, h, Q, R, H, u)
        measures = []

        ellipses = []
        for i in range(self.measures[index][0].size):
            measure_t = self.measures[index][:, i]
            measure_t.shape = (self.n // 2, 1)
            measures.append(measure_t)
            self.kFilter_model.predict(measure_t, np.array(measures))
            kalman_output = self.kFilter_model.get_current_guess()
            output = np.append(output, kalman_output, axis=1)
            cov_ = self.kFilter_model.P[0:2, 0:2]
            mean_ = (self.kFilter_model.x_hat[0, 0], self.kFilter_model.x_hat[1, 0])
            ellipses.append(self.cov_ellipse(mean=mean_, cov=cov_))
        self.trajectories[len(self.trajectories.keys())] = output[:, 1:]  # delete the first column (initial data)
        err_arr = np.array(self.kFilter_model.error_array).squeeze()
		# self.cov_ellipse(err_arr, np.mean(err_arr, axis = 0), self.kFilter_model.R)
        self.ellipses[len(self.ellipses.keys())] = ellipses

    def plot(self, index=None, title="Object Position", x_label="x", y_label="y", z_label="z", ax=None, ellipse_freq=0):
        if index is None:
            index = len(self.processes.keys()) - 1
        process = self.processes[index]
        measure = self.measures[index]
        output = self.trajectories[index]
        ellipses = self.ellipses[index]
        legend = False
        if ax is None:
            fig, ax = plt.subplots()
            legend = True
        plt.rcParams.update({'font.size': 10})

        if self.n // 2 == 2:
            line1, = ax.plot(process[0], process[1], lw=1.5, markersize=8, marker=',')
            line2 = ax.scatter(measure[0], measure[1], s=15, lw=1.5, marker='+')
            line3, = ax.plot(output[0], output[1], lw=0.4, markersize=8, marker='.')
            lines = [line1, line2, line3]

            ax.set_title(title)
            ax.set_xlabel(y_label)
            ax.set_ylabel(x_label)
            ax.patches = []
            if ellipse_freq != 0:
                for j, ellipse in enumerate(ellipses):
                    if j % ellipse_freq == 0:
                        new_c=copy(ellipse)
                        ax.add_patch(new_c)
            # ax.set_aspect(1)
            # plt.figtext(.93, .5, "  Parameters \nx0 = ({},{})\nQ={}\nR={}\nts={}".format(str(self.generator.xt0[0,0]), str(self.generator.xt0[1,0]),str(self.generator.Q), str(self.generator.R), str(self.measures[index][0].size)))
            if legend is True:
                ax.legend(["Process", "Filter", "Measure", "Covariance"])
            return lines
        elif self.n // 2 == 3:
            # title = title + ", seed=" + str(self.seed_value)
            ax = plt.axes(projection='3d')
            ax.scatter3D(process[0], process[1], process[2], lw=1.5, marker=',')
            ax.scatter3D(measure[0], measure[1], measure[2], lw=0.4, marker='+')
            ax.scatter3D(output[0], output[1], output[2], lw=0.4, marker='.')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel(z_label)
            ax.set_title(title)
            plt.legend(["Process", "Filter", "Measure"])
            plt.show()
        else:
            print("Number of dimensions cannot be graphed.")

    def plot_all(self, ellipse_freq = 0):
        num_plots = len(self.processes)
        num_rows = int(np.ceil(num_plots / 3))
        if num_plots > 3:
            fig, ax = plt.subplots(num_rows, 3)
            fig.set_figheight(8)
            fig.set_figwidth(12)
            plt.subplots_adjust(hspace=.5, bottom=.15)
            lines = []
            for i in range(0, len(self.processes)):
                lines = self.plot(index=i, title="Plot " + str(i + 1), ax=ax[i // 3, i % 3], ellipse_freq=ellipse_freq)
            if num_plots % 3 == 1:  # one plot on last row
                ax[num_rows - 1, 1].remove()
            if num_plots % 3 != 0:  # one or two plots
                ax[num_rows - 1, 2].remove()
                fig.legend(handles=lines, labels=["Process", "Filter", "Measure"], loc='center',
                           bbox_to_anchor=(.73, .25))
            else:
                fig.legend(handles=lines, labels=["Process", "Filter", "Measure"], loc='lower center')
        else:
            self.plot(ellipse_freq=ellipse_freq)

    def cov_ellipse(self, mean, cov, zoom_factor=5, p=0.95):
        s = -2 * np.log(1 - p)
        w, v = np.linalg.eig(s*cov)
        print(v)
        ang = np.arctan2(v[0, 0], v[0, 1]) / np.pi * 180
        print(ang)
        ellipse = Ellipse(xy=mean, width=zoom_factor*w[0], height=zoom_factor*w[1], angle=ang, edgecolor='g', fc='none', lw=1)
        return ellipse

def cov_ellipse_fancy(X, mean, cov, p=[0.99, 0.95, 0.90]):
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(12, 12))
    # colors = Cube1_4.mpl_colors
    axes = plt.gca()
    axes.set_aspect(1)
    # colors_array = np.array([colors[0]] * X.shape[0])
    for i in range(len(p)):
        s = -2 * np.log(1 - p[i])
        w, v = np.linalg.eig(s * cov)
        # w = np.sqrt(w)
        ang = np.atan2(v[0, 0], v[1, 0]) / np.pi * 180
        ellipse = Ellipse(xy=mean, width=2 * w[0], height=2 * w[1], angle=ang, lw=2, fc="none", label=str(p[i]))
        cos_angle = np.cos(np.radians(180. - ang))
        sin_angle = np.sin(np.radians(180. - ang))

        x_val = (X[:, 0] - mean[0]) * cos_angle - (X[:, 1] - mean[1]) * sin_angle
        y_val = (X[:, 0] - mean[0]) * sin_angle + (X[:, 1] - mean[1]) * cos_angle

        rad_cc = (x_val ** 2 / (w[0]) ** 2) + (y_val ** 2 / (w[1]) ** 2)
        # colors_array[np.where(rad_cc <= 1.)[0]] = colors[i+1]

        axes.add_patch(ellipse)
    axes.scatter(X[:, 0], X[:, 1], linewidths=0, alpha=1)
    plt.legend(title="p-value", loc=2, prop={'size': 15}, handleheight=0.01)
    plt.show()

