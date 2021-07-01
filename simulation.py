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
from matplotlib.patches import Ellipse
plt.rcParams["figure.figsize"] = (12,8)

#The Simulation class runs the data generator and the kalman filter to simulate an object in 2D.
class Simulation:
    def __init__(self, generator, kFilter, seed_value=1):
        """
        Constructs a simulation environment for one-line plotting data

        :param generator: Data Generator object
        :param kFilter: function for predicting the trajectory
        """
        self.seed_value = seed_value #a seed value for the random number generator
        self.generator = generator #our data generator
        self.kFilter = kFilter #the kalman filter class
        self.kFilter_model = None #kalman filter object
        self.n = generator.n #number of dimensions
        self.processes = dict() #dictionary for all the generated processes
        self.measures = dict() #dictionary for all the generated measures
        self.trajectories = dict() #predictions from the kalman filter
        self.descs = dict() #the descriptions for the graphs to be plotted
        self.ellipses = dict() #confidence ellipses

    #the generate functions takes in the number of time_steps of data to be generated and then proceeds to use the
    #data generator object to create the dictionary of processes and measures. 
    def generate(self, time_steps):
        """
        Generates process and measurement data
        """
        random.seed(self.seed_value)

        #we generate the process data and the measure data and assign it to the instances of processes and measures
        process = self.generator.process(time_steps)
        self.processes[len(self.processes.keys())] = process
        self.measures[len(self.measures.keys())] = self.generator.measure(process)

        # NOTE: This is hardcoded to support only one single object for now
        self.descs[len(self.descs.keys())] = {
            "x0": str(self.generator.xt0[0][0]),
            "y0": str(self.generator.xt0[0][1]),
            "Tangent Variance": str(self.generator.Q[2, 2]),
            "Normal Variance": str(self.generator.Q[3, 3]),
            "Measurement Noise Variance": str(self.generator.R[1, 1]),
            "Time Steps": str(time_steps)
        }

    #We use the kalman filter and the generated data to predict the trajectory of the simulated object
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

        #Extract the necessary functions and jacobians from the DataGenerator
        f = self.generator.process_function
        jac = self.generator.process_jacobian
        h = self.generator.measurement_function

        # Set up the filter with the desired parameters to test
        # NOTE: Currently hardcoded to be single target
        self.kFilter_model = self.kFilter(x0[0], f, jac, h, Q, R, H, u)

        # Set up lists to store objects for later plotting
        measures = []
        ellipses = []
        # Iterate through each time step for which we have measurements
        for measures_t in self.measures[index]:

            # Obtain a set of guesses for the current location of the object given the measurements
            # Note this will need to change later to incorporate multiple objects
            if len(measures_t) > 0:
                for measure_t in measures_t:
                    # Store each measure so we can compare future points to previous
                    # points using a distance metric
                    measures.append(measure_t)

                    # Process the point using the filter
                    self.kFilter_model.predict(measure_t, np.array(measures))
                    kalman_output = self.kFilter_model.get_current_guess()
                    output = np.append(output, kalman_output, axis=1)
            else:
                #If we don't have any measurements we need to guess for each object
                self.kFilter_model.predict(None, np.array(measures))
                kalman_output = self.kFilter_model.get_current_guess()
                output = np.append(output, kalman_output, axis=1)

            # Store the ellipse for later plottingS
            cov_ = self.kFilter_model.P[:2, :2]
            mean_ = (self.kFilter_model.x_hat[0, 0], self.kFilter_model.x_hat[1, 0])
            ellipses.append(self.cov_ellipse(mean=mean_, cov=cov_))

        # Store our output as an experiment
        self.trajectories[len(self.trajectories.keys())] = output[:, 1:]  # delete the first column (initial data)

        # Store the error of the Kalman filter
        err_arr = np.array(self.kFilter_model.error_array).squeeze()

        self.ellipses[len(self.ellipses.keys())] = ellipses

    def experiment(self, ts, **kwargs):
        for arg in kwargs.items():
            for value in arg[1]:
                self.generator = self.generator.mutate(**{arg[0]: value})
                self.generate(ts)
                self.predict()

    def experiment_plot(self, ts, var, **kwargs):
        """
        Run multiple experiments and plot all experiments run
        :param ts: Number of time steps to run
        :param var: Variable to display in title. This should change across experiments
        :param kwargs: Values to test in experiments
        :return:
        """
        self.experiment(ts, **kwargs)
        self.plot_all(var)


    '''We plot our trajectory based on the predicted trajectories given by the kalman filter object. '''
    def plot(self, var = "Time Steps", index=None, title="Object Position", x_label="x", y_label="y", z_label="z", ax=None, ellipse_freq=0):
        if index is None:
            index = len(self.processes.keys()) - 1

        #Create lists of points from the stored experiments
        process = self.processes[index]
        process = [point for sublist in process for point in sublist]
        process = np.array(process).squeeze().T


        measure = self.measures[index]
        measure = [point for sublist in measure for point in sublist]
        measure = np.array(measure).squeeze().T

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

            # Add the parameters we use. Note that nu is hardcoded as R[0,0] since the measurement noise is independent in both directions
            #ax.set_title(title + "\n" + self.descs[index], loc="left", y=1)
            ax.set_title(title + "\n" + var + " = " + str(self.descs[index][var]))
            ax.set_xlabel(y_label)
            ax.set_ylabel(x_label)
            ax.patches = []
            if ellipse_freq != 0:
                count = 0
                for j, ellipse in enumerate(ellipses):
                    if j % (1/ellipse_freq) == 0:
                        count+=1
                        # print(j)
                        # print("---", j)
                        new_c=copy(ellipse)
                        ax.add_patch(new_c)
                print(count)
            ax.set_aspect(1)
            ax.axis('square')

            #Below is an old method, if we want to include the full Q and R matrix
            #plt.figtext(.93, .5, "  Parameters \nx0 = ({},{})\nQ={}\nR={}\nts={}".format(str(self.generator.xt0[0,0]), str(self.generator.xt0[1,0]), str(self.generator.Q), str(self.generator.R), str(self.measures[index][0].size)))

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

    '''the plot_all function takes in a variable name, and an ellipse frequency between 0 and 1. Then, all stored experiments
    are plotted in one single figure with subplots'''
    def plot_all(self, var = "Time Steps", ellipse_freq = 0):
        num_plots = len(self.processes)
        num_rows = int(np.ceil(num_plots / 3))
        if num_plots > 3:
            fig, ax = plt.subplots(num_rows, 3)
            fig.set_figheight(8)
            fig.set_figwidth(12)
            plt.subplots_adjust(hspace=.5, bottom=.15)
            lines = []
            for i in range(0, len(self.processes)):
                lines = self.plot(index=i, var = var, ax=ax[i // 3, i % 3], ellipse_freq=ellipse_freq)
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
        plt.tight_layout()

    '''This function clears all the processes, measures, trajectories, descriptions, and the ellipses.'''
    def clear(self):
        self.processes = dict()
        self.measures = dict()
        self.trajectories = dict()
        self.descs = dict()
        self.ellipses = dict()


    '''The cov ellipse returns an ellipse path that can be added to a plot based on the given mean, covariance matrix
    zoom_factor, and the p-value'''
    def cov_ellipse(self, mean, cov, zoom_factor=2, p=0.95):
        #the s-value takes into account the p-value given
        s = -2 * np.log(1 - p)
        #the w and v variables give the eigenvalues and the eigenvectors of the covariance matrix scaled by s
        w, v = np.linalg.eig(s*cov)
        w = np.sqrt(w)
        #calculate the tilt of the ellipse
        ang = np.arctan2(v[0, 0], v[1, 0]) / np.pi * 180
        ellipse = Ellipse(xy=mean, width=zoom_factor*w[0], height=zoom_factor*w[1], angle=ang, edgecolor='g', fc='none', lw=1)
        return ellipse


'''The same as the cov_ellipse function, but draws multiple p-values depending on the passed on list. One can also 
plot the scattered values using this function to see which points are outliers. '''
def cov_ellipse_fancy(X, mean, cov, p=(0.99, 0.95, 0.90)):
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(12, 12))
    colors = ["black", "red", "purple", "blue"]
    colors = Cube1_4.mpl_colors
    axes = plt.gca()
    axes.set_aspect(1)
    colors_array = np.array([colors[0]] * X.shape[0])

    #for loop to individually draw each of the p-ellipses. 
    for i in range(len(p)):
        s = -2 * np.log(1 - p[i])
        w, v = np.linalg.eig(s * cov)
        w = np.sqrt(w)
        ang = np.arctan2(v[0, 0], v[1, 0]) / np.pi * 180
        ellipse = Ellipse(xy=mean, width=2 * w[0], height=2 * w[1], angle=ang,edgecolor=colors[i+1], lw=2, fc="none", label=str(p[i]))
        cos_angle = np.cos(np.radians(180. - ang))
        sin_angle = np.sin(np.radians(180. - ang))

        x_val = (X[:, 0] - mean[0]) * cos_angle - (X[:, 1] - mean[1]) * sin_angle
        y_val = (X[:, 0] - mean[0]) * sin_angle + (X[:, 1] - mean[1]) * cos_angle

        #calculating whether a point is inside an ellipse. If it is, we change the color of the point to a specific desired color. 
        rad_cc = (x_val ** 2 / (w[0]) ** 2) + (y_val ** 2 / (w[1]) ** 2)
        colors_array[np.where(rad_cc <= 1.)[0]] = colors[i+1]

        axes.add_patch(ellipse)
    #plot the scattered points with the ellipses. 
    axes.scatter(X[:, 0], X[:, 1], linewidths=0, alpha=1, c = colors_array)
    plt.legend(title="p-value", loc=2, prop={'size': 15}, handleheight=0.01)
    plt.show()


