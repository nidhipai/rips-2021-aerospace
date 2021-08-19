# Multi-Hypothesis Tracking of Space Objects and Targets (IPAM RIPS Aerospace 2021)

This is the project developed by the Aerospace team in the IPAM RIPS program in 2021, investigating satellite tracking with multi-hypothesis tracking (MHT). For a quick start, see the Dashboard section below.

## Introduction 

Over 20,000 artificial satellites currently orbit the Earth, and thousands more are launched each year. To prevent satellites from colliding as the sky becomes more cluttered, tracking the trajectories of objects in Earth's orbit is critical. Sensors such as radars are currently used to measure satellite locations; however, these measurements are inherently noisy and may include false alarms. Dealing with this noise is especially challenging when tracking many objects in close physical proximity, such as in the Starlink internet system satellite constellation. Multi-Hypothesis Tracking (MHT) is the most prevalent method for tracking multiple space objects. We present the MHT algorithm and our experiments for testing its capability in different tracking scenarios. This implementation features a novel track scoring criteria based on the Chi-squared test for variance. In addition, we develop heuristics for gating, thresholds for adding new tracks or incorporating missed measurements, and a modification to the traditional MHT algorithm that prevents new objects from being detected from false alarms prematurely.

## Data Generation

First, data is generated, and all data generator classes are children of the `data_generator.py` class. Data for multiple targets is generated in one of two files, `multiobject_simple.py` or `multiobject_fixed.py`. The difference is that the sensor for `multiobject_simple.py` can see the entire field, whereas the sensor in `multiobject_fixed.py` has a limited frame of view, and objects enter and exit the field of view.

## Multi-hypothesis Tracking

The primary algorithm developed is a multi-hypothesis tracking algorithm. The class responsible for MHT is `tracker3.py`, and the main method in that class is the `predict` method. There are several steps in the algorithm that run at every time step, and they mostly each have their own class with a `predict` method. TOMHT uses a tree of tracks, and these tracks are represented as `Track` objects (from `track.py`).

1. Scan in measurements: The predict method takes in an array of measurements for the time step.
2. Gating: Gating, contained in `gating_mht.py`, removes some observation-to-target matching possibilities, based on Mahalanobis distance or Euclidean distance. This class relies on the `Distances` class in `distances_mht.py`.
3. Track maintenance: Track maintenance, contained in `track_maintenance.py`, is responsible for both track creation and track scoring. In this project, we implement a novel method for track scoring based on the chi-squared test for variances.
4. Hypothesis computation: Contained in `hypothesis_comp.py`, this class is responsible for picking the best tracks. It relies on the max weighted clique algorithm from the NetworkX package.
5. Pruning: N-scan pruning is used in this implementation of MHT. There is a class called `pruning.py`, which removes tracks that do not share an ancestor with any best hypothesis track N times back.
6. Filter update: The filter update is performed in the predict method in for `tracker.py`. It uses the Kalman filter in `kalmanfilter3.py`.

## Single-Hypothesis Tracking

Before implementing the MHT algorithm, we also implemented a single-hypothesis tracking algorithm based on global nearest neighbors. The code for this algorithm is in the `pipeline` folder. The class responsible for coordinating SHT is the `Tracker` class in `tracker2.py`. The steps are as follows:

1. Scan in measurements: Measurements are passed into the `predict()` method of the `Tracker` class.
2. Gating: Like with MHT, gating (in `gating.py`) removes observation-to-target matching possibilities. This relies on the `Distances` class.
3. Data association: Data association (in `data_association.py`) uses global nearest neighbors based on Mahalanobis distance. This uses the linear sum assignment solver in the NumPy package.
4. Track maintenance: This class is responsible for creating new tracks for unassociated measurements and deleting old tracks that continually miss measurements.
5. Filter update: The `filter_predict.py` class’s predict method simply updates the predictions for each track using the Kalman filter in `kalmanfilter2.py`.

For a greater overview of multi-target tracking (MTT), including SHT and MHT, please refer to the survey paper [Vo 2015](http://ba-ngu.vo-au.com/vo/VMBCOMV_MTT_WEEE15.pdf).

## Error Metrics

The following metrics are used to measure algorithm performance and can be found/called as static methods of the MTTMetrics class in `mtt_metrics.py`.

For single-target tracking we use a simple root mean squared error (RMSE) to quantify performance.

For multiple objects, we used the Multi-Object Tracking Accuracy and Precision (MOTA and MOTP) metrics described in the paper [2008 Bernardin](https://link.springer.com/content/pdf/10.1155%2F2008%2F246309.pdf).

## Simulation

The product here contains simulation tools to generate object data, track the objects, visualize the trajectories and predictions, and then report error metrics about the tracking. These steps are all encapsulated in the `Simulation` class in `simulation.py`. The `Simulation` class is capable of storing results from multiple experiments.

## Files

The main algorithm is a track oriented multi-hypothesis tracking algorithm, stored in the `mht` folder.

Before implementing the MHT algorithm, we also implemented a single-hypothesis tracking algorithm based on global nearest neighbors. The code for this algorithm is in the `pipeline` folder.

The `html` folder stores the files necessary to generate documentation, created with Sphinx.

The `old` folder contains work from previous iterations of our work, mostly as we were testing the Kalman filter and working in the single target case.

## Set Up

### Dashboard

To run the dashboard, simply download the files, navigate to the `mtt` folder, and run the command `python dashboard.py`.

On Linux, it might be necessary to move the dashboard.py file outside the `mtt` folder and run the command `python dashboard.py` there.

Then, copy the localhost link and enter it into the url of an internet browser. There, further instructions on the dashboard itself will guide you regarding the specifics of using the dashboard.

### Other Usages

The simulation can also be run without the dashboard using a Jupyter notebook. Then, each class can be modified and tested individually, with plotting done in matplotlib. For an example of this, refer to `MHT_notebook_example.ipynb`.

Also, in `presets.py`, there is a method that returns an MHT `Tracker` object with standard parameters.

## Credits & Acknowledgements

This project is the work of Salvador Balkus, Nidhi Pai, Eduardo Sosa, and Tony Zeng. We are very grateful for the support of our academic mentor Jean-Michel Maldauge, as well as our industry mentors Daniel Agress, Jaime Cruz, James Gidney, and Ryan Handzo at The Aerospace Corporation.

We would also like to thank the Institute for Pure and Applied Math at UCLA for hosting us and The Aerospace Corporation for sponsoring our project.
