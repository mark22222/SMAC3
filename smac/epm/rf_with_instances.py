import numpy as np
import logging

import pyrfr

from smac.configspace import CategoricalHyperparameter
from smac.epm.base_epm import AbstractEPM


__author__ = "Aaron Klein"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Aaron Klein"
__email__ = "kleinaa@cs.uni-freiburg.de"
__version__ = "0.0.1"


class RandomForestWithInstances(AbstractEPM):

    '''
    Interface to the random forest that takes instance features
    into account.

    Parameters
    ----------
    types: np.ndarray (D)
        Specifies the number of categorical values of an input dimension. Where
        the i-th entry corresponds to the i-th input dimension. Let say we have
        2 dimension where the first dimension consists of 3 different
        categorical choices and the second dimension is continuous than we
        have to pass np.array([2, 0]). Note that we count starting from 0.
    bounds: np.ndarray (D)
        Specifies the bounds for continuous features
    instance_features: np.ndarray (I, K)
        Contains the K dimensional instance features
        of the I different instances
    num_trees: int
        The number of trees in the random forest.
    do_bootstrapping: bool
        Turns on / off bootstrapping in the random forest.
    ratio_features: float
        The ratio of features that are considered for splitting.
    min_samples_split: int
        The minimum number of data points to perform a split.
    min_samples_leaf: int
        The minimum number of data points in a leaf.
    max_depth: int

    eps_purity: float

    max_num_nodes: int

    seed: int
        The seed that is passed to the random_forest_run library.
    '''

    def __init__(self, types, bounds,
                 instance_features=None,
                 num_trees=10,
                 do_bootstrapping=True,
                 n_points_per_tree=-1,
                 ratio_features=5. / 6.,
                 min_samples_split=3,
                 min_samples_leaf=3,
                 max_depth=20,
                 eps_purity=1e-8,
                 max_num_nodes=1000,
                 seed=42):

        self.instance_features = instance_features
        self.types = types
        self.bounds = bounds
        self.rng = pyrfr.regression.default_random_engine(seed)

        self.rf_opts = pyrfr.regression.forest_opts()
        self.rf_opts.num_trees = num_trees
        self.rf_opts.seed = seed
        self.rf_opts.do_bootstrapping = do_bootstrapping
        max_features = 0 if ratio_features >= 1.0 else \
            max(1, int(types.shape[0] * ratio_features))
        self.rf_opts.max_features = max_features
        self.rf_opts.min_samples_to_split = min_samples_split
        self.rf_opts.min_samples_in_leaf = min_samples_leaf
        self.rf_opts.max_depth = max_depth
        self.rf_opts.epsilon_purity = eps_purity
        self.rf_opts.max_num_nodes = max_num_nodes

        self.n_points_per_tree = n_points_per_tree
        self.rf = None  # type: pyrfr.regression.binary_rss_forest

        # This list well be read out by save_iteration() in the solver
        self.hypers = [num_trees, max_num_nodes, do_bootstrapping,
                       n_points_per_tree, ratio_features, min_samples_split,
                       min_samples_leaf, max_depth, eps_purity, seed]
        self.seed = seed

        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

        # Never use a lower variance than this
        self.var_threshold = 10 ** -5

    def train(self, X, y, **kwargs):
        """Trains the random forest on X and y.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        Y : np.ndarray [n_samples, ]
            The corresponding target values.

        Returns
        -------
        self
        """

        self.X = X
        self.y = y.flatten()

        if self.n_points_per_tree <= 0:
            self.rf_opts.num_data_points_per_tree = self.X.shape[0]
        else:
            self.rf_opts.num_data_points_per_tree = self.n_points_per_tree
        self.rf = pyrfr.regression.binary_rss_forest()
        self.rf.options = self.rf_opts
        data = self.__init_data_container(self.X, self.y)
        self.rf.fit(data, rng=self.rng)
        return self

    def __init_data_container(self, X, y):
        """Fills a pyrfr default data container, s.t. the forest knows categoricals and bounds for continous data
        
        Parameters
        ----------
        X : np.ndarray [n_samples, n_features]
            Input data points
        y : np.ndarray [n_samples, ]
            Corresponding target values

        Returns
        -------
        data: pyrfr.regression.default_data_container
            The filled data container that pyrfr can interpret
        """
        # retrieve the types and the bounds from the ConfigSpace
        data = pyrfr.regression.default_data_container(X.shape[1])

        for i, (mn, mx) in enumerate(self.bounds):
            if np.isnan(mx):
                data.set_type_of_feature(i, mn)
            else:
                data.set_bounds_of_feature(i, mn, mx)

        for row_X, row_y in zip(X, y):
            data.add_data_point(row_X, row_y)
        return data

    def predict(self, X):
        """Predict means and variances for given X.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples, n_features (config + instance
        features)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """
        if len(X.shape) != 2:
            raise ValueError(
                'Expected 2d array, got %dd array!' % len(X.shape))
        if X.shape[1] != self.types.shape[0]:
            raise ValueError('Rows in X should have %d entries but have %d!' %
                             (self.types.shape[0], X.shape[1]))

        means, vars_ = [], []
        for row_X in X:
            mean, var = self.rf.predict_mean_var(row_X)
            means.append(mean)
            vars_.append(var)
        means = np.array(means)
        vars_ = np.array(vars_)

        return means.reshape((-1, 1)), vars_.reshape((-1, 1))

    def predict_marginalized_over_instances(self, X):
        """Predict mean and variance marginalized over all instances.

        Returns the predictive mean and variance marginalised over all
        instances for a set of configurations.

        Parameters
        ----------
        X : np.ndarray of shape = [n_features (config), ]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """

        if self.instance_features is None or \
                len(self.instance_features) == 0:
            mean, var = self.predict(X)
            var[var < self.var_threshold] = self.var_threshold
            var[np.isnan(var)] = self.var_threshold
            return mean, var
        else:
            n_instance_features = self.instance_features.shape[1]
            n_instances = len(self.instance_features)

        if len(X.shape) != 2:
            raise ValueError(
                'Expected 2d array, got %dd array!' % len(X.shape))
        if X.shape[1] != self.types.shape[0] - n_instance_features:
            raise ValueError('Rows in X should have %d entries but have %d!' %
                             (self.types.shape[0] - n_instance_features,
                              X.shape[1]))

        mean = np.zeros(X.shape[0])
        var = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            X_ = np.hstack(
                (np.tile(x, (n_instances, 1)), self.instance_features))
            means, vars = self.predict(X_)
            # use only mean of variance and not the variance of the mean here
            # since we don't want to reason about the instance hardness distribution
            var_x = np.mean(vars) # + np.var(means)
            if var_x < self.var_threshold:
                var_x = self.var_threshold

            var[i] = var_x
            mean[i] = np.mean(means)

        var[var < self.var_threshold] = self.var_threshold
        var[np.isnan(var)] = self.var_threshold

        if len(mean.shape) == 1:
            mean = mean.reshape((-1, 1))
        if len(var.shape) == 1:
            var = var.reshape((-1, 1))

        return mean, var
