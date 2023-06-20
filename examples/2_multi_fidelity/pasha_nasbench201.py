"""
Stochastic Gradient Descent On Multiple Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example for optimizing a Multi-Layer Perceptron (MLP) across multiple (dataset) instances.

Alternative to budgets, here wlog. we consider instances as a fidelity type. An instance represents a specific
scenario/condition (e.g. different datasets, subsets, transformations) for the algorithm to run. SMAC then returns the
algorithm that had the best performance across all the instances. In this case, an instance is a binary dataset i.e.,
digit-2 vs digit-3.

If we use instance as our fidelity, we need to initialize scenario with argument instance. In this case the argument
budget is no longer required by the target function. But due to the scenario instance argument,
the target function now is required to have an instance argument.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from smac import MultiFidelityFacade as MFFacade
from smac import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

from smac.intensifier.pasha import PASHA
from nas_201_api import NASBench201API as API

if __name__ == "__main__":
    api = API('NAS-Bench-201-v1_1-096897.pth', verbose=False)

    configSpace = ConfigurationSpace({"A": (0, len(api)-1)})
    api.show(1)
    def train(config: Configuration, seed: int = 0):
        #classifier = SVC(C=configSpace["C"], random_state=seed)
        #scores = cross_val_score(classifier, iris.data, iris.target, cv=5)
        scores = cross_val_score
        return 1 - np.mean(scores)

    scenario = Scenario(
        configSpace,
        walltime_limit=30,  # We want to optimize for 30 seconds
        n_trials=5000,  # We want to try max 5000 different trials
        min_budget=1,  # Use min one instance
        max_budget=45,  # Use max 45 instances (if we have a lot of instances we could constraint it here)
    )
    intesifier = PASHA(scenario, incumbent_selection="highest_budget")
    # Create our SMAC object and pass the scenario and the train method
    smac = MFFacade(
        scenario,
        model.train,
        intensifier=intesifier,
        overwrite=True,
    )

    # Now we start the optimization process
    incumbent = smac.optimize()

    default_cost = smac.validate(model.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")
