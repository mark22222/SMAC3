"""
Progressive Asynchronous Successive Halving Algorithm (PASHA) On Image Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example for finding model (hyperparameter) configurations yielding high accuracy using PASHA,
evaluated seperately on three NASBench201 datasets (CIFAR-10, CIFAR-100, ImageNet16-120).

The NATS-Bench library and benchmark files were used in order to benchmark these NAS algorithms in Python.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from sklearn import datasets
import pandas as pd


from smac import MultiFidelityFacade as MFFacade
from smac import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

from smac.intensifier.pasha import PASHA
from nats_bench import create

if __name__ == "__main__":

    # Prepare seeds for PASHA to be evaluated on different SMAC configurations.
    seeds = [i for i in range(10)]
    datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
    # Create an API for the Topology Search Space (TSS) using the corresponding benchmark.
    api = create(str(Path("..\\..\\NATS-tss-v1_0-3ffb9-simple").resolve()),
                 'tss',
                 fast_mode=True,
                 verbose=False)

    configSpace = ConfigurationSpace({"A": (0, len(api)-1)})

    experiment_dataframe = pd.DataFrame(columns=["Dataset", "Seed", "Incumbent", "Incumbent Cost", "Runtime", "Default Cost"])
    for seed in seeds:
        for dataset in datasets:
            time_tracker = []
            def train(config: Configuration, seed: int = 0, budget: int = 12):
                #classifier = SVC(C=configSpace["C"], random_state=seed)
                #scores = cross_val_score(classifier, iris.data, iris.target, cv=5)
                validation_accuracy, latency, time_cost, current_total_time_cost = api.simulate_train_eval(config["A"],
                                                                                                           dataset=dataset,
                                                                                                           iepoch=budget - 1,
                                                                                                           hp="200")
                time_tracker.append(time_cost)
                return 100 - validation_accuracy



            scenario = Scenario(
                configSpace,
                walltime_limit=30,  # We want to optimize for 30 seconds
                n_trials=5000,  # We want to try max 5000 different trials
                min_budget=1,  # Use min one instance
                max_budget=200,  # Use max 45 instances (if we have a lot of instances we could constraint it here)
                seed=seed
            )
            intesifier = PASHA(scenario, eta=2, incumbent_selection="highest_budget")
            # Create our SMAC object and pass the scenario and the train method
            smac = MFFacade(
                scenario,
                train,
                intensifier=intesifier,
                overwrite=True,

            )

            # Now we start the optimization process
            incumbent = smac.optimize()

            default_cost = smac.validate(configSpace.get_default_configuration())
            print(f"Default cost: {default_cost}")

            incumbent_cost = smac.validate(incumbent)
            print(f"Incumbent cost: {incumbent_cost}")

            print(time_tracker)
            print(sum(time_tracker))

            experiment_dataframe.loc[len(experiment_dataframe)] = {'Dataset': dataset, 'Seed': seed, 'Incumbent': incumbent['A'], 'Incumbent Cost': incumbent_cost, 'Runtime': sum(time_tracker), 'Default Cost': default_cost}

    experiment_dataframe.to_csv(Path("./experiment_results.csv").resolve(), index=False)
