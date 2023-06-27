from __future__ import annotations

from typing import Any, Iterator

import math
from collections import defaultdict
import json
import numpy as np
from ConfigSpace import Configuration

from smac.constants import MAXINT
from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.runhistory import TrialInfo, TrialKey
from smac.runhistory.dataclasses import InstanceSeedBudgetKey, StatusType
from smac.runhistory.errors import NotEvaluatedError
from smac.scenario import Scenario
from smac.utils.configspace import get_config_hash
from smac.utils.data_structures import batch
from smac.utils.logging import get_logger
from smac.utils.pareto_front import calculate_pareto_front, sort_by_crowding_distance

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class PASHA(AbstractIntensifier):
    """
    Implementation of Progressive Asynchronous Successive Halving Algorithm (PASHA).
    This implementation alters and extends Successive Halving, which PASHA is based on.
    Internally, a tracker keeps track of configurations and their stage (rung).

    The behaviour of this intensifier is as follows:

    First, add random configurations (configs) to the lowest rung, rung 0. Trials evaluate these configs given a certain budget which
    is dependent on the rung, a higher rung meaning a higher budget. Configs are ranked by the cost returned by the trials.
    As soon as there are eta (the reduction factor) configs in a rung, a promotion to the next rung can be made for the highest-ranked config.
    For any rung i, rung i+1 then contains 1/eta configs from rung i, for which new runs/Trials are then started using a higher budget etc.
    If we are running out of promotable configs, we simply add more configs to the lowest rung.
    We stop if the ranking of configs in the two highest rungs has become consistent.

    Note
    ----
    The implementation natively supports brackets from Hyperband. However, in the case of PASHA,
    only one bracket is used.

    Parameters
    ----------
    eta : int, defaults to 3
        The reduction factor. It controls the proportion of configurations kept going from one rung to the next
    n_seeds : int, defaults to 1
        How many seeds to use for each instance.
    instance_seed_order : str, defaults to "shuffle_once"
        How to order the instance-seed pairs. Can be set to:

        - `None`: No shuffling at all and use the instance-seed order provided by the user.
        - `shuffle_once`: Shuffle the instance-seed keys once and use the same order across all runs.
        - `shuffle`: Shuffles the instance-seed keys for each bracket individually.
    incumbent_selection : str, defaults to "highest_observed_budget"
        How to select the incumbent when using budgets. Can be set to:

        - `any_budget`: Incumbent is the best on any budget i.e., best performance regardless of budget.
        - `highest_observed_budget`: Incumbent is the best in the highest budget run so far.
        - `highest_budget`: Incumbent is selected only based on the highest budget.
    seed : int, defaults to None
        Internal seed used for random events like shuffle seeds.
    """

    def __init__(
            self,
            scenario: Scenario,
            eta: int = 3,
            n_seeds: int = 1,
            instance_seed_order: str | None = "shuffle_once",
            incumbent_selection: str = "highest_observed_budget",
            seed: int | None = None,
    ):
        super().__init__(
            scenario=scenario,
            n_seeds=n_seeds,
            seed=seed,
        )

        self._eta = eta
        self._instance_seed_order = instance_seed_order
        self._incumbent_selection = incumbent_selection
        self._highest_observed_budget_only = False if incumbent_selection == "any_budget" else True

        # Global variables derived from scenario
        self._min_budget = self._scenario.min_budget
        self._max_budget = self._scenario.max_budget

        # Tracking the current maximum resources
        self._Rt = eta * self._min_budget
        # Tracking the current maximum rung
        self._Kt = math.floor(math.log(self._Rt / self._min_budget, eta))
        # Tracking the rung index
        self._t = 0

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "eta": self._eta,
                "instance_seed_order": self._instance_seed_order,
                "incumbent_selection": self._incumbent_selection,
            }
        )

        return meta

    def reset(self) -> None:
        """Reset the internal variables of the intensifier including the tracker."""
        super().reset()

        # This tracker records all (rung, (config, trial)) key-value pairs.
        self._tracker: dict[int, dict[Configuration, TrialInfo]] = defaultdict(dict)

    def __post_init__(self) -> None:
        """Post initialization steps after the runhistory has been set."""
        super().__post_init__()

        # We generate our instance seed pairs once
        is_keys = self.get_instance_seed_keys_of_interest()

        # Budgets, followed by lots of sanity-checking
        eta = self._eta
        min_budget = self._min_budget
        max_budget = self._max_budget

        if max_budget is not None and min_budget is not None and max_budget < min_budget:
            raise ValueError("Max budget has to be larger than min budget.")

        if self.uses_instances:
            if isinstance(min_budget, float) or isinstance(max_budget, float):
                raise ValueError("PASHA requires integer budgets when using instances.")

            min_budget = min_budget if min_budget is not None else 1
            max_budget = max_budget if max_budget is not None else len(is_keys)

            if max_budget > len(is_keys):
                raise ValueError(
                    f"Max budget of {max_budget} can not be greater than the number of instance-seed "
                    f"keys ({len(is_keys)})."
                )

            if max_budget < len(is_keys):
                logger.warning(
                    f"Max budget {max_budget} does not include all instance seed  " f"pairs ({len(is_keys)})."
                )
        else:
            if min_budget is None or max_budget is None:
                raise ValueError(
                    "PASHA requires the parameters min_budget and max_budget defined in the scenario."
                )

            if len(is_keys) != 1:
                raise ValueError("PASHA supports only one seed when using budgets.")

        if min_budget is None or min_budget <= 0:
            raise ValueError("Min budget has to be larger than 0.")

        budget_type = "INSTANCES" if self.uses_instances else "BUDGETS"
        logger.info(
            f"PASHA uses budget type {budget_type} with eta {eta}, "
            f"min budget {min_budget}, and max budget {max_budget}."
        )

        # Pre-computing variables
        max_iter = self._get_max_iterations(eta, max_budget, min_budget)
        budgets, n_configs = self._compute_configs_and_budgets_for_stages(eta, max_budget, max_iter)

        # Global variables
        self._min_budget = min_budget
        self._max_budget = max_budget

        # Stage variables, depending on the bracket (0 is the bracket here since PASHA only has one bracket)
        self._max_iterations: dict[int, int] = {0: max_iter + 1}
        self._n_configs_in_stage: dict[int, list] = {0: n_configs}
        self._budgets_in_stage: dict[int, list] = {0: budgets}

    @staticmethod
    def _get_max_iterations(eta: int, max_budget: float | int, min_budget: float | int) -> int:
        return int(np.floor(np.log(max_budget / min_budget) / np.log(eta)))

    @staticmethod
    def _compute_configs_and_budgets_for_stages(
            eta: int, max_budget: float | int, max_iter: int, s_max: int | None = None
    ) -> tuple[list[int], list[int]]:
        if s_max is None:
            s_max = max_iter

        n_initial_challengers = math.ceil((eta ** max_iter) * (s_max + 1) / (max_iter + 1))

        # How many configs in each stage
        lin_space = -np.linspace(0, max_iter, max_iter + 1)
        n_configs_ = np.floor(n_initial_challengers * np.power(eta, lin_space))
        n_configs = np.array(np.round(n_configs_), dtype=int).tolist()

        # How many budgets in each stage
        lin_space = -np.linspace(max_iter, 0, max_iter + 1)
        budgets = (max_budget * np.power(eta, lin_space)).tolist()

        return budgets, n_configs

    def get_state(self) -> dict[str, Any]:  # noqa: D102
        # Replace config by dict
        tracker: dict[int, dict[str, dict]] = defaultdict(dict)
        for key in list(self._tracker.keys()):
            for config in self._tracker[key]:
                trialinfo_dict = {"config": json.dumps(config.get_dictionary()),
                                  "instance": self._tracker[key][config].instance,
                                  "seed": self._tracker[key][config].seed,
                                  "budget": self._tracker[key][config].budget}
                tracker[key][json.dumps(config.get_dictionary())] = trialinfo_dict

        return {"tracker": tracker, "t": self._t, "K_t": self._Kt, "R_t": self._Rt}

    def set_state(self, state: dict[str, Any]) -> None:  # noqa: D102
        self._tracker: dict[int, dict[Configuration, TrialInfo]] = defaultdict(dict)
        self._t = state["t"]
        self._Kt = state["K_t"]
        self._Rt = state["R_t"]
        tracker = state["tracker"]

        for key in list(tracker.keys()):
            for config_dict in tracker[key]:
                config = Configuration(self._scenario.configspace, json.load(config_dict))

                self._tracker[key][config] = TrialInfo(config=config,
                                                       instance=tracker[key][config]["instance"],
                                                       seed=tracker[key][config]["seed"],
                                                       budget=tracker[key][config]["budget"])

    @property
    def uses_seeds(self) -> bool:  # noqa: D102
        return True

    @property
    def uses_budgets(self) -> bool:  # noqa: D102
        if self._scenario.instances is None:
            return True

        return False

    @property
    def uses_instances(self) -> bool:  # noqa: D102
        if self._scenario.instances is None:
            return False

        return True

    def print_tracker(self) -> None:
        """Prints the number of configurations in each bracket/stage."""
        messages = []
        for rung in self._tracker.items():
            messages.append(f"--- Rung {rung[0]}: {len(self._tracker[rung[0]].keys())} Configs")

        if len(messages) > 0:
            logger.debug(f"{self.__class__.__name__} statistics:")

        for message in messages:
            logger.info(message)

    def get_trials_of_interest(
            self,
            config: Configuration,
            *,
            validate: bool = False,
            seed: int | None = None,
    ) -> list[TrialInfo]:  # noqa: D102
        is_keys = self.get_instance_seed_keys_of_interest(validate=validate, seed=seed)
        budget = None

        # When we use budgets, we always evaluated on the highest budget only
        if self.uses_budgets:
            budget = self._max_budget

        trials = []
        for key in is_keys:
            trials.append(TrialInfo(config=config, instance=key.instance, seed=key.seed, budget=budget))

        return trials

    def get_instance_seed_budget_keys(
            self, config: Configuration, compare: bool = False
    ) -> list[InstanceSeedBudgetKey]:
        """Returns the instance-seed-budget keys for a given configuration. This method supports ``highest_budget``,
        which only returns the instance-seed-budget keys for the highest budget (if specified). In this case, the
        incumbents in ``update_incumbents`` are only changed if the costs on the highest budget are lower.

        Parameters
        ----------
        config: Configuration
            The Configuration to be queried
        compare : bool, defaults to False
            Get rid of the budget information for comparing if the configuration was evaluated on the same
            instance-seed keys.
        """
        isb_keys = []
        for rung in range(self._Kt):
            if config in self._tracker[rung].keys():
                entry = config
                config_id = self.runhistory.get_config_id(entry)
                rung_dict = self._tracker[rung]
                tk = TrialKey(config_id, rung_dict[entry].instance, rung_dict[entry].seed, rung_dict[entry].budget)
                value = self.runhistory[tk]
                isb_keys += [InstanceSeedBudgetKey(self._tracker[rung][config].instance, self._tracker[rung][config].seed, self._tracker[rung][config].budget)]
                break

        if compare:
            # Get rid of duplicates
            isb_keys = list(
                set([InstanceSeedBudgetKey(instance=key.instance, seed=key.seed, budget=None) for key in isb_keys])
            )

        return isb_keys

    def __iter__(self) -> Iterator[TrialInfo]:  # noqa: D102
        self.__post_init__()

        # Log brackets/stages
        logger.info("Number of configs in stage:")
        for bracket, n in self._n_configs_in_stage.items():
            logger.info(f"--- Bracket {bracket}: {n}")

        logger.info("Budgets in stage:")
        for bracket, budgets in self._budgets_in_stage.items():
            logger.info(f"--- Bracket {bracket}: {budgets}")

        rh = self.runhistory

        # We have to add already existing trials from the runhistory
        # Idea: We simply add existing configs to the tracker (first stage) but assign a random instance shuffle seed.
        # In the best case, trials (added from the users) are included in the seed and it has not re-computed again.
        # Note: If the intensifier was restored, we don't want to go in here
        if len(self._tracker) == 0:
            bracket = 0
            stage = 0

            # Print ignored budgets
            ignored_budgets = []
            for k in rh.keys():
                if k.budget not in self._budgets_in_stage[0] and k.budget not in ignored_budgets:
                    ignored_budgets.append(k.budget)

            if len(ignored_budgets) > 0:
                logger.warning(
                    f"Trials with budgets {ignored_budgets} will been ignored. Consider adding trials with budgets "
                    f"{self._budgets_in_stage[0]}."
                )

            # We batch the configs because we need n_configs in each iteration
            # If we don't have n_configs, we sample new ones
            # We take the number of configs from the first bracket and the first stage
            n_configs = self._n_configs_in_stage[bracket][stage]
            for configs in batch(rh.get_configs(), n_configs):
                n_rh_configs = len(configs)

                if len(configs) < n_configs:
                    try:
                        config = next(self.config_generator)
                        configs.append(config)
                    except StopIteration:
                        # We stop if we don't find any configuration anymore
                        return

                seed = self._get_next_order_seed()
                self._tracker[(bracket, stage)].append((seed, configs))
                logger.info(
                    f"Added {n_rh_configs} configs from runhistory and {n_configs - n_rh_configs} new configs to "
                    f"PASHA's first bracket and first stage with order seed {seed}."
                )

        while True:
            
            # Obtain a job - a (config, rung) pair.
            config, rung = self._get_job()
            if config is None:
                # No config was promotable nor were we able to generate a new, random config.
                return
            is_key = self.get_instance_seed_keys_of_interest()[0]
            # The resources (the budget) that is available to the job's config. 
            budget = self._min_budget * (self._eta ** rung)
            if budget > self._max_budget:
                # Cap the budget up to the maximum budget.
                budget = self._max_budget
            # Instantiate a trial. It is then yielded for computation.
            ti = TrialInfo(config, is_key.instance, is_key.seed, budget)

            self._tracker[rung][config] = ti

            yield ti

            # Check if a rung update is possible.
            if self._Rt < self._max_budget:
                config_ranking = self._top_k(self._Kt, len(self._tracker[self._Kt].keys()))
                config_ranking_below = self._top_k(self._Kt - 1, len(self._tracker[self._Kt - 1].keys()))[
                                       :len(config_ranking)]
                for config, config_below in zip(config_ranking, config_ranking_below):
                    if config[0] != config_below[0]:
                        self._t += 1
                        self._Rt = (self._eta ** self._t) * self._eta * self._min_budget
                        self._Kt = math.floor(math.log(self._Rt / self._min_budget, self._eta))
                        print(f"Max Rung: {self._Kt}")
                        print(f"Max Budget: {self._Rt}")
                        break

    def _get_job(self):
        """Iterates over the (non-empty) rungs and returns a promotable config and the rung it will be promoted to
        (a job is a (config, rung) pair), if such a config currently exists.
        A config is promotable iff its rank in its current rung is sufficiently high and if it's not yet present in the rung to be promoted to.
        If there is no such config available, a random config is generated (if possible) for the lowest rung, rung 0.
        """
        for rung in reversed(range(self._Kt)):
            candidates = self._top_k(rung, math.floor(len(self._tracker[rung].keys()) / self._eta))
            for cand in candidates:
                # Across a single rung, all configs within it are unique.
                if cand[0] not in self._tracker[rung + 1]:
                    return cand[0], rung + 1
        try:
            config = next(self.config_generator)
        except StopIteration:
            return None, 0
        return config, 0

    def _top_k(self, rung, amount):
        """Returns the <amount> highest ranked configs for a given rung, sorted by rank.
        """
        # Extract all (config, trial) pairs for this rung.
        rung_dict = self._tracker[rung]
        ranking = []
        for entry in rung_dict:
            config_id = self.runhistory.get_config_id(entry)
            tk = TrialKey(config_id, rung_dict[entry].instance, rung_dict[entry].seed, rung_dict[entry].budget)
            value = self.runhistory[tk]
            # We are interested in trials that have terminated, as they possess a cost to be ranked.
            if value.status == StatusType.SUCCESS:
                ranking.append((entry, value.cost, rung_dict[entry]))
        return sorted(ranking, key=lambda x: x[1])[:amount]

    def _get_instance_seed_budget_keys_by_stage(
            self,
            bracket: int,
            stage: int,
            seed: int | None = None,
    ) -> list[InstanceSeedBudgetKey]:
        """Returns all instance-seed-budget keys (isb keys) for the given stage. Each stage
        is associated with a budget (N). Two possible options:

        1) Instance based: We return N isb keys. If a seed is specified, we shuffle the keys before
        returning the first N instances. The budget is set to None here.
        2) Budget based: We return one isb only but the budget is set to N.
        """
        budget: float | int | None = None
        is_keys = self.get_instance_seed_keys_of_interest()

        # We have to differentiate between budgets and instances based here
        # If we are budget based, we always have one instance seed pair only
        # If we are in the instance setting, we have to return a specific number of instance seed pairs

        if self.uses_instances:
            # Shuffle instance seed pairs group-based
            if seed is not None:
                is_keys = self._reorder_instance_seed_keys(is_keys, seed=seed)

            # We only return the first N instances
            N = int(self._budgets_in_stage[bracket][stage])
            is_keys = is_keys[:N]
        else:
            assert len(is_keys) == 1

            # The stage defines which budget should be used (in real-valued setting)
            # No shuffle is needed here because we only have on instance seed pair
            budget = self._budgets_in_stage[bracket][stage]

        isbk = []
        for isk in is_keys:
            isbk.append(InstanceSeedBudgetKey(instance=isk.instance, seed=isk.seed, budget=budget))

        return isbk

    def _get_next_trials(
            self,
            config: Configuration,
            from_keys: list[InstanceSeedBudgetKey],
    ) -> list[TrialInfo]:
        """Returns trials for a given config from a list of instances (instance-seed-budget keys). The returned trials
        have not run or evaluated yet.
        """
        rh = self.runhistory
        evaluated_trials = rh.get_trials(config, highest_observed_budget_only=False)
        running_trials = rh.get_running_trials(config)

        next_trials: list[TrialInfo] = []
        for instance in from_keys:
            trial = TrialInfo(config=config, instance=instance.instance, seed=instance.seed, budget=instance.budget)

            if trial in evaluated_trials or trial in running_trials:
                continue

            next_trials.append(trial)

        return next_trials

    def _get_best_configs(
            self,
            configs: list[Configuration],
            bracket: int,
            stage: int,
            from_keys: list[InstanceSeedBudgetKey],
    ) -> list[Configuration]:
        """Returns the best configurations. The number of configurations is depending on the stage. Raises
        ``NotEvaluatedError`` if not all trials have been evaluated.
        """
        try:
            n_configs = self._n_configs_in_stage[bracket][stage + 1]
        except IndexError:
            return []

        rh = self.runhistory
        configs = configs.copy()

        for config in configs:
            isb_keys = rh.get_instance_seed_budget_keys(config)
            if not all(isb_key in isb_keys for isb_key in from_keys):
                raise NotEvaluatedError

        selected_configs: list[Configuration] = []
        while len(selected_configs) < n_configs:
            # We calculate the pareto front for the given configs
            # We use the same isb keys for all the configs
            all_keys = [from_keys for _ in configs]
            incumbents = calculate_pareto_front(rh, configs, all_keys)

            # Idea: We recursively calculate the pareto front in every iteration
            for incumbent in incumbents:
                configs.remove(incumbent)
                selected_configs.append(incumbent)

        # If we have more selected configs, we remove the ones with the smallest crowding distance
        if len(selected_configs) > n_configs:
            selected_configs = sort_by_crowding_distance(rh, configs, all_keys)[:n_configs]
            logger.debug("Found more configs than required. Removed configs with smallest crowding distance.")

        return selected_configs

    def _get_next_order_seed(self) -> int | None:
        """Next instances shuffle seed to use."""
        # Here we have the option to shuffle the trials when specified by the user
        if self._instance_seed_order == "shuffle":
            seed = self._rng.randint(0, MAXINT)
        elif self._instance_seed_order == "shuffle_once":
            seed = 0
        else:
            seed = None

        return seed

    def _get_next_bracket(self) -> int:
        """PASHA only uses one bracket. Therefore, we always return 0 here."""
        return 0
