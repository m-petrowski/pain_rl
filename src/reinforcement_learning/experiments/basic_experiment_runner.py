from joblib import Parallel, delayed

import copy
import statistics
import importlib
import random
from typing import Tuple

class BasicExperimentRunner:
    def __init__(self, parameter_map: dict):
        self.parameter_map = copy.deepcopy(parameter_map)

        experiment_params = self.parameter_map["experiment_params"]
        self.experiment_name = experiment_params["experiment_name"]
        self.num_experiments = experiment_params["num_experiments"]
        self.num_steps_per_experiment = experiment_params["num_steps_per_experiment"]
        self.random_seed = experiment_params["random_seed"]
        random.seed(self.random_seed)

        module_path, agent_class_name = self.parameter_map["agent_class"].rsplit('.', 1)
        module = importlib.import_module(module_path)
        self.agent_class = getattr(module, agent_class_name)

    def _run_trials_in_parallel(self):
        rewards = Parallel(n_jobs=-1)( #Use all available CPU cores
            delayed(self._run_trial)(
                self.agent_class,
                self.parameter_map,
                self.num_steps_per_experiment,
                self.random_seed,
                i
            )
            for i in range(self.num_experiments)
        )
        return rewards

    def run_experiment(self):
        rewards = self._run_trials_in_parallel()
        mean = statistics.mean(rewards)
        std = statistics.stdev(rewards)
        return mean, std

    def run_experiments_with_all_episode_results(self):
        rewards = self._run_trials_in_parallel()
        return rewards

    @staticmethod
    def _calculate_results_for_trial(agent, num_steps):
        cumulative_reward = 0
        for _ in range(num_steps):
            _, _, _, reward, _ = agent.act_one_step()
            cumulative_reward += reward
        return cumulative_reward

    def _run_trial(self, agent_class, parameter_map, num_steps: int, random_seed: int, seed_offset: int):
        random.seed(random_seed + seed_offset)

        agent = agent_class.from_param_map(copy.deepcopy(parameter_map))

        return self._calculate_results_for_trial(agent, num_steps)
