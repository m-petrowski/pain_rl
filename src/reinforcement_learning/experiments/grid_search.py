import itertools
import copy
import os
import uuid

import pandas as pd
import datetime
from typing import Dict, Any, List, Tuple

from experiments.pain_models import NORMAL_PAIN_PROBABILITIES, CHRONIC_PAIN_PROBABILITIES
from src.reinforcement_learning.experiments.basic_experiment_runner import BasicExperimentRunner

pain_model = {
                "normal": NORMAL_PAIN_PROBABILITIES,
                "chronic": CHRONIC_PAIN_PROBABILITIES
                        }


def set_nested(config: Dict[str, Any], key_path: str, value: Any):
    keys = key_path.split(".")
    d = config
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value

def generate_combinations(base_config, param_grid) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    keys, values = zip(*param_grid.items())
    all_combinations = list(itertools.product(*values))

    combinations = []
    for idx, combo in enumerate(all_combinations):
        config = copy.deepcopy(base_config)
        param_record = {}

        for key, value in zip(keys, combo):
            # Special handling for pain_model dictionaries with "id" and "model"
            if key == "agent_params.pain_model" and isinstance(value, dict) and "model" in value and "id" in value:
                set_nested(config, key, value["model"])
                param_record[key] = value["id"]
            else:
                set_nested(config, key, value)
                param_record[key] = value

        experiment_name = f"grid_search_exp_{idx}"
        config["experiment_params"]["experiment_name"] = experiment_name
        param_record["experiment_name"] = experiment_name

        combinations.append((experiment_name, config, param_record))

    return combinations

def generate_param_combinations(param_grid):
    keys, values = zip(*param_grid.items())
    all_combinations = list(itertools.product(*values))
    combo_set = set()
    for combo in all_combinations:
        combo = dict(zip(keys, combo))
        if combo["agent_params.w3"] == 0:
            combo["agent_params.aspiration_level"] = 0
        if combo["agent_params.w4"] == 0:
            combo["agent_params.pain_model"] = "normal"

        combo_tuple = tuple(sorted(combo.items()))
        combo_set.add(combo_tuple)


    return sorted(list(combo_set))

def generate_config(base_config, param_grid_tuple):
    config = copy.deepcopy(base_config)
    param_grid_map = dict(param_grid_tuple)
    pain_model_name = param_grid_map["agent_params.pain_model"]
    param_grid_map["agent_params.pain_model"] = pain_model[param_grid_map["agent_params.pain_model"]]
    for key, value in param_grid_map.items():
        set_nested(config, key, value)

    param_grid_map["agent_params.pain_model"] = pain_model_name
    experiment_name = f"experiment_{uuid.uuid4()}"
    config["experiment_params"]["experiment_name"] = experiment_name
    param_grid_map["experiment_name"] = experiment_name
    return experiment_name, config, param_grid_map

def generate_configs(base_config, param_grid_tuples):
    configs = []
    for param_grid_tuple in param_grid_tuples:
        configs.append(generate_config(base_config, param_grid_tuple))
    return configs

class GridSearchRunner:
    def __init__(
        self,
        base_config: Dict[str, Any],
        param_grid: Dict[str, List[Any]],
        config_save_dir: str = "./configs",
        results_file: str = "grid_search_results.csv",
    ):
        self.base_config = base_config
        self.param_grid = param_grid
        self.config_save_dir = config_save_dir
        self.results_file = results_file
        self.records = []




    def run(self):
        start_time = datetime.datetime.now()
        print("Gridsearch started at", start_time)
        combinations = generate_combinations(self.base_config, self.param_grid)

        # If the results file doesn't exist, write headers first
        first_write = not os.path.exists(self.results_file)

        for idx, (experiment_name, config, param_record) in enumerate(combinations):

            runner = BasicExperimentRunner(config)
            mean, std = runner.run_experiment()

            param_record["mean_result"] = mean
            param_record["std_result"] = std
            self.records.append(param_record)

            print(f"[{idx + 1}/{len(combinations)}] {experiment_name}: Mean={mean:.2f}, Std={std:.2f}")

            result_df = pd.DataFrame([param_record])
            result_df.to_csv(self.results_file, mode='a', header=first_write, index=False)
            first_write = False

        print(f"\n✅ Grid search completed. Results saved to: {self.results_file}")
        print("Gridsearch started at", start_time)
        print("Gridsearch ended at", datetime.datetime.now())


def run_batch(batch_combinations: List[Tuple[str, Dict[str, Any], Dict[str, Any]]], results_file):
    records = []
    start_time = datetime.datetime.now()
    print("Gridsearch started at", start_time)
    first_write = not os.path.exists(results_file)

    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    for idx, (experiment_name, config, param_record) in enumerate(batch_combinations):
        runner = BasicExperimentRunner(config)
        mean, std = runner.run_experiment()

        param_record["mean_result"] = mean
        param_record["std_result"] = std
        records.append(param_record)

        print(f"[{idx + 1}/{len(batch_combinations)}] {experiment_name}: Mean={mean:.2f}, Std={std:.2f}")

        result_df = pd.DataFrame([param_record])
        result_df.to_csv(results_file, mode='a', header=first_write, index=False)
        first_write = False

    print(f"\n✅ Grid search completed. Results saved to: {results_file}")
    print("Gridsearch started at", start_time)
    print("Gridsearch ended at", datetime.datetime.now())
