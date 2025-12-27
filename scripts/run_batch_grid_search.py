import sys
import os
import json

from src.reinforcement_learning.experiments.grid_search import run_batch, generate_param_combinations, generate_configs

# Script that is executed by pbs jobs to run the grid search on the batches

start = int(sys.argv[1])
end = int(sys.argv[2])
experiment_folder_path = sys.argv[3]
results_file = sys.argv[4]

root_folder_path = os.path.dirname(os.path.normpath(experiment_folder_path)) + "/"

with open(experiment_folder_path + "base_config.json") as file:
    base_config = json.load(file)

with open(experiment_folder_path + "param_grid.json") as file:
    param_grid = json.load(file)

combinations = generate_param_combinations(param_grid)
total = len(combinations)

if end is None or end > total:
    end = total

combinations = combinations[start:end]
config_combinations = generate_configs(base_config, combinations)

run_batch(config_combinations, results_file)