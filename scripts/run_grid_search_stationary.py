import json

from experiments.pain_models import NORMAL_PAIN_PROBABILITIES, CHRONIC_PAIN_PROBABILITIES
from src.reinforcement_learning.experiments.grid_search import GridSearchRunner

#Stationary
root_folder_path = "../experiments/grid_search/"
experiment_folder_path = root_folder_path + "stationary/"
with open(experiment_folder_path + "base_config.json") as file:
    base_config = json.load(file)

pain_models = [{"id": "chronic", "model": CHRONIC_PAIN_PROBABILITIES}, {"id": "normal", "model": NORMAL_PAIN_PROBABILITIES}]


param_grid = {
    "agent_params.w1": [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
    "agent_params.w2": [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
    "agent_params.w3": [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
    "agent_params.w4": [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
    "agent_params.aspiration_level": [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
    "agent_params.pain_model": pain_models
}

grid_search = GridSearchRunner(base_config=base_config, param_grid=param_grid, config_save_dir=experiment_folder_path + "configs/", results_file=experiment_folder_path + "results_stationary.csv")
grid_search.run()