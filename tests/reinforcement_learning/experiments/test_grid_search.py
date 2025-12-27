import json

from src.reinforcement_learning.experiments.grid_search import generate_param_combinations, generate_config

with open("resources/param_map_binary_pain_is_0.json") as file:
    base_config = json.load(file)

def test_param_combinations_config_generation():

    param_grid = {
        "agent_params.w1": [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
        "agent_params.w2": [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
        "agent_params.w3": [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
        "agent_params.w4": [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
        "agent_params.aspiration_level": [0.1, 0.3, 0.7, 0.9, 1],
        "agent_params.alpha": [0.1, 0.3, 0.7, 0.9],
        "agent_params.epsilon": [0.01, 0.1],
        "agent_params.pain_model": ["normal", "chronic"]
    }
    combos = generate_param_combinations(param_grid)
    generate_config(base_config, combos[0])
