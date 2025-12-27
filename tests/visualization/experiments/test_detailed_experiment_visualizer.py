import numpy as np

from src.visualization.experiments.detailed_experiment_visualizer import DetailedExperimentVisualizer
from experiments.pain_models import NORMAL_PAIN_PROBABILITIES


def get_dummy_params_list():
    return {
        "environment_params": {
            "size": 7,
            "possible_starting_positions": [[0, 6], [1, 6], [0, 5], [1, 5]],
            "possible_initial_food_positions": [[5, 0], [6, 0], [5, 1], [6, 1]],
            "obstacle_positions": [[3, 2], [3, 3], [3, 4], [2, 3], [4, 3], [3, 0], [3, 6]],
            "lifetime_learning": False,
            "stationary": False,
            "food_location_change_steps": 1250
        },
        "agent_params": {
            "w1": 0.7,
            "w2": 0.1,
            "w3": 0.2,
            "w4": 0.1,
            "pain_model": NORMAL_PAIN_PROBABILITIES,
            "aspiration_level": 0.0,
            "alpha": 0.3,
            "gamma": 0.99,
            "epsilon": 0.01}
    }

def test_experiment_name_generation_1():
    params = get_dummy_params_list()
    visualizer = DetailedExperimentVisualizer(params, np.array([1, 2]), "str")
    experiment_name = visualizer.get_experiment_name()

    assert experiment_name == "Objective + Expect + Compare + Normal Pain"

def test_experiment_name_generation_2():
    params = get_dummy_params_list()
    params["agent_params"]["w3"] = 0.0
    visualizer = DetailedExperimentVisualizer(params, np.array([1, 2]), "str")
    experiment_name = visualizer.get_experiment_name()

    assert experiment_name == "Objective + Expect + Normal Pain"

def test_experiment_name_generation_3():
    params = get_dummy_params_list()
    params["agent_params"]["w2"] = 0.0
    params["agent_params"]["w4"] = 0.0
    visualizer = DetailedExperimentVisualizer(params, np.array([1, 2]), "str")
    experiment_name = visualizer.get_experiment_name()

    assert experiment_name == "Objective + Compare"

def test_environment_name_generation_1():
    params = get_dummy_params_list()
    visualizer = DetailedExperimentVisualizer(params, np.array([1, 2]), "str")
    env_name = visualizer.get_environment_name()

    assert env_name == "Non Stationary"

def test_environment_name_generation_2():
    params = get_dummy_params_list()
    params["environment_params"]["size"] = 13
    params["environment_params"]["stationary"] = True
    params["environment_params"]["lifetime_learning"] = True
    visualizer = DetailedExperimentVisualizer(params, np.array([1, 2]), "str")
    env_name = visualizer.get_environment_name()

    assert env_name == "Sparse Lifetime Stationary"



