import json
import math

from src.reinforcement_learning.experiments.basic_experiment_runner import BasicExperimentRunner

def test_runner_construction():
    with open("resources/param_map.json") as file:
        param_map = json.load(file)

    BasicExperimentRunner(param_map)

def test_runner():
    with open("resources/param_map_binary_pain_stationary.json") as file:
        param_map = json.load(file)

    runner = BasicExperimentRunner(param_map)
    mean, std = runner.run_experiment()
    assert math.isclose(mean, 2300.62, abs_tol=0.01)
    assert math.isclose(std, 59.65, abs_tol=0.01)


