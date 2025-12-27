import os
import json
from typing import List, Tuple
import pandas as pd

from src.reinforcement_learning.experiments.basic_experiment_runner import BasicExperimentRunner


def _load_json_folder(folder_path: str) -> List[dict]:
    json_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                json_list.append(data)

    return json_list


class MultiExperimentRunner:
    def __init__(self, experiment_folder_path: str):
        self.experiment_folder_path = experiment_folder_path
        self.parameter_maps = _load_json_folder(experiment_folder_path)
        self.basic_runners = [BasicExperimentRunner(parameter_map) for parameter_map in self.parameter_maps]

    def run_all(self) -> Tuple[List[str], List[Tuple[float, float]]]:
        names = [
            pm["experiment_params"]["experiment_name"]
            for pm in self.parameter_maps
        ]
        results = [
            runner.run_experiment()
            for runner in self.basic_runners
        ]

        self.save_results_to_csv(names, results)

        return names, results

    def save_results_to_csv(self, names: List[str], results: List[Tuple[float, float]]):
        folder_name = os.path.basename(os.path.normpath(self.experiment_folder_path))
        results_folder = os.path.join(self.experiment_folder_path, "results")
        os.makedirs(results_folder, exist_ok=True)

        filename = f"{folder_name}_results.csv"
        file_path = os.path.join(results_folder, filename)

        data = {
            "Experiment Name": names,
            "Mean": [result[0] for result in results],
            "Standard Deviation": [result[1] for result in results]
        }

        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

        print(f"Results saved to {file_path}")

