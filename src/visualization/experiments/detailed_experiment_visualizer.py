import os

import numpy as np
import matplotlib.pylab as plt

from experiments.plotting_variables import *
from experiments.pain_models import NORMAL_PAIN_PROBABILITIES, CHRONIC_PAIN_PROBABILITIES
normal_pain_model = NORMAL_PAIN_PROBABILITIES
chronic_pain_model = CHRONIC_PAIN_PROBABILITIES

class DetailedExperimentVisualizer:

    def __init__(self, param_map, results: np.ndarray, save_directory: str):

        self.params = param_map
        self.experiment_name = self._generate_experiment_name()
        self.environment_name = self._generate_environment_name()
        self.results = results
        self.save_dir = save_directory

    def _generate_experiment_name(self):
        agent_params = self.params["agent_params"]
        names_list = []
        if agent_params["w1"] > 0:
            names_list.append("Objective")
        if agent_params["w2"] > 0:
            names_list.append("Expect")
        if agent_params["w3"] > 0:
            names_list.append("Compare")
        if agent_params["w4"] > 0:
            if agent_params["pain_model"] == normal_pain_model:
                names_list.append("Normal Pain")
            elif agent_params["pain_model"] == chronic_pain_model:
                names_list.append("Chronic Pain")
            else:
                names_list.append("Unknown Pain Model")

        if not names_list:
            return "Experiment without valid weights"
        else:
            name = " + ".join(names_list)
            return name

    def _generate_environment_name(self):
        env_params = self.params["environment_params"]
        names_list = []
        if env_params["size"] > 7:
            names_list.append("Sparse")
        if env_params["lifetime_learning"]:
            names_list.append("Lifetime")
        if env_params["stationary"]:
            names_list.append("Stationary")
        else:
            names_list.append("Non Stationary")
        name = " ".join(names_list)
        return name

    def get_experiment_name(self):
        return self.experiment_name

    def get_environment_name(self):
        return self.environment_name

    def change_param_map_and_results(self, param_map, results: np.ndarray):
        self.params = param_map
        self.experiment_name = self._generate_experiment_name()
        self.environment_name = self._generate_environment_name()
        self.results = results

    def change_save_directory(self, save_directory: str):
        self.save_dir = save_directory

    def save_results_plot(self):
        """
        Saves a matplotlib plot of the mean and std deviation over time for each of the 4 raw metrics
        (excluding Objective Pain) and 3 cumulative metrics (objective reward, subjective reward, happiness).
        """
        assert self.results.ndim == 3
        num_experiments, num_metrics, num_steps = self.results.shape

        metric_names = [
            "Objective reward",
            "Well-being",
            "Happiness",
            "Subjective pain",
            "Cum. obj. reward",
            "Cum. well-being",
            "Cum. happiness"
        ]

        # Compute cumulative metrics
        cumulative_objective = np.cumsum(self.results[:, 0, :], axis=1)
        cumulative_subjective = np.cumsum(self.results[:, 1, :], axis=1)
        cumulative_happiness = np.cumsum(self.results[:, 2, :], axis=1)

        # Drop Objective Pain (index 3)
        filtered_results = np.concatenate([
            self.results[:, :3, :],  # Objective Reward, Subjective Reward, Happiness
            self.results[:, 4:5, :],  # Subjective Pain (skip Objective Pain at index 3)
            cumulative_objective[:, np.newaxis, :],
            cumulative_subjective[:, np.newaxis, :],
            cumulative_happiness[:, np.newaxis, :]
        ], axis=1)  # Shape: (num_trials, 7, num_steps)

        final_cum_obj_reward = cumulative_objective[:, -1]
        mean_cum_obj_reward = final_cum_obj_reward.mean()
        std_cum_obj_reward = final_cum_obj_reward.std()

        time = np.arange(num_steps)
        fig, axes = plt.subplots(7, 1, figsize=(12, 16), sharex=True)
        fig.suptitle(self.experiment_name, fontsize=16)

        agent_params = self.params["agent_params"]
        description_lines = [
            f"Environment: {self.environment_name}",
            f"Trials: {self.results.shape[0]}, Steps: {self.results.shape[2]}",
            f"w1: {agent_params['w1']}, w2: {agent_params['w2']}, w3: {agent_params['w3']}, w4: {agent_params['w4']}"
        ]

        if agent_params["w3"] > 0:
            description_lines.append(f"Aspiration Level: {agent_params['aspiration_level']}")

        description_lines.append(
            f"Gamma: {agent_params['gamma']}, Epsilon: {agent_params['epsilon']}, Alpha: {agent_params['alpha']}")
        description_lines.append(
            f"Final Cumulative Objective Reward: {mean_cum_obj_reward:.2f} ± {std_cum_obj_reward:.2f}")

        description_text = "\n".join(description_lines)
        fig.text(0.01, 0.93, description_text, ha='left', va='bottom', fontsize=10)

        for i in range(7):
            mean = filtered_results[:, i, :].mean(axis=0)
            std = filtered_results[:, i, :].std(axis=0)

            ax = axes[i]
            ax.plot(time, mean, label=f"Mean {metric_names[i]}", color='blue')
            ax.fill_between(time, mean - std, mean + std, color='blue', alpha=0.2, label="±1 Std Dev")
            ax.set_ylabel(metric_names[i])
            ax.legend(loc="upper left")
            ax.grid(True)

        axes[-1].set_xlabel("Time Steps")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        os.makedirs(self.save_dir, exist_ok=True)
        filename = "summary_" + self.experiment_name.replace(" + ", "_") + ".png"
        filename = filename.replace(" ", "_")
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        plt.close()

    def _plot_single_metric(self, metric_data, metric_name, filename):
        """
        Internal helper to plot a single metric with mean ± std shading.
        """
        assert metric_data.ndim == 2, "metric_data must be (num_trials, num_steps)"

        time = np.arange(metric_data.shape[1])
        mean = metric_data.mean(axis=0)
        std = metric_data.std(axis=0)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time, mean, color='blue', label=f"Mean {metric_name}")
        ax.fill_between(time, mean - std, mean + std, color='blue', alpha=0.2, label="±1 Std Dev")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{self.experiment_name}")
        ax.legend(loc="upper left")
        ax.grid(True)

        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_objective_reward(self, file_prefix):
        self._plot_single_metric(self.results[:, 0, :], "Objective reward", file_prefix + "_" + "objective_reward.png")

    def plot_subjective_reward(self, file_prefix):
        self._plot_single_metric(self.results[:, 1, :], "Well-being", file_prefix + "_" + "subjective_reward.png")

    def plot_happiness(self, file_prefix):
        self._plot_single_metric(self.results[:, 2, :], "Happiness", file_prefix + "_" + "happiness.png")

    def plot_subjective_pain(self, file_prefix):
        self._plot_single_metric(self.results[:, 4, :], "Subjective pain", file_prefix + "_" + "subjective_pain.png")

    def plot_cumulative_objective_reward(self, file_prefix):
        cum = np.cumsum(self.results[:, 0, :], axis=1)
        self._plot_single_metric(cum, "Cum. obj. reward", file_prefix + "_" + "cumulative_objective_reward.png")

    def plot_cumulative_subjective_reward(self, file_prefix):
        cum = np.cumsum(self.results[:, 1, :], axis=1)
        self._plot_single_metric(cum, "Cum. well-being", file_prefix + "_" + "cumulative_subjective_reward.png")

    def plot_cumulative_happiness(self, file_prefix):
        cum = np.cumsum(self.results[:, 2, :], axis=1)
        self._plot_single_metric(cum, "Cum. happiness", file_prefix + "_" + "cumulative_happiness.png")

    def save_selected_metrics_plot(self, metrics, filename=None, title=None):
        """
        Plot only the requested metrics (in the given order) with mean ± std shading.
        Uses consistent fonts and pain colors.
        """
        assert self.results is not None and self.results.ndim == 3, "Call run_experiment() first."
        num_trials, _, num_steps = self.results.shape
        time = np.arange(num_steps)

        def _cum(idx):
            return np.cumsum(self.results[:, idx, :], axis=1)

        metric_sources = {
            "Objective reward": lambda: self.results[:, 0, :],
            "Well-being": lambda: self.results[:, 1, :],
            "Happiness": lambda: self.results[:, 2, :],
            "Objective pain": lambda: self.results[:, 3, :],
            "Subjective pain": lambda: self.results[:, 4, :],
            "Cum. obj. reward": lambda: _cum(0),
            "Cum. well-being": lambda: _cum(1),
            "Cum. happiness": lambda: _cum(2),
        }

        for m in metrics:
            if m not in metric_sources:
                raise ValueError(f"Unknown metric '{m}'. Valid keys are: {list(metric_sources.keys())}")


        height_per_subplot = 2.2  # inches per subplot (constant)
        fig_height = height_per_subplot * len(metrics)
        fig_width = 12
        fig, axes = plt.subplots(len(metrics), 1, figsize=(fig_width, fig_height), sharex=True)
        if len(metrics) == 1:
            axes = [axes]

        fig.suptitle(title if title is not None else self.experiment_name,
                     fontsize=title_font_size)

        agent_params = self.params["agent_params"]
        pain_model_transition = agent_params["pain_model"]["transition_probabilities"]
        cumulative_objective = np.cumsum(self.results[:, 0, :], axis=1)
        final_cum_obj_reward = cumulative_objective[:, -1]
        mean_cum_obj_reward = final_cum_obj_reward.mean()
        std_cum_obj_reward = final_cum_obj_reward.std()

        plot_color = chronic_pain_color if pain_model_transition == chronic_pain_model["transition_probabilities"] else normal_pain_color

        description_lines = [
            f"Environment: {self.environment_name}, Trials: {self.results.shape[0]}, Steps: {self.results.shape[2]}",
            f"w1: {agent_params['w1']}, w2: {agent_params['w2']}, w3: {agent_params['w3']}, w4: {agent_params['w4']}",
        ]
        if agent_params["w3"] > 0:
            description_lines[1] = description_lines[1] + f", Aspiration Level: {agent_params['aspiration_level']}"
        description_lines.append(f"Gamma: {agent_params['gamma']}, Epsilon: {agent_params['epsilon']}, Alpha: {agent_params['alpha']}")
        description_lines.append(
            f"Final cumulative objective reward: {mean_cum_obj_reward:.2f} ± {std_cum_obj_reward:.2f}"
        )
        description_text = "\n".join(description_lines)
        fig.text(0.01, 0.92, description_text, ha='left', va='bottom', fontsize=caption_font_size)

        for ax, metric_name in zip(axes, metrics):
            data = metric_sources[metric_name]()  # (num_trials, num_steps)
            mean = data.mean(axis=0)
            std = data.std(axis=0)

            ax.plot(time, mean, label=f"Mean {"subjective reward" if metric_name == "Well-being" else metric_name.lower()}", color=plot_color)
            ax.fill_between(time, mean - std, mean + std, color=plot_color, alpha=0.2,
                            label="± 1 SD")

            ax.set_ylabel(metric_name, fontsize=axis_labels_font_size)
            ax.tick_params(axis='both', labelsize=axis_labels_font_size)
            ax.legend(loc="lower right", fontsize=label_font_size)
            ax.grid(True)

        axes[-1].set_xlabel("Time steps", fontsize=axis_labels_font_size)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        os.makedirs(self.save_dir, exist_ok=True)
        if filename is None:
            safe_metrics = "_".join(
                m.replace(" ", "").replace(".", "").replace("(", "").replace(")", "")
                .replace("+", "plus").replace("−", "-").replace("/", "_")
                for m in metrics
            )
            filename = f"summary_selected_{safe_metrics}.png"
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
