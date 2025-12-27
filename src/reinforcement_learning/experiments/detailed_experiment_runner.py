import numpy as np
from src.reinforcement_learning.experiments.basic_experiment_runner import BasicExperimentRunner


class DetailedExperimentRunner(BasicExperimentRunner):

    def __init__(self, config):
        super().__init__(config)
        self._last_results = None

    @staticmethod
    def _calculate_results_for_trial(agent, num_steps):
        objective_rewards = []
        subjective_rewards = []
        happiness = []
        obj_pain = []
        subj_pain = []

        for step in range(num_steps):
            _, _, _, cur_obj_reward, cur_subj_reward = agent.act_one_step()
            cur_obj_pain = agent.get_current_pain()
            cur_subj_pain = agent.get_subjective_pain()
            cur_happiness = cur_subj_reward + cur_subj_pain
            objective_rewards.append(cur_obj_reward)
            subjective_rewards.append(cur_subj_reward)
            happiness.append(cur_happiness)
            obj_pain.append(cur_obj_pain)
            subj_pain.append(cur_subj_pain)

        results = np.array([
            objective_rewards,
            subjective_rewards,
            happiness,
            obj_pain,
            subj_pain
        ], dtype=np.float32) #Shape (num_metrics, num_steps) First dimension is the metric, second is the time step

        return results

    def run_experiment(self):
        results = self._run_trials_in_parallel()
        results = np.stack(results, axis=0) #Shape (num_experiments, 5, num_steps) First dimension is the trial, second the metric, third is the time step
        self._last_results = results
        return results

    def get_cumulative_objective_stats(self):
        assert self._last_results is not None, "run_experiment() must be called first."
        cumulative = np.cumsum(self._last_results[:, 0, :], axis=1)
        final_values = cumulative[:, -1]
        return final_values.mean(), final_values.std()

    def get_cumulative_subjective_stats(self):
        assert self._last_results is not None, "run_experiment() must be called first."
        cumulative = np.cumsum(self._last_results[:, 1, :], axis=1)
        final_values = cumulative[:, -1]
        return final_values.mean(), final_values.std()



