import pandas as pd
import scipy.stats as stats
import json
import copy
import statistics
import math

from src.reinforcement_learning.experiments.basic_experiment_runner import BasicExperimentRunner
from experiments.pain_models import NORMAL_PAIN_PROBABILITIES, CHRONIC_PAIN_PROBABILITIES

experiment_name = "non_stationary" # change this to change the experiment

experiment_folder_path = "../batch_grid_search/" + experiment_name + "/"
normal_pain = NORMAL_PAIN_PROBABILITIES
chronic_pain = CHRONIC_PAIN_PROBABILITIES


def get_reward_category(row):
    """Defines the reward category based on which weights are active."""
    w1, w2, w3 = row['w1'] > 0, row['w2'] > 0, row['w3'] > 0
    if w1 and w2 and w3: return 'All'
    if w1 and w2: return 'Objective+Expect'
    if w1 and w3: return 'Objective+Compare'
    if w2 and w3: return 'Expect+Compare'
    if w1: return 'Objective only'
    if w2: return 'Expect only'
    if w3: return 'Compare only'
    return 'Unknown'


def run_single_experiment(params_row, base_config):
    """
    Configures and runs a single experiment for a given set of parameters
    using the user's actual BasicExperimentRunner.
    """
    param_map = copy.deepcopy(base_config)

    param_map["agent_params"]["w1"] = float(params_row['w1'])
    param_map["agent_params"]["w2"] = float(params_row['w2'])
    param_map["agent_params"]["w3"] = float(params_row['w3'])
    param_map["agent_params"]["w4"] = float(params_row['w4'])
    param_map["agent_params"]["alpha"] = float(params_row['alpha'])
    param_map["agent_params"]["aspiration_level"] = float(params_row['roh'])
    param_map["agent_params"]["epsilon"] = float(params_row['epsilon'])

    if params_row['pain_model'] == 'normal':
        param_map["agent_params"]["pain_model"] = normal_pain
    elif params_row['pain_model'] == 'chronic':
        param_map["agent_params"]["pain_model"] = chronic_pain
    else:  # 'no' pain
        param_map["agent_params"]["pain_model"] = normal_pain

    runner = BasicExperimentRunner(param_map)
    results = runner.run_experiments_with_all_episode_results()

    if len(results) > 1:
        actual_mean = statistics.mean(results)
        actual_std = statistics.stdev(results)
    elif len(results) == 1:
        actual_mean = results[0]
        actual_std = 0
    else:
        actual_mean = 0
        actual_std = 0

    print(f"Ran experiment for {params_row['category']} - {params_row['pain_model']}: "
          f"Actual Mean={actual_mean:.2f}, "
          f"Actual Std={actual_std:.2f}")

    expected_mean = params_row['mean']
    expected_std = params_row['std']

    if not math.isclose(actual_mean, expected_mean, rel_tol=1e-3) or \
            not math.isclose(actual_std, expected_std, rel_tol=1e-3):
        raise ValueError(
            f"\n\n--- REPRODUCIBILITY CHECK FAILED ---\n"
            f"Experiment: {params_row['category']} - {params_row['pain_model']}\n"
            f"  Expected Mean: {expected_mean:.4f}, Got: {actual_mean:.4f}\n"
            f"  Expected Std:  {expected_std:.4f}, Got: {actual_std:.4f}\n"
            f"Check if your experiment code has changed."
        )

    return results


df = pd.read_csv(experiment_folder_path + 'best_performing_agents.csv')

with open(experiment_folder_path + "base_config.json") as file:
    base_config = json.load(file)

df['category'] = df.apply(get_reward_category, axis=1)

results_df = df.copy()
results_df['t_stat_vs_no_pain'] = None
results_df['p_value_vs_no_pain'] = None

# Group by category and perform paired t-tests
for category_name, group in df.groupby('category'):
    print(f"\n--- Processing Category: {category_name} ---")

    try:
        params_no_pain = group[group['pain_model'] == 'no'].iloc[0]
        params_normal_pain = group[group['pain_model'] == 'normal'].iloc[0]
        params_chronic_pain = group[group['pain_model'] == 'chronic'].iloc[0]
    except IndexError:
        print(f"Skipping category '{category_name}' due to missing agent types.")
        continue

    # Re-run experiments to get paired, raw episode data
    results_no_pain = run_single_experiment(params_no_pain, base_config)
    results_normal_pain = run_single_experiment(params_normal_pain, base_config)
    results_chronic_pain = run_single_experiment(params_chronic_pain, base_config)

    # --- Perform Paired T-Tests (One-Sided) ---

    # 1. Normal Pain vs. No Pain
    t_stat_normal, p_value_normal = stats.ttest_rel(
        results_normal_pain,
        results_no_pain,
        alternative='greater'  # Test if mean(normal) > mean(no)
    )
    print(f"T-Test Normal vs No Pain: t-stat={t_stat_normal:.3f}, p-value={p_value_normal:.5f}")

    # 2. Chronic Pain vs. No Pain
    t_stat_chronic, p_value_chronic = stats.ttest_rel(
        results_chronic_pain,
        results_no_pain,
        alternative='greater'  # Test if mean(chronic) > mean(no)
    )
    print(f"T-Test Chronic vs No Pain: t-stat={t_stat_chronic:.3f}, p-value={p_value_chronic:.5f}")

    idx_normal = params_normal_pain.name
    idx_chronic = params_chronic_pain.name

    results_df.loc[idx_normal, ['t_stat_vs_no_pain', 'p_value_vs_no_pain']] = [t_stat_normal, p_value_normal]
    results_df.loc[idx_chronic, ['t_stat_vs_no_pain', 'p_value_vs_no_pain']] = [t_stat_chronic, p_value_chronic]

file_path = base_config["experiment_params"]["experiment_name"].lower().replace(' ', '_') + '_statistical_results.csv'
results_df.to_csv(file_path, index=False)
print("\nStatistical analysis complete. Results saved to " + file_path)


