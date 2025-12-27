# Exploration Through Introspection: A Self-Aware Reward Model

This repository contains the source code and reproduction steps for the paper **"Exploration Through Introspection: A Self-Aware Reward Model"**.

## 🛠 Prerequisites & Installation

Please ensure you have **Python 3.12** installed. We recommend using a virtual environment to manage dependencies. The project folder should be named `pain_rl`.

```bash
# Create and activate virtual environment (example using venv)
python3.12 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```
If this does not work for some reason, install `joblib`, `matplotlib`, `numpy`, `pandas`, `pygame-ce`, `pytest` and `scipy` manually.

-----

## 🚀 Reproducing Grid Search Results

> **Note:** The grid search over the selected parameter space is computationally expensive. Execution is designed for **HPC clusters using PBS scheduling**. Running this on consumer hardware is not recommended.

### 1\. Configure and Generate Jobs (Local)

Before deploying to the cluster, generate the job files locally:

1.  Open `scripts/generate_batch_jobs.py`.
2.  **Select Experiment:** Set `experiment_name` to `"stationary"`, `"non_stationary"`, or `"lifetime"`.
3.  **Set Remote Path:** Update `source_code_location` to point to the root folder where this project will reside on your HPC file system.
4.  **Configure Parameters:**
      * **Search Space:** Edit `experiments/batch_grid_search/[experiment_name]/param_grid.json` to define the grid search values.
      * **Fixed Parameters:** Edit `base_config.json` in the same folder for parameters that remain constant (e.g., discount factor $\gamma$). *Note: Values in `param_grid.json` override `base_config.json`.*
5.  **Generate Files:** From the project root folder, run the script locally to create the `.pbs` files.

<!-- end list -->

```bash
python -m scripts.generate_batch_jobs
```

### 2\. Deploy and Execute (HPC Cluster)

1.  **Transfer Code:** Upload the entire project (including the newly generated `.pbs` files) to the `source_code_location` on your HPC cluster defined in the previous step.

2.  **Launch Jobs:** Create and run a shell script to submit all jobs.

    *Example `start_batch_grid_search.sh`:*

    ```bash
    #!/bin/bash

    # Navigate to the pbs_jobs directory (adjust path as needed)
    cd /gpfs/project/user/pain_rl/experiments/batch_grid_search/stationary/pbs_jobs

    for job in *.pbs; do
        qsub "$job"
    done
    ```

3.  **Run the script:**

    ```bash
    chmod +x start_batch_grid_search.sh
    ./start_batch_grid_search.sh
    ```

### 3\. Merge Results

Once all jobs have finished:

1.  Open `scripts/merge_results.py`.
2.  Set `experiment_name` to the experiment you ran.
3.  Run the script from the project root to aggregate batch results into a single `.csv` file.

<!-- end list -->

```bash
python -m scripts.merge_results
```

We recommend copying the `experiments/batch_grid_search/[experiment_name]/results` folder back to your local project copy and executing the steps there.

-----

## 📊 Reproducing Figures and Plots

*Prerequisite: Complete the [Grid Search](#-reproducing-grid-search-results) step first.*

1.  Navigate to the specific experiment folder (e.g., `experiments/batch_grid_search/stationary`).
2.  Run the corresponding analysis notebook (e.g., `analysis_stationary.ipynb`).
3.  Generated figures will be saved in the `analysis_plots` folder within that directory.

-----

## 📈 Reproducing Statistical Tests

*Prerequisite: You must have generated `best_performing_agents.csv` by running the [analysis notebooks](experiments/batch_grid_search/stationary/analysis_stationary.ipynb) in the previous step.*

1.  Open `experiments/statistical_test_analysis/statistical_analysis_runner.py`.
2.  Update the `experiment_name` variable to match the desired experiment.
3.  Run the script:

<!-- end list -->

```bash
python experiments/statistical_test_analysis/statistical_analysis_runner.py
```

The resulting `.csv` containing statistical test results will be generated in the same folder.

## Acknowledgements
Parts of the code in this repository are inspired by the following paper:

* **Title:** The pursuit of happiness: A reinforcement learning perspective on habituation and comparisons
* **Authors:** Rachit Dubey, Thomas L. Griffiths, Peter Dayan
* **Source:** [https://doi.org/10.1371/journal.pcbi.1010316](https://doi.org/10.1371/journal.pcbi.1010316)
* **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

Their code is available on [GitHub](https://github.com/rachit-dubey/happiness_RL).