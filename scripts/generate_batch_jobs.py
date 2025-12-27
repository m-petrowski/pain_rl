import os
import json
from src.reinforcement_learning.experiments.grid_search import generate_param_combinations

experiment_name = "stationary"
source_code_location = "/gpfs/project/user/pain_rl" #location of this source code on the hpc cluster
root_folder_path = "experiments/batch_grid_search/"

experiment_folder_path = root_folder_path + experiment_name + "/"
with open(experiment_folder_path + "base_config.json") as f:
    base_config = json.load(f)

with open(experiment_folder_path + "param_grid.json") as file:
    param_grid = json.load(file)

combinations = generate_param_combinations(param_grid)
total = len(combinations)
batch_size = 500
os.makedirs(experiment_folder_path + "pbs_jobs", exist_ok=True)

for i in range(0, total, batch_size):
    start = i
    end = min(i + batch_size, total)
    job_index = i // batch_size

    job_script = f"""#!/bin/bash
    #PBS -l select=1:ncpus=8:mem=1GB
    #PBS -l walltime=5:00:00
    #PBS -A DialSys
    #PBS -N grid_batch_{job_index}

    cd {source_code_location}

    module load Python/3.12.3
    PIP_CONFIG_FILE=/software/python/pip.conf pip install --user joblib matplotlib numpy pandas pygame-ce pytest scipy

    python -m scripts.run_batch_grid_search {start} {end} {experiment_folder_path} {experiment_folder_path}results/batch_{job_index}.csv
    """

    with open(f"{experiment_folder_path}/pbs_jobs/grid_batch_{job_index}.pbs", "w") as f:
        f.write(job_script)

print(f"✅ Created {((total-1)//batch_size)+1} batch job scripts in pbs_jobs/")