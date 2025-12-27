import pandas as pd
import os
import glob

# Set the path to your CSV files
experiment_name = "stationary"

# Get a list of all CSV files in the folder
csv_files = glob.glob(os.path.join("experiments/batch_grid_search/" + experiment_name + "/results", '*.csv'))

# Read and combine all CSV files
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

file_name = "results_" + experiment_name + ".csv"
combined_df.to_csv("experiments/batch_grid_search/" + experiment_name + "/" + file_name, index=False)
print("Combined CSV saved as " + file_name)

