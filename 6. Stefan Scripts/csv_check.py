import os
import pandas as pd

training_folder = 'training'

# Iterate through the files in the training folder
for file in os.listdir(training_folder):
    if file.endswith('.csv'):
        file_path = os.path.join(training_folder, file)
        
        df = pd.read_csv(file_path)

        # Check if the necessary columns exist in the DataFrame
        if 'levenshtein' not in df.columns or 'cosine' not in df.columns:
            print(f"Skipping {file}: Missing 'levenshtein' or 'cosine' columns.")
            continue

        # Filter rows with Levenshtein scores > 0 and cosine scores < 1
        filtered_df = df[(df['levenshtein'] > 0) | (df['cosine'] < 1)]

        if not filtered_df.empty:
            print(f"{file}: Found {len(filtered_df)} rows with levenshtein > 0 or cosine < 1")
        else:
            print(f"{file}: No rows with levenshtein > 0 and cosine < 1")
