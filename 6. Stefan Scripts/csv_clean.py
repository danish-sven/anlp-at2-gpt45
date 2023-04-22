"""
This script reads multiple CSV files containing sentence pairs (Original and Corrected)
from a specified folder (CSV). It calculates the Levenshtein distance and cosine similarity
between the Original and Corrected sentences. The script then filters the pairs based on
minimum row count and writes the resulting DataFrame to a new CSV file in the 'training' folder.

Usage:
    Place the input CSV files in the 'CSV' folder and run the script.
"""

import os
import pandas as pd
from Levenshtein import distance as levenshtein_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cosine_similarity(text1, text2):
    """
    Calculate the cosine similarity between two text strings.

    Args:
        text1 (str): The first text string.
        text2 (str): The second text string.

    Returns:
        float: The cosine similarity between the two text strings.
    """
    try:
        vectorizer = TfidfVectorizer().fit_transform([text1, text2])
        return cosine_similarity(vectorizer)[0][1]
    except ValueError:
        return 0


def calculate_metrics(row):
    """
    Calculate the Levenshtein distance and cosine similarity for a DataFrame row.

    Args:
        row (pd.Series): A row from a DataFrame containing 'Original' and 'Corrected' columns.

    Returns:
        pd.Series: A new row with 'levenshtein' and 'cosine' columns.
    """
    original = row['Original']
    corrected = row['Corrected']

    if isinstance(original, str) and isinstance(corrected, str):
        lev = levenshtein_distance(original, corrected)
        cos = calculate_cosine_similarity(original, corrected)
    else:
        lev = 0
        cos = 1

    return pd.Series({'levenshtein': lev, 'cosine': cos})


csv_folder = 'CSV'
training_folder = 'training'
min_rows = 1000
threshold_levenshtein = 5
threshold_cosine = 0.8

if not os.path.exists(training_folder):
    os.makedirs(training_folder)

for file in os.listdir(csv_folder):
    if file.endswith('.csv'):
        file_path = os.path.join(csv_folder, file)

        df = pd.read_csv(file_path)

        # Check if the necessary columns exist in the DataFrame
        if 'Original' not in df.columns or 'Corrected' not in df.columns:
            print(f"Skipping {file}: Missing 'Original' or 'Corrected' columns.")
            continue

        # Check if the DataFrame has enough rows
        if len(df) < min_rows:
            print(f"Skipping {file}: Fewer than {min_rows} rows.")
            continue

        df[['levenshtein', 'cosine']] = df.apply(calculate_metrics, axis=1)

        if len(df) >= min_rows:
            new_file_path = os.path.join(training_folder, file)
            df.to_csv(new_file_path, index=False)
