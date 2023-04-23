"""
This script calculates the cosine similarity between text in two columns of a CSV file.
It processes the data in parallel using multiple processes to speed up the calculation.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool, cpu_count

# Function to process a chunk of the DataFrame and calculate cosine similarities
def process_chunk(args):
    df_chunk, vectorizer, column1, column2 = args

    # Transform the text in each column into a TF-IDF matrix
    matrix1 = vectorizer.transform(df_chunk[column1].astype(str))
    matrix2 = vectorizer.transform(df_chunk[column2].astype(str))

    # Calculate the cosine similarity between the corresponding rows of the two matrices
    cosine_similarities = cosine_similarity(matrix1, matrix2).diagonal()
    
    # Add the cosine similarity values to the DataFrame chunk
    df_chunk['cosine'] = cosine_similarities

    return df_chunk

# Function to calculate cosine similarity in parallel for a given CSV file
def calculate_cosine_similarity_parallel(csv_file, column1, column2, chunksize=1000):
    # Read the input CSV file
    df = pd.read_csv(csv_file)

    # Initialize the 'cosine' column if it doesn't exist
    if 'cosine' not in df.columns:
        df['cosine'] = None

    # Create a TfidfVectorizer and fit it on the text data in the first column
    vectorizer = TfidfVectorizer()
    vectorizer.fit(df[column1].astype(str))

    total_rows = len(df)
    num_chunks = (total_rows // chunksize) + 1

    # Split the DataFrame into chunks for parallel processing
    df_chunks = [df[i * chunksize:(i + 1) * chunksize] for i in range(num_chunks)]

    # Process the chunks in parallel using a pool of worker processes
    with Pool(cpu_count()) as pool:
        results = pool.map(
            process_chunk, [(chunk, vectorizer, column1, column2) for chunk in df_chunks]
        )

    # Combine the processed chunks back into a single DataFrame
    df_result = pd.concat(results, ignore_index=True)

    # Save the DataFrame with calculated cosine similarities to a new CSV file
    df_result.to_csv("/Users/stefanhall/Documents/Studies/MDSI/ANLP/AT2/anlp-at2-gpt45/2. Raw Data/Japanese_to_English_cosine.csv", index=False)

# Main entry point for the script
if __name__ == '__main__':
    csv_file = "/Users/stefanhall/Documents/Studies/MDSI/ANLP/AT2/anlp-at2-gpt45/2. Raw Data/Japanese_to_English_cosine.csv"
    calculate_cosine_similarity_parallel(csv_file, "original", "corrected")
