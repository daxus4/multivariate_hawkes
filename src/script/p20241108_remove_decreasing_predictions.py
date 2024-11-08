import os
from typing import List

import pandas as pd

from src.constants import SIMULATIONS_FOLDER


def get_recursive_subdirectories(directory: str) -> List[str]:
    subdirs = []
    for root, dirs, _ in os.walk(directory):
        for d in dirs:
            subdirs.append(os.path.join(root, d))
    return subdirs

def get_df_without_decreasing_predictions(
    df: pd.DataFrame, col: str='predicted'
) -> pd.DataFrame:
    """
    Removes rows where the predicted value is lower than any predicted value in previous rows.
    """
    return df[df[col] >= df[col].cummax()].reset_index(drop=True)
    
def filter_and_save_orderbook_changes(directory_path: str) -> None:
    """
    Searches for .tsv files starting with 'orderbook_changes_' in the given directory,
    applies filtering to remove decreasing predictions, and saves the results to new files.

    Parameters:
    - directory_path (str): Path to the directory containing the .tsv files.
    """
    # Traverse the directory and subdirectories
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            # Check if file name starts with 'orderbook_changes_' and has a .tsv extension
            if file_name.startswith("orderbook_changes_") and file_name.endswith(".tsv"):
                file_path = os.path.join(root, file_name)
                
                # Load the file into a DataFrame
                df = pd.read_csv(file_path, sep='\t')
                
                # Apply the filtering function
                filtered_df = get_df_without_decreasing_predictions(df)
                
                # Define the new file name with the 'filtered_' prefix
                new_file_name = f'filtered_{file_name}'
                new_file_path = os.path.join(root, new_file_name)
                
                # Save the filtered DataFrame to a new .tsv file
                filtered_df.to_csv(new_file_path, sep='\t', index=False)


if __name__ == "__main__":
    simulations_subdirs = get_recursive_subdirectories(SIMULATIONS_FOLDER)
    for subdir in simulations_subdirs:
        filter_and_save_orderbook_changes(subdir)
        print(f"Filtered orderbook changes in {subdir} successfully.")

