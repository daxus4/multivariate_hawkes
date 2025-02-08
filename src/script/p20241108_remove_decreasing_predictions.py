import os
from typing import List

import pandas as pd

from src.constants import COE_SIMULATION_DATAFRAMES_FOLDER


def get_recursive_subdirectories(directory: str) -> List[str]:
    subdirs = []
    for root, dirs, _ in os.walk(directory):
        if not dirs:  # If `dirs` is empty, this is a final subdirectory
            relative_path = os.path.relpath(root, start=directory)
            subdirs.append(relative_path)

    return subdirs


def get_df_without_decreasing_predictions(
    df: pd.DataFrame, col: str = "predicted"
) -> pd.DataFrame:
    """
    Removes rows where the predicted value is lower than any predicted value in previous rows.
    """
    return df[df[col] >= df[col].cummax()].reset_index(drop=True)


def get_df_without_increasing_predictions(
    df: pd.DataFrame, col: str = "predicted"
) -> pd.DataFrame:
    """
    Removes rows where the predicted value is higher than any predicted value in the following rows.
    """
    return df[df[col] <= df[col].iloc[::-1].cummin().iloc[::-1]].reset_index(drop=True)


def filter_and_save_orderbook_changes(
    main_path, relative_directory, main_subfolder: str = "filtered"
) -> None:
    """
    Searches for .tsv files starting with 'orderbook_changes_' in specified directories,
    applies filtering to remove decreasing predictions, and saves the results in a new path.

    Parameters:
    - main_path (str): The main path containing the directories.
    - relative_directory (str): Relative subdirectory to search in.
    """
    filtered_base_path = os.path.join(main_path, main_subfolder)
    os.makedirs(filtered_base_path, exist_ok=True)

    abs_dir = os.path.join(main_path, relative_directory)

    filtered_dir = os.path.join(filtered_base_path, relative_directory)
    os.makedirs(filtered_dir, exist_ok=True)

    # Look for .tsv files starting with 'orderbook_changes_' in this directory
    for file_name in os.listdir(abs_dir):
        if file_name.startswith("orderbook_changes_") and file_name.endswith(".tsv"):
            file_path = os.path.join(abs_dir, file_name)

            df = pd.read_csv(file_path, sep="\t")
            filtered_df = get_df_without_increasing_predictions(df)

            new_file_path = os.path.join(filtered_dir, file_name)

            filtered_df.to_csv(new_file_path, sep="\t", index=False)
            print(f"Filtered file saved as: {new_file_path}")


if __name__ == "__main__":
    simulations_subdirs = get_recursive_subdirectories(COE_SIMULATION_DATAFRAMES_FOLDER)
    for subdir in simulations_subdirs:
        filter_and_save_orderbook_changes(COE_SIMULATION_DATAFRAMES_FOLDER, subdir)
        print(f"Filtered orderbook changes in {subdir} successfully.")
