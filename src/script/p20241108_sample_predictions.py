import os
from itertools import product
from typing import List

import pandas as pd

from src.constants import COE_SIMULATION_DATAFRAMES_FOLDER

pairs = ["BTC_USD", "BTC_USDT", "ETH_USD", "ETH_USDT", "ETH_BTC"]
base_imbalances = [5, 10, 15]
methods = [
    "multivariate_hawkes",
    "univariate_hawkes",
    "poisson",
    "moving_average",
    "naive",
]


def get_immediate_subfolders_for_multiple_directories(
    folder_list: List[str],
) -> List[str]:
    all_subfolders = []
    for directory in folder_list:
        subfolders = [
            os.path.join(directory, d)
            for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))
        ]
        all_subfolders.extend(subfolders)

    return all_subfolders


def get_min_row_count_multiple_files(file_list: List[str]) -> int:
    min_row_count = 1e10
    for file in file_list:
        row_count = sum(1 for _ in open(file))
        if min_row_count is None or row_count < min_row_count:
            min_row_count = row_count

    return min_row_count


def extract_and_save_random_rows(input_file: str, output_file: str, n: int) -> None:
    df = pd.read_csv(input_file, sep="\t")
    random_rows = df.sample(n=n)
    random_rows = random_rows.sort_values(by=["real"])
    random_rows.to_csv(output_file, sep="\t", index=False)


def get_recursive_subdirectories(directory: str) -> List[str]:
    subdirs = []
    for root, dirs, _ in os.walk(directory):
        if not dirs:  # If `dirs` is empty, this is a final subdirectory
            relative_path = os.path.relpath(root, start=directory)
            subdirs.append(relative_path)

    return subdirs


if __name__ == "__main__":
    filtered_simulation_subfolder = os.path.join(
        COE_SIMULATION_DATAFRAMES_FOLDER, "filtered"
    )

    directories = get_recursive_subdirectories(filtered_simulation_subfolder)
    directories = [os.path.join(filtered_simulation_subfolder, d) for d in directories]

    for base_imbalance, pair in product(base_imbalances, pairs):
        pair_base_imbalance_dirs = [
            d
            for d in directories
            if f"bi_level_{base_imbalance}" in d and f"\\{pair}\\" in d
        ]

        prediction_files = [
            f
            for f in os.listdir(pair_base_imbalance_dirs[0])
            if os.path.isfile(os.path.join(pair_base_imbalance_dirs[0], f))
            and f.endswith(".tsv")
            and f.startswith("orderbook_changes_")
        ]

        for prediction_file in prediction_files:
            current_simulation_prediction_files = [
                os.path.join(d, prediction_file) for d in pair_base_imbalance_dirs
            ]

            min_row_count = get_min_row_count_multiple_files(
                current_simulation_prediction_files
            )
            min_row_count = min_row_count - 1  # Remove header row

            for (
                current_simulation_prediction_file
            ) in current_simulation_prediction_files:
                output_file = current_simulation_prediction_file.replace(
                    "filtered", "filtered_sampled"
                )
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                extract_and_save_random_rows(
                    current_simulation_prediction_file, output_file, min_row_count
                )
