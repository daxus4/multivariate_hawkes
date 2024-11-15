import os
from typing import List

import pandas as pd

from src.constants import COE_SIMULATION_DATAFRAMES_FOLDER

pairs = ['BTC_USD', 'BTC_USDT', 'ETH_USD', 'ETH_USDT', 'ETH_BTC']

def get_immediate_subfolders_for_multiple_directories(
    folder_list: List[str]
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
    min_row_count = -1
    for file in file_list:
        row_count = sum(1 for _ in open(file))
        if min_row_count is None or row_count < min_row_count:
            min_row_count = row_count

    return min_row_count

def extract_and_save_random_rows(
    input_file: str, output_file: str, n: int
) -> None:
    df = pd.read_csv(input_file, sep='\t')
    random_rows = df.sample(n=n)
    random_rows = random_rows.sort_values(by=['real'])
    random_rows.to_csv(output_file, sep='\t', index=False)

if __name__ == '__main__':
    filtered_simulation_subfolder = os.path.join(COE_SIMULATION_DATAFRAMES_FOLDER, 'filtered')

    method_simulation_subfolders = [
        d
        for d in os.listdir(filtered_simulation_subfolder)
        if os.path.isdir(os.path.join(filtered_simulation_subfolder, d))
    ]

    method_simulation_subfolders = [
        os.path.join(d, pair)
        for d in method_simulation_subfolders
        for pair in pairs
        if os.path.isdir(os.path.join(filtered_simulation_subfolder, d, pair))
    ]

    method_simulation_subfolders_full_path = [
        os.path.join(filtered_simulation_subfolder, d)
        for d in method_simulation_subfolders
    ]

    training_params_simulation_subfolders = get_immediate_subfolders_for_multiple_directories(
        method_simulation_subfolders_full_path
    )

    simulated_time_training_params_simulation_subfolders = get_immediate_subfolders_for_multiple_directories(
        training_params_simulation_subfolders
    )

    prediction_files = [
        f
        for f in os.listdir(simulated_time_training_params_simulation_subfolders[0])
        if os.path.isfile(os.path.join(simulated_time_training_params_simulation_subfolders[0], f))
        and f.endswith('.tsv') and f.startswith('orderbook_changes_')
    ]

    for prediction_file in prediction_files:
        current_simulation_prediction_files = [
            os.path.join(d, prediction_file)
            for d in simulated_time_training_params_simulation_subfolders
        ]

        min_row_count = get_min_row_count_multiple_files(current_simulation_prediction_files)
        min_row_count = min_row_count - 1 # Remove header row

        for current_simulation_prediction_file in current_simulation_prediction_files:
            output_file = current_simulation_prediction_file.replace(
                'filtered', 'filtered_sampled'
            )
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            extract_and_save_random_rows(
                current_simulation_prediction_file, output_file, min_row_count
            )





