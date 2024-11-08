import os

import pandas as pd

from src.lob_data_loader.lob_data_loader import LOBDataLoader
from src.lob_period.lob_period_extractor import LOBPeriodExtractor

pairs = ['BTC_USD', 'ETH_USD', 'BTC_USDT', 'ETH_USDT', 'ETH_BTC']
training_times = [300, 600, 900]

for pair in pairs:
    for training_time in training_times:
        # Define the folder path where your orderbook files are located
        orderbook_changes_path = f'C:\\Users\\Admin\\Desktop\\phd\\multivariate_hawkes\\data\\orderbook_changes\\' + pair + '\\'
        params_path = 'C:\\Users\\Admin\\Desktop\\phd\\multivariate_hawkes\\data\\trained_params\\poisson\\' + pair + f'\\training_time_{training_time}\\'
        os.makedirs(params_path, exist_ok=True)

        df = pd.read_csv(
            f'C:\\Users\\Admin\\Desktop\\phd\\hawkes_coe\\hawkes\\data_{pair}\\hawkes_best_decays\\hawkes_decay_10min.tsv',
            sep='\t'
        )

        # Loop through each row in the dataframe
        for index, row in df.iterrows():
            timestamp = int(row['timestamp'])
            timestamp_sim = int(row['timestamp_density'])
            
            # Find the corresponding file with the timestamp
            # Check both the normal and interrupted version
            file_name_normal = f'orderbook_changes_{timestamp}'
            file_name_interrupted = f'orderbook_changes_{timestamp}_interrupted'
            
            # Check if the file exists in the folder
            if os.path.exists(os.path.join(orderbook_changes_path, file_name_normal + '.tsv')):
                base_file_name = file_name_normal
            elif os.path.exists(os.path.join(orderbook_changes_path, file_name_interrupted + '.tsv')):
                base_file_name = file_name_interrupted
            else:
                print(f"File for timestamp {timestamp} not found!")
                continue  # Skip this row if no corresponding file is found
            
            lob_df_loader = LOBDataLoader()
            lob_df = lob_df_loader.get_lob_dataframe(
                os.path.join(orderbook_changes_path, base_file_name + '.tsv'), 10
            )

            lob_period_extractor = LOBPeriodExtractor(lob_df)

            start_time = (
                timestamp_sim - training_time
            )
            end_time = timestamp_sim

            lob_period = lob_period_extractor.get_lob_period(start_time, end_time)
            lob_df_for_events = lob_period.get_lob_df_with_timestamp_column()
            lob_df_for_events = lob_df_for_events[lob_df_for_events['Return'] != 0]
            mean_number_of_events = len(lob_df_for_events) / training_time
            

            # Create the three new files with suffixes _alpha, _beta, and _mu
            param_file = os.path.join(params_path, f'{base_file_name}_{timestamp_sim}_mu.txt')
            
            # Write the corresponding values to each file
            with open(param_file, 'w') as f_alpha:
                f_alpha.write(str(mean_number_of_events))
            
        print("Files created successfully.")