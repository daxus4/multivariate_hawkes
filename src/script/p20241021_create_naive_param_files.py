import os

import pandas as pd

pairs = [
    "BTC_USD",
    "BTC_USDT",
    "ETH_USD",
    "ETH_USDT",
    "ETH_BTC",
]

for pair in pairs:
    # Define the folder path where your orderbook files are located
    orderbook_changes_path = (
        f"C:\\Users\\Admin\\OneDrive - Politecnico di Milano\\phd\\dati\\hawkes\\orderbook_changes\\"
        + pair
        + "\\"
    )
    params_path = (
        "C:\\Users\\Admin\\OneDrive - Politecnico di Milano\\phd\\code\\multivariate_hawkes\\data\\trained_params\\naive\\"
        + pair
        + "\\next_time_jump_seconds_ 1\\"
    )
    os.makedirs(params_path, exist_ok=True)

    df = pd.read_csv(
        f"C:\\Users\\Admin\\OneDrive - Politecnico di Milano\\phd\\dati\\hawkes\\trained_params\\univariate_hawkes_new_data\\{pair}\\hawkes_decay_10min.tsv",
        sep="\t",
    )

    start_simulation_df = pd.read_csv(
        f"C:\\Users\\Admin\\OneDrive - Politecnico di Milano\\phd\\dati\\hawkes\\orderbook_changes\\{pair}\\start_simulations_sequential_first50.csv"
    )

    df = df.merge(
        start_simulation_df, on=["timestamp", "timestamp_density"], how="inner"
    )
    print(len(df.index))

    # Loop through each row in the dataframe
    for index, row in df.iterrows():
        timestamp = int(row["timestamp"])
        timestamp_sim = int(row["timestamp_density"])

        # Find the corresponding file with the timestamp
        # Check both the normal and interrupted version
        file_name_normal = f"orderbook_changes_{timestamp}"
        file_name_interrupted = f"orderbook_changes_{timestamp}_interrupted"

        # Check if the file exists in the folder
        if os.path.exists(
            os.path.join(orderbook_changes_path, file_name_normal + ".tsv")
        ):
            base_file_name = file_name_normal
        elif os.path.exists(
            os.path.join(orderbook_changes_path, file_name_interrupted + ".tsv")
        ):
            base_file_name = file_name_interrupted
        else:
            print(f"File for timestamp {timestamp} not found!")
            continue  # Skip this row if no corresponding file is found

        # Create the three new files with suffixes _alpha, _beta, and _mu
        param_file = os.path.join(
            params_path, f"{base_file_name}_{timestamp_sim}_next_event_time_jump.txt"
        )

        # Write the corresponding values to each file
        with open(param_file, "w") as f_alpha:
            f_alpha.write(str(1))

    print("Files created successfully.")
