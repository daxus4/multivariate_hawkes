import os

import pandas as pd

pairs = [
    # "BTC_USD",
    # "BTC_USDT",
    "ETH_USD",
    "ETH_USDT",
    "ETH_BTC",
]
training_times = [300, 600, 900]

for pair in pairs:
    for training_time in training_times:
        # Define the folder path where your orderbook files are located
        orderbook_changes_path = (
            f"C:\\Users\\Admin\\OneDrive - Politecnico di Milano\\phd\\dati\\hawkes\\orderbook_changes\\"
            + pair
            + "\\"
        )
        params_path = (
            "C:\\Users\\Admin\\OneDrive - Politecnico di Milano\\phd\\code\\multivariate_hawkes\\data\\trained_params\\univariate_hawkes\\"
            + pair
            + "\\training_time_"
            + str(training_time)
            + "\\"
        )
        os.makedirs(params_path, exist_ok=True)

        training_time_in_minutes = training_time // 60

        df = pd.read_csv(
            f"C:\\Users\\Admin\\OneDrive - Politecnico di Milano\\phd\\dati\\hawkes\\trained_params\\univariate_hawkes_new_data\\{pair}\\hawkes_decay_{training_time_in_minutes}min.tsv",
            sep="\t",
        )

        # Loop through each row in the dataframe
        for index, row in df.iterrows():
            timestamp = int(row["timestamp"])
            timestamp_sim = int(row["timestamp_density"])
            alpha = row["alpha"]
            decay = row["decay"]
            baseline = row["baseline"]

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
            alpha_file = os.path.join(
                params_path, f"{base_file_name}_{timestamp_sim}_alpha.txt"
            )
            beta_file = os.path.join(
                params_path, f"{base_file_name}_{timestamp_sim}_beta.txt"
            )
            mu_file = os.path.join(
                params_path, f"{base_file_name}_{timestamp_sim}_mu.txt"
            )

            # Write the corresponding values to each file
            with open(alpha_file, "w") as f_alpha:
                f_alpha.write(str(alpha))

            with open(beta_file, "w") as f_beta:
                f_beta.write(str(decay))

            with open(mu_file, "w") as f_mu:
                f_mu.write(str(baseline))

        print("Files created successfully.")
