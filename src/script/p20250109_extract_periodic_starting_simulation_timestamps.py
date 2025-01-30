import os
from typing import Dict, List

import pandas as pd

SIMULATION_TIME_DURATION = pd.Timedelta("2min")
TRAINING_TIME_DURATION = pd.Timedelta("30min")
PATH_ORDERBOOK_DIRECTORY = "C:\\Users\\Admin\\OneDrive - Politecnico di Milano\\phd\\dati\\hawkes\\orderbook_changes\\{}\\"
MARKETS = ["BTC_USD", "BTC_USDT", "ETH_BTC", "ETH_USD", "ETH_USDT"]


def get_preprocessed_orderbook_df(
    orderbook_df: pd.DataFrame,
    initial_offset: pd.Timedelta,
    final_offset: pd.Timedelta,
    safety_offset: pd.Timedelta = pd.Timedelta("0s"),
) -> pd.DataFrame:
    orderbook_df["Timestamp"] = pd.to_datetime(orderbook_df["Timestamp"], unit="ms")

    min_correct_timestamp = (
        orderbook_df["Timestamp"].min() + initial_offset + safety_offset
    )
    max_correct_timestamp = (
        orderbook_df["Timestamp"].max() - final_offset - safety_offset
    )

    orderbook_df = orderbook_df[
        (orderbook_df["Timestamp"] >= min_correct_timestamp)
        & (orderbook_df["Timestamp"] <= max_correct_timestamp)
    ].copy()

    orderbook_df["MidPrice"] = (
        orderbook_df["AskPrice1"] + orderbook_df["BidPrice1"]
    ) / 2
    orderbook_df["Difference"] = -orderbook_df["MidPrice"] + orderbook_df[
        "MidPrice"
    ].shift(-1)
    orderbook_df = orderbook_df.dropna()
    orderbook_df = orderbook_df[orderbook_df["Difference"] != 0]

    return orderbook_df


def get_files(directory: str) -> List[str]:
    return [
        f
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.endswith(".tsv")
    ]


def save_densities_table(config: Dict, file_path: str):
    df_map = {
        "timestamp": [],
        "timestamp_density": [],
    }

    for k, v in config.items():
        num_densities = len(v)
        timestamp = k.split("_")[2].split(".")[0]
        df_map["timestamp"].extend([float(timestamp)] * num_densities)

        for info_density in v:
            df_map["timestamp_density"].append(info_density)

    df_map = pd.DataFrame(df_map)
    df_map["timestamp"] = df_map["timestamp"].apply(lambda x: int(x))

    df_map.to_csv(file_path, index=False)


if __name__ == "__main__":
    for market in MARKETS:
        path_orderbook_directory = PATH_ORDERBOOK_DIRECTORY.format(market)

        file_densities_map = dict()

        for orderbook_file_path in get_files(path_orderbook_directory):
            df = pd.read_csv(
                os.path.join(path_orderbook_directory, orderbook_file_path), sep="\t"
            )
            df = get_preprocessed_orderbook_df(
                df,
                TRAINING_TIME_DURATION,
                SIMULATION_TIME_DURATION,
                safety_offset=pd.Timedelta("1s"),
            )
            df["Timestamp"] = df["Timestamp"].dt.floor("s")

            if not df.empty:
                # Initialize variables
                simulation_starts = []

                start_time = df["Timestamp"].min()
                end_time = df["Timestamp"].max()

                # create a list of simulation start times starting from start_time with a duration of training_delta + simulation_delta
                while (
                    start_time + TRAINING_TIME_DURATION + SIMULATION_TIME_DURATION
                    <= end_time
                ):
                    start_simulation_timestamp = int(
                        pd.Timestamp(start_time + TRAINING_TIME_DURATION).timestamp()
                    )
                    simulation_starts.append(start_simulation_timestamp)
                    start_time = (
                        start_time + SIMULATION_TIME_DURATION + pd.Timedelta(seconds=1)
                    )
                if len(simulation_starts) > 0:
                    file_densities_map[orderbook_file_path] = simulation_starts

        save_densities_table(
            file_densities_map,
            os.path.join(path_orderbook_directory, "start_simulations_sequential.csv"),
        )

        # save file_densities_map in json
        # with open(f"data_{market}/file_densities_map.json", "w") as f:
        #    json.dump(file_densities_map, f)
