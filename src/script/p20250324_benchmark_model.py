import os
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

import src.constants as CONST
from src.conf.events_conf.events_conf import EventsConf
from src.conf.training.model.multivariate_hawkes_training_conf import (
    MultivariateHawkesTrainingConf,
)
from src.conf.training.training_conf import TrainingConf
from src.lob_data_loader.loading_info_getter import LoadingInfoGetter
from src.lob_data_loader.lob_data_loader import LOBDataLoader
from src.lob_period.lob_period_extractor import LOBPeriodExtractor
from src.multivariate_hawkes_training.lob_event_combinator import LOBEventCombinator

CONF_EVENTS_FILENAME = "mid_price_change_events_conf.yml"
CONF_TRAINING_FILENAME = "training_conf.yml"
CONF_MULTIVARIATE_HAWKES_TRAINING_FILENAME = "multivariate_hawkes_training_conf.yml"


def get_conf(path: str) -> Dict:
    with open(path, "r") as f:
        conf = yaml.safe_load(f)
    return conf


def get_event_type_times_maps_with_combined_types(
    event_type_times_map: List[Dict[str, np.ndarray]],
    combined_name_events_to_combine_map: Dict[str, List[str]],
) -> List[Dict[str, np.ndarray]]:

    lob_event_combinator = LOBEventCombinator([event_type_times_map])

    for (
        combination_name,
        lob_events_to_combine,
    ) in combined_name_events_to_combine_map.items():
        event_type_times_maps = (
            lob_event_combinator.get_event_type_times_maps_with_new_combination(
                lob_events_to_combine,
                combination_name=combination_name,
            )
        )
        lob_event_combinator.event_type_times_maps = event_type_times_maps

    return event_type_times_maps


def get_event_type_times_maps_filtered(
    event_type_times_map: List[Dict[str, np.ndarray]], events_to_compute: List[str]
) -> List[Dict[str, np.ndarray]]:
    return [
        {
            key: value
            for key, value in event_type_times.items()
            if key in events_to_compute
        }
        for event_type_times in event_type_times_map
    ]


if __name__ == "__main__":
    multivariate_hawkes_training_conf_map = get_conf(
        os.path.join(
            CONST.CONF_TRAINING_MODEL_FOLDER, CONF_MULTIVARIATE_HAWKES_TRAINING_FILENAME
        )
    )
    multivariate_hawkes_training_conf = MultivariateHawkesTrainingConf.from_dict(
        multivariate_hawkes_training_conf_map
    )

    training_conf_map = get_conf(
        os.path.join(CONST.CONF_TRAINING_FOLDER, CONF_TRAINING_FILENAME)
    )
    training_conf = TrainingConf.from_dict(training_conf_map)

    events_conf_map = get_conf(
        os.path.join(CONST.CONF_EVENTS_FOLDER, CONF_EVENTS_FILENAME)
    )
    events_conf = EventsConf.from_dict(events_conf_map)

    pair_orderbook_changes_path = os.path.join(
        CONST.ORDERBOOK_CHANGES_FOLDER, training_conf.pair
    )
    periods_df = pd.read_csv(
        os.path.join(
            pair_orderbook_changes_path, CONST.SIMULATION_START_TIMESTAMPS_FILE
        )
    )

    loading_info_for_all_dfs = LoadingInfoGetter(periods_df).get_loading_info(
        lob_df_folder_path=pair_orderbook_changes_path,
        lob_df_prefix=CONST.ORDERBOOK_CHANGES_FILE_PREFIX,
    )

    training_time_file_likelihood_map = {
        training_time: {"file": [], "score": []}
        for training_time in training_conf.seconds_in_a_period
    }

    training_times_cpu = []

    for loading_info in loading_info_for_all_dfs:
        lob_df_loader = LOBDataLoader()
        lob_df = lob_df_loader.get_lob_dataframe(loading_info.path, 10)

        lob_period_extractor = LOBPeriodExtractor(lob_df)

        for start_simulation_time in loading_info.start_times:
            for training_time_seconds in training_conf.seconds_in_a_period:
                start_time = start_simulation_time - training_time_seconds

                end_time = start_simulation_time + 120

                lob_period = lob_period_extractor.get_lob_period(start_time, end_time)
                lob_df_for_events = lob_period.get_lob_df_with_timestamp_column()

                pbid = (
                    lob_df_for_events["BidPrice1"] - lob_df_for_events[f"BidPrice{5}"]
                )
                pask = (
                    lob_df_for_events[f"AskPrice{5}"] - lob_df_for_events["AskPrice1"]
                )
                lob_df_for_events["BaseImbalance"] = (pbid - pask) / (pbid + pask)
                lob_df_for_events["MidPrice"] = (
                    lob_df_for_events[f"AskPrice1"] + lob_df_for_events["BidPrice1"]
                ) / 2
                lob_df_for_events["Return"] = (
                    lob_df_for_events["MidPrice"]
                    - lob_df_for_events["MidPrice"].shift(1)
                ) / lob_df_for_events["MidPrice"].shift(1)

                lob_df_for_events = lob_df_for_events[lob_df_for_events["Return"] != 0]

                lob_df_for_events["Return-1"] = lob_df_for_events["Return"].shift(1)
                lob_df_for_events["Return-2"] = lob_df_for_events["Return"].shift(2)
                lob_df_for_events["Return-3"] = lob_df_for_events["Return"].shift(3)
                lob_df_for_events["Return-4"] = lob_df_for_events["Return"].shift(4)
                lob_df_for_events["Return-5"] = lob_df_for_events["Return"].shift(5)

                lob_df_for_events["AvgPastReturn"] = lob_df_for_events[
                    ["Return-1", "Return-2", "Return-3", "Return-4", "Return-5"]
                ].mean(axis=1)

                lob_df_for_events = lob_df_for_events[
                    (lob_df_for_events["Timestamp"] >= training_time_seconds)
                ]

                lob_df_for_events = lob_df_for_events[
                    [
                        "Timestamp",
                        "BaseImbalance",
                        "MidPrice",
                        "Return",
                        "AvgPastReturn",
                    ]
                ]

                prefix = os.path.basename(loading_info.path)
                prefix = os.path.splitext(prefix)[0]

                lob_df_for_events.to_csv(
                    f"data/coe_dataframes/benchmark/{prefix}_{start_simulation_time}.tsv",
                    index=False,
                    sep="\t",
                )
