# VEDERE PERCHE MOBVING AVEREAGE NON FUNZIONA, VEDERE PERCHE MULTIVARIATE NON FUNZIONA
# FARE QUESTO PRIMA DI FILTER E SAMPLE, COSI POSSO UNIRE BI AI TEMPI CON UN CONCATENATE ORIZZONTALE
import os
from itertools import product
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

import src.constants as CONST
from src.conf.coe.coe_conf import CoeConf
from src.conf.events_conf.events_conf import EventsConf
from src.conf.training.model.multivariate_hawkes_training_conf import (
    MultivariateHawkesTrainingConf,
)
from src.lob_data_loader.loading_info_getter import LoadingInfoGetter
from src.lob_data_loader.lob_data_loader import LOBDataLoader
from src.lob_period.lob_period_extractor import LOBPeriodExtractor
from src.multivariate_hawkes_training.lob_event_combinator import LOBEventCombinator

COE_CONF_PATH = os.path.join(CONST.CONF_COE_FOLDER, CONST.COE_CONF_FILE)


def get_conf(path: str) -> MultivariateHawkesTrainingConf:
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


def save_training_df_if_not_exist(
    pair: str,
    bi_level: int,
    training_time_seconds: int,
    start_simulation_time: int,
    lob_df_for_events: pd.DataFrame,
    prefix_lob: str,
) -> None:
    coe_training_dataframe_filename = f"{prefix_lob}_{start_simulation_time}.tsv"

    coe_training_dataframe_folder = os.path.join(
        CONST.COE_TRAINING_DATAFRAMES_FOLDER,
        pair,
        f"bi_level_{bi_level}_training_seconds_{training_time_seconds}",
    )

    if not os.path.exists(coe_training_dataframe_folder):
        os.makedirs(coe_training_dataframe_folder)

    coe_training_dataframe_path = os.path.join(
        coe_training_dataframe_folder, coe_training_dataframe_filename
    )

    if not os.path.exists(coe_training_dataframe_path):
        lob_df_for_events_training = lob_df_for_events[
            lob_df_for_events["Timestamp"] < start_simulation_time
        ]

        lob_df_for_events_training.to_csv(
            coe_training_dataframe_path, sep="\t", index=False
        )


def get_simulation_methods_pair_params_dirs(pair: str) -> List[str]:
    simulation_methods_pair_dirs = [
        os.path.join(CONST.SIMULATIONS_FOLDER, d, pair)
        for d in os.listdir(CONST.SIMULATIONS_FOLDER)
        if os.path.isdir(os.path.join(CONST.SIMULATIONS_FOLDER, d))
    ]

    simulation_methods_pair_params_dirs = list()
    for folder in simulation_methods_pair_dirs:
        subfolders = [
            os.path.join(folder, sub)
            for sub in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, sub))
        ]
        simulation_methods_pair_params_dirs.extend(subfolders)

    return simulation_methods_pair_params_dirs


def get_decomposed_path(path: str) -> List[str]:

    components = []
    while path not in (os.sep, ""):
        path, folder = os.path.split(path)
        if folder:
            components.append(folder)

    components.append(path)
    components.reverse()

    return components


if __name__ == "__main__":
    coe_conf_map = get_conf(COE_CONF_PATH)
    coe_conf = CoeConf.from_dict(coe_conf_map)

    events_conf_map = get_conf(
        os.path.join(CONST.CONF_EVENTS_FOLDER, "mid_price_change_events_conf.yml")
    )
    events_conf = EventsConf.from_dict(events_conf_map)

    for pair in coe_conf.pairs:
        pair_orderbook_changes_path = os.path.join(CONST.ORDERBOOK_CHANGES_FOLDER, pair)
        periods_df = pd.read_csv(
            os.path.join(
                pair_orderbook_changes_path, CONST.SIMULATION_START_TIMESTAMPS_FILE
            )
        )

        loading_info_for_all_dfs = LoadingInfoGetter(periods_df).get_loading_info(
            lob_df_folder_path=pair_orderbook_changes_path,
            lob_df_prefix=CONST.ORDERBOOK_CHANGES_FILE_PREFIX,
        )

        for (
            loading_info,
            bi_level,
            training_time_seconds,
            simulation_period_seconds,
        ) in product(
            loading_info_for_all_dfs,
            coe_conf.bi_levels,
            coe_conf.training_time_seconds,
            coe_conf.simulation_periods_seconds,
        ):
            lob_df_loader = LOBDataLoader()
            lob_df = lob_df_loader.get_lob_dataframe(loading_info.path, bi_level)

            lob_period_extractor = LOBPeriodExtractor(lob_df)

            for start_simulation_time in loading_info.start_times:
                start_coe_training_time = start_simulation_time - training_time_seconds
                end_simulation_time = start_simulation_time + simulation_period_seconds

                lob_period = lob_period_extractor.get_lob_period(
                    start_coe_training_time, end_simulation_time
                )
                lob_df_for_events = lob_period.get_lob_df_with_timestamp_column()

                lob_df_for_events["Timestamp"] = (
                    lob_df_for_events["Timestamp"] - start_coe_training_time
                )

                lob_df_for_events = lob_df_for_events[lob_df_for_events["Return"] != 0][
                    ["Timestamp", "BaseImbalance", "Return"]
                ]

                prefix_lob = os.path.basename(loading_info.path)
                prefix_lob = os.path.splitext(prefix_lob)[0]

                save_training_df_if_not_exist(
                    pair,
                    bi_level,
                    training_time_seconds,
                    start_simulation_time,
                    lob_df_for_events,
                    prefix_lob,
                )

                lob_df_for_events_simulation = lob_df_for_events[
                    (lob_df_for_events["Timestamp"] >= start_simulation_time)
                    & (lob_df_for_events["Timestamp"] < end_simulation_time)
                ]
                lob_df_for_events_simulation["Timestamp"] = (
                    lob_df_for_events_simulation["Timestamp"] - start_simulation_time
                )

                simulation_methods_pair_params_dirs = (
                    get_simulation_methods_pair_params_dirs(pair)
                )

                for (
                    simulation_method_pair_params_dir
                ) in simulation_methods_pair_params_dirs:
                    simulation_filename = f"{prefix_lob}_{start_simulation_time}.tsv"

                    simulation_df = pd.read_csv(
                        os.path.join(
                            simulation_method_pair_params_dir, simulation_filename
                        ),
                        sep="\t",
                    )

                    simulation_df_for_coe = pd.concat(
                        [
                            simulation_df.reset_index(drop=True),
                            lob_df_for_events_simulation[
                                ["BaseImbalance", "Return"]
                            ].reset_index(drop=True),
                        ],
                        axis=1,
                    )

                    decomposed_path = get_decomposed_path(
                        simulation_method_pair_params_dir
                    )
                    training_params = decomposed_path[-1]
                    simulation_method = decomposed_path[-3]

                    coe_simulation_dataframe_folder = os.path.join(
                        CONST.COE_SIMULATION_DATAFRAMES_FOLDER,
                        simulation_method,
                        pair,
                        training_params,
                        f"simulation_seconds_{simulation_period_seconds}",
                    )

                    if not os.path.exists(coe_simulation_dataframe_folder):
                        os.makedirs(coe_simulation_dataframe_folder)

                    coe_simulation_dataframe_path = os.path.join(
                        coe_simulation_dataframe_folder, simulation_filename
                    )

                    simulation_df_for_coe.to_csv(
                        coe_simulation_dataframe_path, sep="\t", index=False
                    )
