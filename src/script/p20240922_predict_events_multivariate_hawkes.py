import os
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

import src.multivariate_hawkes_training.constants as CONST
from src.conf.multivariate_hawkes_training_conf import MultivariateHawkesTrainingConf
from src.events_extractor.multivariate_lob_events_extractor import (
    MultivariateLOBEventsExtractor,
)
from src.lob_data_loader.loading_info_getter import LoadingInfoGetter
from src.lob_data_loader.lob_data_loader import LOBDataLoader
from src.lob_period.lob_period_extractor import LOBPeriodExtractor
from src.multivariate_hawkes_training.event_type_times_maps_formatter import (
    EventTypeTimesMapsFormatter,
)
from src.multivariate_hawkes_training.lob_event_combinator import LOBEventCombinator
from src.multivariate_hawkes_training.multivariate_hawkes_trainer_with_greedy_beta_search import (
    MultivariateHawkesTrainerWithGreedyBetaSearch,
)


def get_conf(path: str) -> MultivariateHawkesTrainingConf:
    with open(path, 'r') as f:
        conf = yaml.safe_load(f)
    return conf


def get_event_type_times_maps_with_combined_types(
    event_type_times_map: List[Dict[str, np.ndarray]],
    combined_name_events_to_combine_map: Dict[str, List[str]]
) -> List[Dict[str, np.ndarray]]:
    
    lob_event_combinator = LOBEventCombinator([event_type_times_map])

    for combination_name, lob_events_to_combine in combined_name_events_to_combine_map.items():
        event_type_times_maps = lob_event_combinator.get_event_type_times_maps_with_new_combination(
            lob_events_to_combine,
            combination_name=combination_name,
        )
        lob_event_combinator.event_type_times_maps = event_type_times_maps
    
    return event_type_times_maps

def get_event_type_times_maps_filtered(
    event_type_times_map: List[Dict[str, np.ndarray]],
    events_to_compute: List[str]
) -> List[Dict[str, np.ndarray]]:
    return [
        {key: value for key, value in event_type_times.items() if key in events_to_compute}
        for event_type_times in event_type_times_map
    ]

if __name__ == "__main__":
    multivariate_hawkes_training_conf_map = get_conf(
        os.path.join(CONST.CONF_FOLDER, CONST.MULTIVARIATE_HAWKES_TRAINING_CONF_FILE)
    )

    multivariate_hawkes_training_conf = MultivariateHawkesTrainingConf.from_dict(
        multivariate_hawkes_training_conf_map
    )

    periods_df = pd.read_csv(multivariate_hawkes_training_conf.period_df_path)

    loading_info_for_all_dfs = LoadingInfoGetter(periods_df).get_loading_info(
        lob_df_folder_path=multivariate_hawkes_training_conf.lob_df_folder_path,
        lob_df_prefix=multivariate_hawkes_training_conf.lob_df_file_prefix,
    )

    for loading_info in loading_info_for_all_dfs:
        lob_df_loader = LOBDataLoader()
        lob_df = lob_df_loader.get_lob_dataframe(
            loading_info.path, multivariate_hawkes_training_conf.base_imbalance_level
        )

        lob_period_extractor = LOBPeriodExtractor(lob_df)

        for start_simulation_time in loading_info.start_times:
            METTERE STAR_TIME E END_TIME PER ESTRARRE
            START_TIME = START SIMULATION - WARMUP PERIOD DURATION
            END_TIME = START SIMULATION + SIMULATION END

            lob_period = lob_period_extractor.get_lob_period(start_time, end_time)
            lob_df_for_events = lob_period.get_lob_df_with_timestamp_column()

            lob_df_for_events['Timestamp'] = lob_df_for_events['Timestamp'] * 1000

            lob_events_extractor = MultivariateLOBEventsExtractor(
                lob_df_for_events,
                multivariate_hawkes_training_conf.num_levels_in_a_side,
                multivariate_hawkes_training_conf.num_levels_for_which_save_events
            )

            event_type_times_map = lob_events_extractor.get_events()
            event_type_times_map = {
                key.name: value for key, value in event_type_times_map.items()
            }

            event_type_times_maps = get_event_type_times_maps_with_combined_types(
                event_type_times_map,
                multivariate_hawkes_training_conf.combined_event_types_map
            )

            event_type_times_maps = get_event_type_times_maps_filtered(
                event_type_times_maps,
                multivariate_hawkes_training_conf.events_to_compute
            )

            event_type_times_map_formatter = EventTypeTimesMapsFormatter()

            event_type_times_formatted = event_type_times_map_formatter.get_events_types_periods(
                event_type_times_maps,
                multivariate_hawkes_training_conf.events_to_compute
            )

            event_type_times_formatted_in_seconds = [
                [times / 1000 for times in event_type_times]
                for event_type_times in event_type_times_formatted
            ]
