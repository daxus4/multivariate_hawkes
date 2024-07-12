import os

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


def get_conf(path: str):
    with open(path, 'r') as f:
        conf = yaml.safe_load(f)
    return conf


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


    event_type_times_maps = list()

    for loading_info in loading_info_for_all_dfs:
        lob_df_loader = LOBDataLoader()
        lob_df = lob_df_loader.get_lob_dataframe(
            loading_info.path, multivariate_hawkes_training_conf.base_imbalance_level
        )

        lob_period_extractor = LOBPeriodExtractor(lob_df)

        for start_simulation_time in loading_info.start_times:
            start_time = (
                start_simulation_time - multivariate_hawkes_training_conf.seconds_in_a_period
            )

            end_time = start_simulation_time

            lob_period = lob_period_extractor.get_lob_period(start_time, end_time)
            lob_df_for_events = lob_period.get_lob_df_with_timestamp_column()

            lob_events_extractor = MultivariateLOBEventsExtractor(
                lob_df_for_events,
                multivariate_hawkes_training_conf.num_levels_in_a_side,
                multivariate_hawkes_training_conf.num_levels_for_which_save_events
            )

            event_type_times_map = lob_events_extractor.get_events()
            event_type_times_map = {
                key.name: value for key, value in event_type_times_map.items()
            }

            event_type_times_maps.append(event_type_times_map)

    lob_event_combinator = LOBEventCombinator(
        event_type_times_maps
    )

    event_type_times_maps = lob_event_combinator.get_event_type_times_maps_with_new_combination(
        CONST.LOB_EVENT_TO_COMBINE,
        combination_name = CONST.MID_PRICE_CHANGE_EVENT_NAME,
    )

    event_type_times_map_formatter = EventTypeTimesMapsFormatter()

    event_type_times_formatted = event_type_times_map_formatter.get_events_types_periods(
        event_type_times_maps
    )

    trainer = MultivariateHawkesTrainerWithGreedyBetaSearch(
        event_type_times_formatted, CONST.BETA_TO_TRAIN
    )
    hawkes_kernel = trainer.get_trained_kernel(CONST.BETA_VALUE_TO_TEST)

    prefix = os.path.basename(multivariate_hawkes_training_conf.period_df_path)
    prefix = os.path.join(CONST.DATA_FOLDER, CONST.HAWKES_PARAMETERS_FOLDER, prefix)
    
    np.savetxt(f'{prefix}_mu', hawkes_kernel.baseline)
    np.savetxt(f'{prefix}_alpha', hawkes_kernel.adjacency)
    np.savetxt(f'{prefix}_beta', hawkes_kernel.decays)
             
