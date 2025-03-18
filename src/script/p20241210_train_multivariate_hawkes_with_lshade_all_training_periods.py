import os
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

import src.constants as CONST
from src.conf.events_conf.events_conf import EventsConf
from src.conf.training.training_conf import TrainingConf
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
from src.multivariate_hawkes_training.multivariate_hawkes_trainer_with_lshade import (
    MultivariateHawkesTrainerWithLShade,
)

mu_lower_bound = 0.0
mu_upper_bound = 1.2
rho_lower_bound = 0.0
rho_upper_bound = 1
beta_lower_bound = 0.0
beta_upper_bound = 100.0

initial_population_size = 30_000
max_generations = 250
memory_size = 70
p = 0.33
max_number_fitness_evaluations = 500_000
regularization_param = 0
instability_param = 500
seed = 444
number_of_best_individuals_without_neighborhood = 5
relative_tolerance_for_best_individuals_neighborhood = 0.05
absolute_tolerance_for_best_individuals_neighborhood = 0.05


CONF_EVENTS_FILENAME = "mid_price_increase_and_decrease_events_conf.yml"
CONF_TRAINING_FILENAME = "training_conf.yml"


def check_influence(event_x_df, event_y_df, median_offset):
    influenced_count = 0

    # For each event of type X, check if there is at least one event of type Y within the median offset
    for _, x_row in event_x_df.iterrows():
        # Get the time of event X
        x_time = x_row["Event Time"]

        # Check for events of type Y occurring within the median offset
        y_in_range = event_y_df[
            (event_y_df["Event Time"] > x_time)
            & (event_y_df["Event Time"] < x_time + median_offset)
        ]

        # If at least one Y event is found within the offset range, count it
        if not y_in_range.empty:
            influenced_count += 1

    # Return the proportion of events of type X influenced by event Y
    proportion_influenced = influenced_count / len(event_x_df)
    return proportion_influenced


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
        training_time: {"file": [], "likelihood": []}
        for training_time in training_conf.seconds_in_a_period
    }

    for loading_info in loading_info_for_all_dfs:
        lob_df_loader = LOBDataLoader()
        lob_df = lob_df_loader.get_lob_dataframe(loading_info.path, 10)

        lob_period_extractor = LOBPeriodExtractor(lob_df)

        for start_simulation_time in loading_info.start_times:
            for training_time_seconds in training_conf.seconds_in_a_period:
                start_time = start_simulation_time - training_time_seconds

                end_time = start_simulation_time

                lob_period = lob_period_extractor.get_lob_period(start_time, end_time)
                lob_df_for_events = lob_period.get_lob_df_with_timestamp_column()

                lob_df_for_events["Timestamp"] = lob_df_for_events["Timestamp"] * 1000

                lob_events_extractor = MultivariateLOBEventsExtractor(
                    lob_df_for_events,
                    events_conf.num_levels_in_a_side,
                    events_conf.num_levels_for_which_save_events,
                )

                event_type_times_map = lob_events_extractor.get_events()
                event_type_times_map = {
                    key.name: value for key, value in event_type_times_map.items()
                }

                event_type_times_maps = get_event_type_times_maps_with_combined_types(
                    event_type_times_map, events_conf.combined_event_types_map
                )

                event_type_times_maps = get_event_type_times_maps_filtered(
                    event_type_times_maps, events_conf.events_to_compute
                )

                events_df = pd.DataFrame(
                    [
                        (k, v)
                        for k, values in event_type_times_maps[0].items()
                        for v in values
                    ],
                    columns=["Event Type", "Event Time"],
                )
                events_df = events_df.sort_values("Event Time", ascending=True)
                events_df["Offset Shared"] = events_df["Event Time"].diff()

                influence_matrix = pd.DataFrame(
                    index=events_conf.events_to_compute,
                    columns=events_conf.events_to_compute,
                    dtype=float,
                )

                median_offset = events_df["Offset Shared"].median()

                for event_x in events_conf.events_to_compute:
                    for event_y in events_conf.events_to_compute:
                        proportion = check_influence(
                            events_df[events_df["Event Type"] == event_x],
                            events_df[events_df["Event Type"] == event_y],
                            median_offset,
                        )
                        influence_matrix.at[event_y, event_x] = proportion

                influence_matrix = influence_matrix.to_numpy()

                correlated_couples = influence_matrix > 0.06
                correlated_couples = correlated_couples.flatten()

                mu_counts = len(events_conf.events_to_compute)
                rho_counts = len(events_conf.events_to_compute) ** 2
                beta_counts = len(events_conf.events_to_compute) ** 2

                lower_bounds = np.concatenate(
                    (
                        mu_lower_bound * np.ones(mu_counts),
                        np.where(
                            correlated_couples,
                            rho_lower_bound * np.ones(rho_counts),
                            np.zeros(rho_counts),
                        ),
                        np.where(
                            correlated_couples,
                            beta_lower_bound * np.ones(rho_counts),
                            100 * np.ones(rho_counts),
                        ),
                    )
                )

                upper_bounds = np.concatenate(
                    (
                        mu_upper_bound * np.ones(mu_counts),
                        np.where(
                            correlated_couples,
                            rho_upper_bound * np.ones(rho_counts),
                            np.zeros(rho_counts),
                        ),
                        np.where(
                            correlated_couples,
                            beta_upper_bound * np.ones(rho_counts),
                            100 * np.ones(rho_counts),
                        ),
                    )
                )

                fixed_parameters_values = np.concatenate(
                    (
                        np.full(len(events_conf.events_to_compute), np.nan),
                        np.where(
                            correlated_couples,
                            np.nan * np.ones(rho_counts),
                            np.zeros(rho_counts),
                        ),
                        np.where(
                            correlated_couples,
                            np.nan * np.ones(rho_counts),
                            beta_upper_bound * np.ones(rho_counts),
                        ),
                    )
                )

                fixed_parameters_values = np.full(mu_counts + rho_counts * 2, np.nan)

                event_type_times_map_formatter = EventTypeTimesMapsFormatter()

                event_type_times_formatted = (
                    event_type_times_map_formatter.get_events_types_periods(
                        event_type_times_maps, events_conf.events_to_compute
                    )
                )

                event_type_times_formatted_in_seconds = [
                    [times / 1000 for times in event_type_times]
                    for event_type_times in event_type_times_formatted
                ]

                trainer = MultivariateHawkesTrainerWithLShade(
                    event_type_times_formatted_in_seconds,
                    lower_bounds,
                    upper_bounds,
                    initial_population_size,
                    max_generations,
                    memory_size,
                    p,
                    max_number_fitness_evaluations,
                    regularization_param,
                    instability_param,
                    training_time_seconds,
                    fixed_parameters_values,
                )
                params_dir = os.path.join(
                    CONST.TRAINED_PARAMS_FOLDER,
                    CONST.MULTIVARIATE_HAWKES,
                    training_conf.pair,
                    "lshade_training_time_my_" + str(training_time_seconds),
                )

                if not os.path.exists(params_dir):
                    os.makedirs(params_dir, exist_ok=True)

                prefix = os.path.basename(loading_info.path)
                prefix = os.path.splitext(prefix)[0]
                prefix = os.path.join(params_dir, prefix)

                logs_dir = f"{prefix}_{start_simulation_time}_logs"
                if not os.path.exists(logs_dir):
                    os.makedirs(logs_dir, exist_ok=True)

                np.savetxt(f"{logs_dir}\\fixed_params.txt", fixed_parameters_values)

                hawkes_kernel = trainer.get_trained_kernel(logs_dir)

                training_time_file_likelihood_map[training_time_seconds]["file"].append(
                    f"{prefix}_{start_simulation_time}"
                )
                training_time_file_likelihood_map[training_time_seconds][
                    "likelihood"
                ].append(hawkes_kernel.fitness)

                np.savetxt(f"{prefix}_{start_simulation_time}_mu.txt", hawkes_kernel.mu)
                np.savetxt(
                    f"{prefix}_{start_simulation_time}_rho.txt", hawkes_kernel.rhos
                )
                np.savetxt(
                    f"{prefix}_{start_simulation_time}_beta.txt", hawkes_kernel.betas
                )

    for training_time_seconds in training_conf.seconds_in_a_period:
        params_dir = os.path.join(
            CONST.TRAINED_PARAMS_FOLDER,
            CONST.MULTIVARIATE_HAWKES,
            training_conf.pair,
            "lshade_training_time_my_" + str(training_time_seconds),
        )
        with open(
            os.path.join(params_dir, CONST.ORDER_OF_EVENT_TYPES_FILE), "w"
        ) as file:
            file.writelines(f"{item}\n" for item in events_conf.events_to_compute)

        df = pd.DataFrame(training_time_file_likelihood_map[training_time_seconds])
        df.to_csv(
            os.path.join(params_dir, CONST.LIKELIHOODS_FILE), sep="\t", index=False
        )
