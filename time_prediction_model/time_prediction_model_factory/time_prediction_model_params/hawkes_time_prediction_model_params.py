import os
from typing import List

from src.constants import ORDER_OF_EVENT_TYPES_FILE
from time_prediction_model.time_prediction_model_factory.time_prediction_model_params.poisson_time_prediction_model_params import (
    PoissonTimePredictionModelParams,
)


class HawkesTimePredictionModelParams(PoissonTimePredictionModelParams):
    @classmethod
    def from_files_indications(
        cls, 
        folder_path: str,
        file_start_registration_time: int,
        file_start_simulation_time: int
    ) -> 'HawkesTimePredictionModelParams':
        order_events = cls._get_order_events(folder_path)

        prefix_filename = cls._get_correct_parameter_file_prefix(
            folder_path,
            file_start_registration_time,
            file_start_simulation_time
        )

        params_dict = cls._get_poisson_params_map(
            folder_path, prefix_filename
        )

        alpha = cls._get_param_from_file(
            folder_path,
            prefix_filename,
            "alpha"
        )

        beta = cls._get_param_from_file(
            folder_path,
            prefix_filename,
            "beta"
        )

        params_dict.update({
            "event_types_order": order_events,
            "alpha": alpha,
            "beta": beta
        })

        return cls(params_dict)

    @classmethod
    def _get_order_events(cls, folder_path):
        order_event_file = os.path.join(
            folder_path, ORDER_OF_EVENT_TYPES_FILE
        )
        return cls._get_strings_from_file(order_event_file)

    @classmethod
    def _get_strings_from_file(cls, file_path: str) -> List[str]:
        with open(file_path, 'r') as file:
            lines = file.readlines()  # Read all lines into a list
            lines = [line.strip() for line in lines]  # Remove trailing newlines or spaces
        return lines