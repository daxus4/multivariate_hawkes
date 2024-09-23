from typing import List

import numpy as np

from time_prediction_model.time_prediction_model import TimePredictionModel


class NaiveTimePredictionModel(TimePredictionModel):
    def predict_next_event_time_from_current_time(
        self,
        event_times: List[np.ndarray],
        current_time: float,
    ) -> float:
        next_event_time = event_times[0][-1] + self._parameters["next_event_time_jump"]

        while next_event_time < current_time:
            next_event_time += self._parameters["next_event_time_jump"]

        return next_event_time