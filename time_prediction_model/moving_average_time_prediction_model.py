from typing import List

import numpy as np

from time_prediction_model.time_prediction_model import TimePredictionModel


class MovingAverageTimePredictionModel(TimePredictionModel):
    def predict_next_event_time_from_current_time(
        self,
        event_times: List[np.ndarray],
        current_time: float,
    ) -> float:
        event_times = event_times[0]
        event_times = event_times[
            event_times >= current_time - self._parameters["window_duration_seconds"]
        ]

        mean_next_event_time_jump = self._get_mean_next_event_time_jump(event_times)

        next_event_time = event_times[-1] + mean_next_event_time_jump

        while next_event_time < current_time:
            next_event_time += mean_next_event_time_jump

        return next_event_time

    def _get_mean_next_event_time_jump(self, event_times: np.ndarray) -> float:
        return np.mean(np.diff(event_times))