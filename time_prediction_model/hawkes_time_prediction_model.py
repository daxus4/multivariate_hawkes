from typing import Any, Dict, List

import numpy as np
import tick.hawkes as hk

from time_prediction_model.period_for_simulation import PeriodForSimulation
from time_prediction_model.time_prediction_model import TimePredictionModel


class HawkesTimePredictionModel(TimePredictionModel):
    def __init__(
        self,
        params: Dict[str, Any],
        prediction_period_duration: float,
        seed: int = 1039,
    ) -> None:
        super().__init__(params)
        self._prediction_period_duration = prediction_period_duration
        self._seed = seed

    def predict_next_event_time_from_current_time(
        self,
        period_for_simulation: PeriodForSimulation,
        current_time: float,
    ) -> Dict[str, float]:
        event_times = period_for_simulation.get_ordered_event_times()

        simulated_hawkes = self._get_hawkes_simulation(
            event_times,
            current_time + self._prediction_period_duration,
            self._seed
        )

        predicted_timestamps: List[np.ndarray] = simulated_hawkes.timestamps

        return {
            event_type: predicted_timestamps[i]
            for i, event_type in enumerate(period_for_simulation.event_types_order)
        }

    def _get_hawkes_simulation(
        self,
        event_times: List[np.ndarray],
        max_simulated_time: float,
        seed: int,
    ) -> hk.SimuHawkesExpKernels:
        sim_hawkes = hk.SimuHawkesExpKernels(
            adjacency=self._get_alpha_converted_for_simulation(),
            decays=self._get_beta_converted_for_simulation(),
            baseline=self._get_mu_converted_for_simulation(),
            end_time=max_simulated_time, # vedere se usare max_jumps al posto di end_time
            seed=seed
        )
        sim_hawkes.track_intensity(1)

        sim_hawkes.set_timestamps(event_times, max_simulated_time)
        sim_hawkes.end_time = max_simulated_time

        sim_hawkes.simulate()

        return sim_hawkes

    def _get_alpha_converted_for_simulation(self) -> np.ndarray:
        return self._parameters['alpha']
    
    def _get_beta_converted_for_simulation(self) -> List[np.ndarray]:
        return self._parameters['beta']
    
    def _get_mu_converted_for_simulation(self) -> float:
        return self._parameters['mu']