from typing import Any, Dict, List, Tuple

import numpy as np

from time_prediction_model.hawkes_time_prediction_model import HawkesTimePredictionModel


class PoissonTimePredictionModel(HawkesTimePredictionModel):
    def _get_alpha_converted_for_simulation(self) -> np.ndarray:
        return np.array([[0]])
    
    def _get_beta_converted_for_simulation(self) -> List[np.ndarray]:
        return [np.array([0])]
