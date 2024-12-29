from typing import List

import numpy as np
from tick.hawkes import HawkesExpKern

from optimization.lshade import lshade


class MultivariateHawkesTrainerWithLShade:
    def __init__(
        self,
        event_types_periods: List[List[np.ndarray]],
        lower_boundary: float,
        upper_boundary: float,
        initial_population_size: int = 100,
        max_generations: int = 100,
        memory_size: int = 10,
        p: float = 0.2,
        max_number_fitness_evaluations: int = 1000,
    ) -> None:
        self._event_types_periods = event_types_periods
        self._lower_boundary = lower_boundary
        self._upper_boundary = upper_boundary
        self._initial_population_size = initial_population_size
        self._max_generations = max_generations
        self._memory_size = memory_size
        self._p = p
        self._max_number_fitness_evaluations = max_number_fitness_evaluations

    def get_trained_kernel(self) -> HawkesExpKern:
        event_types_number = len(self._event_types_periods[0])

        def get_fitness(betas: np.ndarray) -> float:
            kernel = self._get_trained_hawkes(betas, event_types_number)
            return -kernel.score()
        
        optimal_betas, _ = lshade(
            get_fitness,
            event_types_number ** 2,
            self._lower_boundary,
            self._upper_boundary,
            initial_population_size=self._initial_population_size,
            max_generations=self._max_generations,
            memory_size=self._memory_size,
            p=self._p,
            max_number_fitness_evaluations=self._max_number_fitness_evaluations
        )

        return self._get_trained_hawkes(optimal_betas, event_types_number)

    def _get_trained_hawkes(self, betas: np.ndarray, event_types_number: int) -> HawkesExpKern:
        beta_matrix = betas.reshape(event_types_number, event_types_number)
        kernel = HawkesExpKern(beta_matrix, penalty='l1')
        kernel.fit(self._event_types_periods)

        return kernel




