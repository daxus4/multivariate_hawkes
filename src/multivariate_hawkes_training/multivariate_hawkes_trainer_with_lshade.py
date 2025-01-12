from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numba.typed import List as NumbaList

from optimization.hawkes_likelihood import get_likelihood_fitness_from_individual
from optimization.lshade import get_initial_random_population, lshade


@dataclass
class TrainedHawkesKernel:
    mu: np.ndarray
    alphas: np.ndarray
    betas: np.ndarray
    fitness: float


class MultivariateHawkesTrainerWithLShade:
    def __init__(
        self,
        event_types_periods: List[List[np.ndarray]],
        gene_lower_boundaries: np.ndarray,
        gene_upper_boundaries: np.ndarray,
        initial_population_size: int = 100,
        max_generations: int = 100,
        memory_size: int = 10,
        p: float = 0.2,
        max_number_fitness_evaluations: int = 1000,
        regularization_param: float = 0.01,
        instability_param: float = 100,
        training_time_duration_seconds: int = 600,
    ) -> None:
        self._event_types_periods = event_types_periods[0]
        self._gene_lower_boundaries = gene_lower_boundaries
        self._gene_upper_boundaries = gene_upper_boundaries
        self._initial_population_size = initial_population_size
        self._max_generations = max_generations
        self._memory_size = memory_size
        self._p = p
        self._max_number_fitness_evaluations = max_number_fitness_evaluations
        self._regularization_param = regularization_param
        self._instability_param = instability_param
        self._training_time_duration_seconds = training_time_duration_seconds

    def get_event_count_for_event_types(self) -> List[int]:
        return [len(event_type) for event_type in self._event_types_periods]

    def convert_individual_to_kernel(
        self, individual: np.ndarray, fitness: float
    ) -> TrainedHawkesKernel:
        num_event_types = len(self._event_types_periods)

        mu = individual[:num_event_types]

        alphas = individual[
            num_event_types : num_event_types + num_event_types * num_event_types
        ].reshape(num_event_types, num_event_types)

        betas = individual[
            num_event_types + num_event_types * num_event_types :
        ].reshape(num_event_types, num_event_types)

        return TrainedHawkesKernel(mu, alphas, betas, fitness)

    def get_trained_kernel(self) -> TrainedHawkesKernel:
        initial_population = get_initial_random_population(
            self._gene_upper_boundaries,
            self._initial_population_size,
            np.array(self.get_event_count_for_event_types()),
            self._training_time_duration_seconds,
        )

        best_individual = lshade(
            self._gene_lower_boundaries,
            self._gene_upper_boundaries,
            initial_population=initial_population,
            max_generations=self._max_generations,
            memory_size=self._memory_size,
            p=self._p,
            max_number_fitness_evaluations=self._max_number_fitness_evaluations,
            end_time=self._training_time_duration_seconds,
            events_times=NumbaList(self._event_types_periods),
            regularization_param=self._regularization_param,
            instability_param=self._instability_param,
        )

        fitness = get_likelihood_fitness_from_individual(
            best_individual,
            self._training_time_duration_seconds,
            NumbaList(self._event_types_periods),
            self._regularization_param,
            self._instability_param,
        )

        return self.convert_individual_to_kernel(best_individual, fitness)
