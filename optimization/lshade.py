import numpy as np
from numba import njit
from numba.typed import List

from optimization.hawkes_likelihood import (
    get_likelihood_fitness_from_individual,
    get_spectral_radius,
)


@njit
def get_initial_random_population_my(
    gene_upper_boundaries: np.ndarray,
    initial_population_size: int,
    event_counts_for_type: np.ndarray,
    training_time_in_seconds: int,
    fixed_parameters_values: np.ndarray,
) -> np.ndarray:
    """
    Returns a NxD array of random individuals, where N is the initial population size
    and D is the problem dimensionality. Supposing that the number of event types is
    2, D is formed by 2 mu, 4 rho (2x2 matrix) and 4 beta (2x2 matrix) parameters.
    """
    type_count = len(event_counts_for_type)
    square_matrix_size = type_count * type_count
    parameters_count = 2 * square_matrix_size + type_count

    initial_population = np.zeros((initial_population_size, parameters_count))

    # fill the initial population with random mu values
    for i in range(type_count):
        initial_population[:, i] = np.random.uniform(
            0,
            gene_upper_boundaries[i],
            initial_population_size,
        )

    # fill the initial population with random rho values
    for i, j in [(i, j) for i in range(type_count) for j in range(type_count)]:
        initial_population[:, type_count + i * type_count + j] = np.random.uniform(
            0,
            gene_upper_boundaries[type_count + i * type_count + j],
            initial_population_size,
        )

    # rescale the rho values for individuals with spectral radius > 1
    for individual_index in range(initial_population_size):
        if individual_index % 1000 == 0:
            print(individual_index)
        rho_values = initial_population[
            individual_index, type_count : type_count + square_matrix_size
        ].reshape((type_count, type_count))

        kernel_spectral_radius = get_spectral_radius(rho_values)
        while kernel_spectral_radius > 1:
            dimension_to_reduce = np.random.randint(0, square_matrix_size)
            initial_population[individual_index, type_count + dimension_to_reduce] = (
                np.random.exponential(scale=0.09)
            )

            # select the maximum element of the vector < 1 and increase it by 5%
            max_elem_lower_than_one = np.max(
                initial_population[
                    individual_index, type_count : type_count + square_matrix_size
                ][
                    initial_population[
                        individual_index, type_count : type_count + square_matrix_size
                    ]
                    < 1
                ]
            )
            max_elem_lower_than_one_index = np.argmax(
                initial_population[
                    individual_index, type_count : type_count + square_matrix_size
                ][
                    initial_population[
                        individual_index, type_count : type_count + square_matrix_size
                    ]
                    < 1
                ]
            )

            initial_population[
                individual_index, type_count + max_elem_lower_than_one_index
            ] = (max_elem_lower_than_one * 1.05)

            rho_values = initial_population[
                individual_index, type_count : type_count + square_matrix_size
            ].reshape((type_count, type_count))
            kernel_spectral_radius = get_spectral_radius(rho_values)

        if kernel_spectral_radius < 0.45:
            random_factor = np.random.uniform(0.5, 0.99)

            rescaled_rho_matrix = rho_values * (random_factor / kernel_spectral_radius)

            initial_population[
                individual_index, type_count : type_count + square_matrix_size
            ] = rescaled_rho_matrix.flatten()

    # fill the initial population with random beta values
    for i, j in [(i, j) for i in range(type_count) for j in range(type_count)]:
        bernoulli_samples = np.random.binomial(n=1, p=0.5, size=initial_population_size)

        random_values = np.where(
            bernoulli_samples == 0,
            np.random.uniform(0, 1, size=initial_population_size),
            np.random.uniform(0, 100, size=initial_population_size),
        )

        initial_population[:, type_count + square_matrix_size + i * type_count + j] = (
            random_values
        )

    fixed_parameters_indices = np.where(np.isnan(fixed_parameters_values))[0]
    initial_population = initial_population[:, fixed_parameters_indices]

    return initial_population


@njit
def get_initial_random_population(
    gene_upper_boundaries: np.ndarray,
    initial_population_size: int,
    event_counts_for_type: np.ndarray,
    training_time_in_seconds: int,
    fixed_parameters_values: np.ndarray,
) -> np.ndarray:
    """
    Returns a NxD array of random individuals, where N is the initial population size
    and D is the problem dimensionality. Supposing that the number of event types is
    2, D is formed by 2 mu, 4 rho (2x2 matrix) and 4 beta (2x2 matrix) parameters.
    """
    type_count = len(event_counts_for_type)
    square_matrix_size = type_count * type_count
    parameters_count = 2 * square_matrix_size + type_count

    initial_population = np.zeros((initial_population_size, parameters_count))

    # fill the initial population with random mu values
    for i in range(type_count):
        current_mu_max_value = 0.2 * event_counts_for_type[i] / training_time_in_seconds
        initial_population[:, i] = np.random.uniform(
            0,
            min(gene_upper_boundaries[i], current_mu_max_value),
            initial_population_size,
        )

    # fill the initial population with random rho values
    for i, j in [(i, j) for i in range(type_count) for j in range(type_count)]:
        current_rho_max_value = min(
            0.2 * event_counts_for_type[i] / event_counts_for_type[j], 0.5
        )

        initial_population[:, type_count + i * type_count + j] = np.random.uniform(
            0,
            min(
                gene_upper_boundaries[type_count + i * type_count + j],
                current_rho_max_value,
            ),
            initial_population_size,
        )

    # rescale the rho values for individuals with spectral radius > 1
    for individual_index in range(initial_population_size):
        rho_values = initial_population[
            individual_index, type_count : type_count + square_matrix_size
        ].reshape((type_count, type_count))

        kernel_spectral_radius = get_spectral_radius(rho_values)
        if kernel_spectral_radius > 1:
            random_factor = np.random.uniform(0.5, 0.99)

            rescaled_rho_matrix = rho_values * (random_factor / kernel_spectral_radius)

            initial_population[
                individual_index, type_count : type_count + square_matrix_size
            ] = rescaled_rho_matrix.flatten()

    # fill the initial population with random beta values
    for i, j in [(i, j) for i in range(type_count) for j in range(type_count)]:
        bernoulli_samples = np.random.binomial(
            n=1, p=0.33, size=initial_population_size
        )

        random_values = np.where(
            bernoulli_samples == 0,
            np.random.uniform(0, 1, size=initial_population_size),
            np.random.uniform(0, 100, size=initial_population_size),
        )

        initial_population[:, type_count + square_matrix_size + i * type_count + j] = (
            random_values
        )

    fixed_parameters_indices = np.where(np.isnan(fixed_parameters_values))[0]
    initial_population = initial_population[:, fixed_parameters_indices]

    return initial_population


@njit
def weighted_lehmar_mean(succeding_rates, individual_fitnesses, trial_fitnesses):
    improvements = individual_fitnesses - trial_fitnesses
    total_improvement = np.sum(improvements)
    weights = improvements / total_improvement

    lehmar_mean = np.sum(weights * succeding_rates**2) / np.sum(
        weights * succeding_rates
    )

    return lehmar_mean


@njit
def get_sample_from_cauchy(location: float, scale: float) -> float:
    quantile = np.random.rand()
    random_number = location + scale * np.tan(np.pi * quantile - np.pi / 2.0)
    return random_number


@njit
def get_clipped_number(number: float, lower_bound: float, upper_bound: float) -> float:
    if number < lower_bound:
        return lower_bound
    elif number > upper_bound:
        return upper_bound
    else:
        return number


@njit
def setdiff1d(array1, elem):
    # Create a boolean mask for elements in array1 that are not in array2
    result_mask = np.ones(array1.shape[0], dtype=np.bool_)
    for i in range(array1.shape[0]):
        if array1[i] == elem:
            result_mask[i] = False

    return array1[result_mask]


@njit
def remove_rows(array, rows_to_remove):
    # Sort the indices to remove (optional but recommended for consistency)
    rows_to_remove = np.sort(rows_to_remove)

    mask = np.ones(array.shape[0], dtype=np.bool_)
    mask[rows_to_remove] = False

    return array[mask]


@njit
def lshade(
    gene_lower_boundaries,
    gene_upper_boundaries,
    initial_population,
    max_generations,
    memory_size,
    p,
    max_number_fitness_evaluations,
    end_time,
    events_times,
    regularization_param,
    instability_param,
) -> np.ndarray:

    population_size = initial_population.shape[0]
    archive_size = initial_population.shape[0]
    problem_dimensionality = initial_population.shape[1]
    archive = np.zeros((archive_size, problem_dimensionality))
    archive_fitnesses = np.empty(archive_size)
    archive_fitnesses.fill(np.nan)

    memory = np.zeros((memory_size, 2)) + 0.5
    MEMORY_MCR_INDEX = 0
    MEMORY_MMR_INDEX = 1

    previous_gen_population = initial_population.copy()

    crossing_rates = np.zeros(population_size)
    mutation_rates = np.zeros(population_size)

    previous_gen_individuals_fitnesses = np.empty(population_size)
    for individual_index in range(population_size):
        previous_gen_individuals_fitnesses[individual_index] = (
            get_likelihood_fitness_from_individual(
                previous_gen_population[individual_index],
                end_time,
                events_times,
                regularization_param,
                instability_param,
            )
        )

    current_number_of_fitness_evaluations = population_size
    memory_index_to_update = 0

    min_number_of_individuals = 4
    generation_number = 1

    best_individual_fitness_index = np.argmin(previous_gen_individuals_fitnesses)
    global_best_individual = previous_gen_population[best_individual_fitness_index]
    global_best_individual_fitness = previous_gen_individuals_fitnesses[
        best_individual_fitness_index
    ]

    while (generation_number <= max_generations) and (
        current_number_of_fitness_evaluations < max_number_fitness_evaluations
    ):
        succeding_crossover_rates = np.empty(population_size)
        succeding_crossover_rates.fill(np.nan)
        succeding_mutation_rates = np.empty(population_size)
        succeding_mutation_rates.fill(np.nan)
        succeding_trials = np.empty((population_size, problem_dimensionality))
        succeding_trial_fitnesses = np.empty(population_size)
        succeding_trials_count = 0
        trail_individuals = np.empty((population_size, problem_dimensionality))
        trial_fitnesses = np.empty(population_size)
        are_individuals_surpassed = np.zeros(population_size)
        next_gen_individuals = previous_gen_population.copy()
        next_gen_individuals_fitnesses = previous_gen_individuals_fitnesses.copy()

        for individual_index in range(population_size):
            memory_index = np.random.randint(0, memory_size)
            if np.isnan(memory[memory_index, MEMORY_MCR_INDEX]):
                crossing_rates[individual_index] = 0
            else:
                crossing_rates[individual_index] = get_clipped_number(
                    np.random.normal(memory[memory_index, MEMORY_MCR_INDEX], 0.1),
                    0.0,
                    1.0,
                )

            generated_mutation_rate = 0
            while generated_mutation_rate <= 0:
                generated_mutation_rate = get_sample_from_cauchy(
                    memory[memory_index, MEMORY_MMR_INDEX], 0.1
                )
                if generated_mutation_rate > 1:
                    generated_mutation_rate = 1

            mutation_rates[individual_index] = generated_mutation_rate

            number_of_best_individuals = round(population_size * p)
            random_best_individual_index = np.random.randint(
                0, number_of_best_individuals
            )

            best_individuals_indices = np.argpartition(
                previous_gen_individuals_fitnesses, number_of_best_individuals
            )[:number_of_best_individuals]

            best_individual = previous_gen_population[
                best_individuals_indices[random_best_individual_index]
            ]

            number_individuals_in_archive = np.sum(~np.isnan(archive_fitnesses))
            individual_indices_not_current = setdiff1d(
                np.arange(population_size + number_individuals_in_archive),
                individual_index,
            )

            random_individual_indices = np.random.choice(
                individual_indices_not_current, 2, replace=False
            )

            random_individual_1 = (
                previous_gen_population[random_individual_indices[0]]
                if random_individual_indices[0] < population_size
                else archive[random_individual_indices[0] - population_size]
            )
            random_individual_2 = (
                previous_gen_population[random_individual_indices[1]]
                if random_individual_indices[1] < population_size
                else archive[random_individual_indices[1] - population_size]
            )
            current_individual = previous_gen_population[individual_index]

            mutant = (
                current_individual
                + mutation_rates[individual_index]
                * (best_individual - current_individual)
                + mutation_rates[individual_index]
                * (random_individual_1 - random_individual_2)
            )

            for gene_index in range(problem_dimensionality):
                if mutant[gene_index] < gene_lower_boundaries[gene_index]:
                    mutant[gene_index] = (
                        gene_lower_boundaries[gene_index]
                        + current_individual[gene_index]
                    ) / 2
                elif mutant[gene_index] > gene_upper_boundaries[gene_index]:
                    mutant[gene_index] = (
                        gene_upper_boundaries[gene_index]
                        + current_individual[gene_index]
                    ) / 2

            random_gene_index_to_mutate = np.random.randint(0, problem_dimensionality)
            crossover_gene_index = (
                np.random.rand(problem_dimensionality)
                < crossing_rates[individual_index]
            )
            crossover_gene_index[random_gene_index_to_mutate] = True

            trail_individual = np.where(
                crossover_gene_index, mutant, current_individual
            )
            trail_individuals[individual_index] = trail_individual
            trial_fitnesses[individual_index] = get_likelihood_fitness_from_individual(
                trail_individual,
                end_time,
                events_times,
                regularization_param,
                instability_param,
            )
            current_number_of_fitness_evaluations = (
                current_number_of_fitness_evaluations + 1
            )

        for individual_index in range(population_size):

            if (
                trial_fitnesses[individual_index]
                <= previous_gen_individuals_fitnesses[individual_index]
            ):
                next_gen_individuals[individual_index] = trail_individuals[
                    individual_index
                ]
                next_gen_individuals_fitnesses[individual_index] = trial_fitnesses[
                    individual_index
                ]

                if (
                    trial_fitnesses[individual_index]
                    < previous_gen_individuals_fitnesses[individual_index]
                ):
                    if number_individuals_in_archive < archive_size:
                        archive[number_individuals_in_archive] = trail_individuals[
                            individual_index
                        ]
                        archive_fitnesses[number_individuals_in_archive] = (
                            trial_fitnesses[individual_index]
                        )
                    else:
                        random_element_index = np.random.randint(0, archive_size)
                        archive[random_element_index] = trail_individuals[
                            individual_index
                        ]
                        archive_fitnesses[random_element_index] = trial_fitnesses[
                            individual_index
                        ]

                    succeding_mutation_rates[succeding_trials_count] = mutation_rates[
                        individual_index
                    ]
                    succeding_crossover_rates[succeding_trials_count] = crossing_rates[
                        individual_index
                    ]

                    succeding_trials[succeding_trials_count] = trail_individuals[
                        individual_index
                    ]
                    succeding_trial_fitnesses[succeding_trials_count] = trial_fitnesses[
                        individual_index
                    ]

                    are_individuals_surpassed[individual_index] = 1

                    succeding_trials_count += 1

                    if (
                        trial_fitnesses[individual_index]
                        < global_best_individual_fitness
                    ):
                        global_best_individual = trail_individuals[individual_index]
                        global_best_individual_fitness = trial_fitnesses[
                            individual_index
                        ]

        if succeding_trials_count > 0:
            if np.nanmax(succeding_mutation_rates) <= 0 or np.isnan(
                memory[memory_index_to_update, MEMORY_MCR_INDEX]
            ):
                memory[memory_index_to_update, MEMORY_MCR_INDEX] = np.nan
            else:
                memory[memory_index_to_update, MEMORY_MCR_INDEX] = weighted_lehmar_mean(
                    succeding_crossover_rates[:succeding_trials_count],
                    previous_gen_individuals_fitnesses[are_individuals_surpassed == 1],
                    succeding_trial_fitnesses[:succeding_trials_count],
                )

            memory[memory_index_to_update, MEMORY_MMR_INDEX] = weighted_lehmar_mean(
                succeding_mutation_rates[:succeding_trials_count],
                previous_gen_individuals_fitnesses[are_individuals_surpassed == 1],
                succeding_trial_fitnesses[:succeding_trials_count],
            )

            memory_index_to_update = (memory_index_to_update + 1) % memory_size

        next_gen_population_size = round(
            (
                (min_number_of_individuals - initial_population.shape[0])
                / max_number_fitness_evaluations
            )
            * current_number_of_fitness_evaluations
            + initial_population.shape[0]
        )

        if next_gen_population_size < population_size:
            if next_gen_population_size < min_number_of_individuals:
                next_gen_individuals = np.empty((0, 2))
                next_gen_individuals_fitnesses = np.empty(0)
            else:
                number_of_individuals_to_delete = (
                    population_size - next_gen_population_size
                )
                worst_individuals_indices = np.argpartition(
                    next_gen_individuals_fitnesses, -number_of_individuals_to_delete
                )[-number_of_individuals_to_delete:]

                next_gen_individuals = remove_rows(
                    next_gen_individuals, worst_individuals_indices
                )
                next_gen_individuals_fitnesses = np.delete(
                    next_gen_individuals_fitnesses, worst_individuals_indices
                )
                population_size = next_gen_population_size

        previous_gen_population = next_gen_individuals
        previous_gen_individuals_fitnesses = next_gen_individuals_fitnesses
        generation_number = generation_number + 1

    return global_best_individual


@njit
def logical_or(arr1, arr2):
    result = np.empty(arr1.shape, dtype=np.bool_)
    for i in range(arr1.size):
        result[i] = bool(arr1[i]) | bool(arr2[i])
    return result


@njit
def are_individuals_close(
    individual_to_check: np.ndarray,
    reference_individual: np.ndarray,
    relative_tolerance: float,
    absolute_tolerance: float,
    num_event_types: int,
) -> bool:
    beta_starting_index = num_event_types + num_event_types * num_event_types
    rhos_to_check = individual_to_check[num_event_types:beta_starting_index]
    betas_to_check = individual_to_check[beta_starting_index:]
    betas_reference = reference_individual[beta_starting_index:]

    mus_rhos_to_check = individual_to_check[:beta_starting_index]
    mus_rhos_reference = reference_individual[:beta_starting_index]

    return np.all(
        logical_or(
            np.abs(mus_rhos_to_check - mus_rhos_reference)
            <= relative_tolerance * np.abs(mus_rhos_reference),
            np.abs(mus_rhos_to_check - mus_rhos_reference) <= absolute_tolerance,
        ),
    ) and np.all(
        logical_or(
            np.abs(betas_to_check - betas_reference)
            <= relative_tolerance * np.abs(betas_reference),
            rhos_to_check <= absolute_tolerance,
        )
    )


@njit
def get_reconstructed_individual(individual, fixed_parameters_values):
    result = fixed_parameters_values.copy()
    nan_indices = np.isnan(result)  # Find where NaNs are located
    result[nan_indices] = individual  # Replace NaNs with values from individual

    return result


@njit
def get_fitness_with_fixed_parameters(
    individual: np.ndarray,
    end_time: float,
    events_times: List[np.ndarray],
    regularization_param: float,
    instability_param: float,
    fixed_parameters_values: np.ndarray,
) -> float:
    reconstructed_individual = get_reconstructed_individual(
        individual, fixed_parameters_values
    )

    return get_likelihood_fitness_from_individual(
        reconstructed_individual,
        end_time,
        events_times,
        regularization_param,
        instability_param,
    )


# I made some changes to this code, if it is correct implement em also in the lshade function
def lshade_save_info(
    gene_lower_boundaries,
    gene_upper_boundaries,
    initial_population,
    max_generations,
    memory_size,
    p,
    max_number_fitness_evaluations,
    end_time,
    events_times,
    regularization_param,
    instability_param,
    path_to_save_info,
    number_of_best_individuals_without_neighborhood,
    relative_tolerance_for_best_individuals_neighborhood,
    absolute_tolerance_for_best_individuals_neighborhood,
    fixed_parameters_values: np.ndarray,  # array wih length n +n^2 + n^2 with nan if not fixed, fixed value if fixed
) -> np.ndarray:
    num_events_types = len(events_times)

    population_size = initial_population.shape[0]
    archive_size = initial_population.shape[0]
    problem_dimensionality = initial_population.shape[1]
    archive = np.zeros((archive_size, problem_dimensionality))
    archive_fitnesses = np.empty(archive_size)
    archive_fitnesses.fill(np.nan)

    memory = np.zeros((memory_size, 2)) + 0.5
    MEMORY_MCR_INDEX = 0
    MEMORY_MMR_INDEX = 1

    previous_gen_population = initial_population.copy()

    crossing_rates = np.zeros(population_size)
    mutation_rates = np.zeros(population_size)

    previous_gen_individuals_fitnesses = np.empty(population_size)
    for individual_index in range(population_size):
        previous_gen_individuals_fitnesses[individual_index] = (
            get_fitness_with_fixed_parameters(
                previous_gen_population[individual_index],
                end_time,
                events_times,
                regularization_param,
                instability_param,
                fixed_parameters_values,
            )
        )

    current_number_of_fitness_evaluations = population_size
    memory_index_to_update = 0

    min_number_of_individuals = 4
    generation_number = 1

    best_individual_fitness_index = np.argmin(previous_gen_individuals_fitnesses)
    global_best_individual = previous_gen_population[best_individual_fitness_index]
    global_best_individual_fitness = previous_gen_individuals_fitnesses[
        best_individual_fitness_index
    ]

    np.savetxt(
        f"{path_to_save_info}\\population_gen{generation_number}.tsv",
        previous_gen_population,
        delimiter="\t",
    )
    np.savetxt(
        f"{path_to_save_info}\\fitnesses_gen{generation_number}.tsv",
        previous_gen_individuals_fitnesses,
        delimiter="\t",
    )

    while (
        (generation_number <= max_generations)
        and (current_number_of_fitness_evaluations < max_number_fitness_evaluations)
        and (population_size > min_number_of_individuals)
    ):
        print(f"Generation {generation_number}")

        succeding_crossover_rates = np.empty(population_size)
        succeding_crossover_rates.fill(np.nan)
        succeding_mutation_rates = np.empty(population_size)
        succeding_mutation_rates.fill(np.nan)
        succeding_trials = np.empty((population_size, problem_dimensionality))
        succeding_trial_fitnesses = np.empty(population_size)
        succeding_trials_count = 0
        trail_individuals = np.empty((population_size, problem_dimensionality))
        trial_fitnesses = np.empty(population_size)
        are_individuals_surpassed = np.zeros(population_size)
        next_gen_individuals = previous_gen_population.copy()
        next_gen_individuals_fitnesses = previous_gen_individuals_fitnesses.copy()

        number_of_best_individuals = round(population_size * p)
        best_individuals_indices = np.argpartition(
            previous_gen_individuals_fitnesses, number_of_best_individuals
        )[:number_of_best_individuals]

        for individual_index in range(population_size):
            memory_index = np.random.randint(0, memory_size)
            if np.isnan(memory[memory_index, MEMORY_MCR_INDEX]):
                crossing_rates[individual_index] = 0
            else:
                crossing_rates[individual_index] = get_clipped_number(
                    np.random.normal(memory[memory_index, MEMORY_MCR_INDEX], 0.1),
                    0.0,
                    1.0,
                )

            generated_mutation_rate = 0
            while generated_mutation_rate <= 0:
                generated_mutation_rate = get_sample_from_cauchy(
                    memory[memory_index, MEMORY_MMR_INDEX], 0.1
                )
                if generated_mutation_rate > 1:
                    generated_mutation_rate = 1

            mutation_rates[individual_index] = generated_mutation_rate

            random_best_individual_index = np.random.randint(
                0, number_of_best_individuals
            )

            best_individual = previous_gen_population[
                best_individuals_indices[random_best_individual_index]
            ]

            number_individuals_in_archive = np.sum(~np.isnan(archive_fitnesses))
            individual_indices_not_current = setdiff1d(
                np.arange(population_size + number_individuals_in_archive),
                individual_index,
            )

            random_individual_indices = np.random.choice(
                individual_indices_not_current, 2, replace=False
            )

            random_individual_1 = (
                previous_gen_population[random_individual_indices[0]]
                if random_individual_indices[0] < population_size
                else archive[random_individual_indices[0] - population_size]
            )
            random_individual_2 = (
                previous_gen_population[random_individual_indices[1]]
                if random_individual_indices[1] < population_size
                else archive[random_individual_indices[1] - population_size]
            )
            current_individual = previous_gen_population[individual_index]

            mutant = (
                current_individual
                + mutation_rates[individual_index]
                * (best_individual - current_individual)
                + mutation_rates[individual_index]
                * (random_individual_1 - random_individual_2)
            )

            for gene_index in range(problem_dimensionality):
                if mutant[gene_index] < gene_lower_boundaries[gene_index]:
                    mutant[gene_index] = (
                        gene_lower_boundaries[gene_index]
                        + current_individual[gene_index]
                    ) / 2
                elif mutant[gene_index] > gene_upper_boundaries[gene_index]:
                    mutant[gene_index] = (
                        gene_upper_boundaries[gene_index]
                        + current_individual[gene_index]
                    ) / 2

            random_gene_index_to_mutate = np.random.randint(0, problem_dimensionality)
            crossover_gene_index = (
                np.random.rand(problem_dimensionality)
                < crossing_rates[individual_index]
            )
            crossover_gene_index[random_gene_index_to_mutate] = True

            trail_individual = np.where(
                crossover_gene_index, mutant, current_individual
            )
            trail_individuals[individual_index] = trail_individual
            trial_fitnesses[individual_index] = get_fitness_with_fixed_parameters(
                trail_individual,
                end_time,
                events_times,
                regularization_param,
                instability_param,
                fixed_parameters_values,
            )
            current_number_of_fitness_evaluations = (
                current_number_of_fitness_evaluations + 1
            )

        for individual_index in range(population_size):

            if (
                trial_fitnesses[individual_index]
                <= previous_gen_individuals_fitnesses[individual_index]
            ):
                next_gen_individuals[individual_index] = trail_individuals[
                    individual_index
                ]
                next_gen_individuals_fitnesses[individual_index] = trial_fitnesses[
                    individual_index
                ]

                if (
                    trial_fitnesses[individual_index]
                    < previous_gen_individuals_fitnesses[individual_index]
                ):
                    if number_individuals_in_archive < archive_size:
                        archive[number_individuals_in_archive] = trail_individuals[
                            individual_index
                        ]
                        archive_fitnesses[number_individuals_in_archive] = (
                            trial_fitnesses[individual_index]
                        )
                    else:
                        random_element_index = np.random.randint(0, archive_size)
                        archive[random_element_index] = trail_individuals[
                            individual_index
                        ]
                        archive_fitnesses[random_element_index] = trial_fitnesses[
                            individual_index
                        ]

                    succeding_mutation_rates[succeding_trials_count] = mutation_rates[
                        individual_index
                    ]
                    succeding_crossover_rates[succeding_trials_count] = crossing_rates[
                        individual_index
                    ]

                    succeding_trials[succeding_trials_count] = trail_individuals[
                        individual_index
                    ]
                    succeding_trial_fitnesses[succeding_trials_count] = trial_fitnesses[
                        individual_index
                    ]

                    are_individuals_surpassed[individual_index] = 1

                    succeding_trials_count += 1

                    if (
                        trial_fitnesses[individual_index]
                        < global_best_individual_fitness
                    ):
                        global_best_individual = trail_individuals[individual_index]
                        global_best_individual_fitness = trial_fitnesses[
                            individual_index
                        ]

        if succeding_trials_count > 0:
            if np.nanmax(succeding_mutation_rates) <= 0 or np.isnan(
                memory[memory_index_to_update, MEMORY_MCR_INDEX]
            ):
                memory[memory_index_to_update, MEMORY_MCR_INDEX] = np.nan
            else:
                memory[memory_index_to_update, MEMORY_MCR_INDEX] = weighted_lehmar_mean(
                    succeding_crossover_rates[:succeding_trials_count],
                    previous_gen_individuals_fitnesses[are_individuals_surpassed == 1],
                    succeding_trial_fitnesses[:succeding_trials_count],
                )

            memory[memory_index_to_update, MEMORY_MMR_INDEX] = weighted_lehmar_mean(
                succeding_mutation_rates[:succeding_trials_count],
                previous_gen_individuals_fitnesses[are_individuals_surpassed == 1],
                succeding_trial_fitnesses[:succeding_trials_count],
            )

            memory_index_to_update = (memory_index_to_update + 1) % memory_size

        # Here goes the code to remove neighbors of the best individuals
        if population_size >= number_of_best_individuals_without_neighborhood:
            best_individuals_indices_to_remove_neighborhood = np.argpartition(
                next_gen_individuals_fitnesses,
                number_of_best_individuals_without_neighborhood,
            )[:number_of_best_individuals_without_neighborhood]

            best_individuals_indices_to_remove_neighborhood = (
                best_individuals_indices_to_remove_neighborhood[
                    np.argsort(
                        next_gen_individuals_fitnesses[
                            best_individuals_indices_to_remove_neighborhood
                        ]
                    )
                ]
            )

            individual_indices_close_to_best_individuals = np.zeros(
                population_size, dtype=int
            )
            for i, current_best_individual_index in enumerate(
                best_individuals_indices_to_remove_neighborhood
            ):
                for individual_index in range(population_size):
                    if (
                        individual_index != current_best_individual_index
                        and individual_index
                        not in best_individuals_indices_to_remove_neighborhood[:i]
                    ):
                        if are_individuals_close(
                            next_gen_individuals[individual_index],
                            next_gen_individuals[current_best_individual_index],
                            relative_tolerance_for_best_individuals_neighborhood,
                            absolute_tolerance_for_best_individuals_neighborhood,
                            num_events_types,
                        ):
                            individual_indices_close_to_best_individuals[
                                individual_index
                            ] = 1
        else:
            individual_indices_close_to_best_individuals = np.zeros(
                population_size, dtype=int
            )

        individual_indices_to_remove = np.where(
            individual_indices_close_to_best_individuals == 1
        )[0]

        next_gen_population_size = round(
            (
                (min_number_of_individuals - initial_population.shape[0])
                / max_number_fitness_evaluations
            )
            * current_number_of_fitness_evaluations
            + initial_population.shape[0]
        )

        if next_gen_population_size < population_size:
            if next_gen_population_size < min_number_of_individuals:
                next_gen_individuals = np.empty((0, 2))
                next_gen_individuals_fitnesses = np.empty(0)
            else:
                number_of_individuals_to_delete = (
                    population_size - next_gen_population_size
                )
                worst_individuals_indices = np.argpartition(
                    next_gen_individuals_fitnesses, -number_of_individuals_to_delete
                )[-number_of_individuals_to_delete:]

                if len(individual_indices_to_remove) > 0:
                    ending_index_to_remove = (
                        (len(individual_indices_to_remove) - min_number_of_individuals)
                        if (
                            next_gen_population_size - len(individual_indices_to_remove)
                            < min_number_of_individuals
                        )
                        else len(individual_indices_to_remove)
                    )

                    worst_individuals_indices = np.unique(
                        np.concatenate(
                            (
                                individual_indices_to_remove[:ending_index_to_remove],
                                worst_individuals_indices,
                            )
                        )
                    )

                next_gen_individuals = remove_rows(
                    next_gen_individuals, worst_individuals_indices
                )
                next_gen_individuals_fitnesses = np.delete(
                    next_gen_individuals_fitnesses, worst_individuals_indices
                )
                population_size = population_size - len(worst_individuals_indices)

        previous_gen_population = next_gen_individuals
        previous_gen_individuals_fitnesses = next_gen_individuals_fitnesses
        generation_number = generation_number + 1

        np.savetxt(
            f"{path_to_save_info}\\population_gen{generation_number}.tsv",
            previous_gen_population,
            delimiter="\t",
        )
        np.savetxt(
            f"{path_to_save_info}\\fitnesses_gen{generation_number}.tsv",
            previous_gen_individuals_fitnesses,
            delimiter="\t",
        )

    return global_best_individual
