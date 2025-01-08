from typing import Callable, Tuple

import numpy as np
from scipy.stats import cauchy


def get_initial_random_population(
    gene_upper_boundaries: np.ndarray,
    initial_population_size: int,
    event_counts_for_type: np.ndarray,
    training_time_in_seconds: int,
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
        current_mu_max_value = 0.2*event_counts_for_type[i]/training_time_in_seconds
        initial_population[:, i] = np.random.uniform(
            0, min(gene_upper_boundaries[i], current_mu_max_value), initial_population_size
        )

    # fill the initial population with random rho values
    for i,j in [(i,j) for i in range(type_count) for j in range(type_count)]:
        current_rho_max_value = min(
            0.2*event_counts_for_type[i]/event_counts_for_type[j],
            0.5
        )

        initial_population[:, type_count + i*type_count + j] = np.random.uniform(
            0,
            min(gene_upper_boundaries[type_count + i*type_count + j], current_rho_max_value),
            initial_population_size
        )
    
    # rescale rho for individual i such that l2 norm of rho_i is less than 1
    #for i in range(type_count):
    #    rho_indices = [type_count + i*type_count + j for j in range(type_count)]
    #    rho_values = initial_population[:, rho_indices]
    #    rho_norms = np.linalg.norm(rho_values, axis=1)
    #    rho_values = rho_values / rho_norms[:, np.newaxis]

    #initial_population[:, rho_indices] = rho_values
    
    # fill the initial population with random beta values
    for i,j in [(i,j) for i in range(type_count) for j in range(type_count)]:
        bernoulli_samples = np.random.binomial(n=1, p=0.5, size=initial_population_size)

        random_values = np.where(
            bernoulli_samples == 0,
            np.random.uniform(0, 1, size=initial_population_size),
            np.random.uniform(0, 100, size=initial_population_size)
        )

        initial_population[:, type_count + square_matrix_size + i*type_count + j] = random_values

    return initial_population

def weighted_lehmar_mean(succeding_rates, individual_fitnesses, trial_fitnesses):
    improvements = individual_fitnesses - trial_fitnesses
    total_improvement = np.sum(improvements)
    weights = improvements / total_improvement

    lehmar_mean = np.sum(
        weights * succeding_rates**2
    ) / np.sum(weights * succeding_rates)

    return lehmar_mean

def lshade(
    fitness: Callable[[np.ndarray], float],
    problem_dimensionality: int,
    gene_lower_boundaries: np.ndarray,
    gene_upper_boundaries: np.ndarray,
    initial_population: np.ndarray,
    initial_population_size: int = 100,
    max_generations: int = 100,
    memory_size: int = 10,
    p: float = 0.2,
    max_number_fitness_evaluations: int = 1000,
) -> Tuple[np.ndarray, float]:
    population_size = initial_population_size
    archive_size = initial_population_size
    archive = np.zeros((archive_size, problem_dimensionality))
    archive_fitnesses = np.empty(archive_size)
    archive_fitnesses.fill(np.nan)

    memory = np.zeros((memory_size, 2)) + 0.5
    MEMORY_MCR_INDEX = 0
    MEMORY_MMR_INDEX = 1

    previous_gen_population = initial_population

    crossing_rates = np.zeros(population_size)
    mutation_rates = np.zeros(population_size)

    previous_gen_individuals_fitnesses = np.array([fitness(individual) for individual in previous_gen_population])
    current_number_of_fitness_evaluations = population_size
    memory_index_to_update = 0

    min_number_of_individuals = 4
    generation_number = 1

    best_individual_fitness_index = np.argmin(previous_gen_individuals_fitnesses)
    global_best_individual = previous_gen_population[best_individual_fitness_index]
    global_best_individual_fitness = previous_gen_individuals_fitnesses[best_individual_fitness_index]

    while (generation_number <= max_generations) and (current_number_of_fitness_evaluations < max_number_fitness_evaluations):
        
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
                crossing_rates[individual_index] = np.clip(
                    np.random.normal(memory[memory_index, MEMORY_MCR_INDEX], 0.1),
                    0,
                    1.0
                )

            generated_mutation_rate = 0
            while generated_mutation_rate <= 0:
                generated_mutation_rate = np.clip(
                    cauchy.rvs(memory[memory_index, MEMORY_MMR_INDEX],0.1),
                    None,
                    1.0
                )
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
            individual_indices_not_current = [
                index for index in range(population_size + number_individuals_in_archive)
                if index != individual_index
            ]
            random_individual_indices =np.random.choice(
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

            mutant = current_individual + mutation_rates[individual_index] * (
                best_individual - current_individual
            ) + mutation_rates[individual_index] * (
                random_individual_1 - random_individual_2
            )

            for gene_index in range(problem_dimensionality):
                if mutant[gene_index] < gene_lower_boundaries[gene_index]:
                    mutant[gene_index] = (
                        gene_lower_boundaries[gene_index] + current_individual[gene_index]
                    ) / 2
                elif mutant[gene_index] > gene_upper_boundaries[gene_index]:
                    mutant[gene_index] = (
                        gene_upper_boundaries[gene_index] + current_individual[gene_index]
                    ) / 2

            random_gene_index_to_mutate = np.random.randint(0, problem_dimensionality)
            crossover_gene_index = np.random.rand(problem_dimensionality) < crossing_rates[individual_index]
            crossover_gene_index[random_gene_index_to_mutate] = True

            trail_individual = np.where(crossover_gene_index, mutant, current_individual)
            trail_individuals[individual_index] = trail_individual
            trial_fitnesses[individual_index] = fitness(trail_individual)
            current_number_of_fitness_evaluations = current_number_of_fitness_evaluations + 1

        for individual_index in range(population_size):

            if (
                trial_fitnesses[individual_index] <=
                previous_gen_individuals_fitnesses[individual_index]
            ):
                next_gen_individuals[individual_index] = trail_individuals[individual_index]
                next_gen_individuals_fitnesses[individual_index] = trial_fitnesses[individual_index]

                if trial_fitnesses[individual_index] < previous_gen_individuals_fitnesses[individual_index]:
                    if number_individuals_in_archive < archive_size:
                        archive[number_individuals_in_archive] = trail_individuals[individual_index]
                        archive_fitnesses[number_individuals_in_archive] = trial_fitnesses[individual_index]
                    else:
                        random_element_index = np.random.randint(0, archive_size)
                        archive[random_element_index] = trail_individuals[individual_index]
                        archive_fitnesses[random_element_index] = trial_fitnesses[individual_index]

                    succeding_mutation_rates[succeding_trials_count] = mutation_rates[individual_index]
                    succeding_crossover_rates[succeding_trials_count] = crossing_rates[individual_index]

                    succeding_trials[succeding_trials_count] = trail_individuals[individual_index]
                    succeding_trial_fitnesses[succeding_trials_count] = trial_fitnesses[individual_index]

                    are_individuals_surpassed[individual_index] = 1

                    succeding_trials_count += 1

                    if trial_fitnesses[individual_index] < global_best_individual_fitness:
                        global_best_individual = trail_individuals[individual_index]
                        global_best_individual_fitness = trial_fitnesses[individual_index]

        if succeding_trials_count > 0:
            if (
                np.nanmax(succeding_mutation_rates) <= 0 or
                np.isnan(memory[memory_index_to_update, MEMORY_MCR_INDEX])
            ):
                memory[memory_index_to_update, MEMORY_MCR_INDEX] = np.nan
            else:
                memory[memory_index_to_update, MEMORY_MCR_INDEX] = weighted_lehmar_mean(
                    succeding_crossover_rates[:succeding_trials_count],
                    previous_gen_individuals_fitnesses[are_individuals_surpassed == 1],
                    succeding_trial_fitnesses[:succeding_trials_count]
                )
            
            memory[memory_index_to_update, MEMORY_MMR_INDEX] = weighted_lehmar_mean(
                succeding_mutation_rates[:succeding_trials_count],
                previous_gen_individuals_fitnesses[are_individuals_surpassed == 1],
                succeding_trial_fitnesses[:succeding_trials_count]
            )

            memory_index_to_update = (memory_index_to_update + 1) % memory_size

        next_gen_population_size = round(
            (
                (min_number_of_individuals - initial_population_size) /
                max_number_fitness_evaluations
            ) * current_number_of_fitness_evaluations + initial_population_size
        )

        if next_gen_population_size < population_size:
            if next_gen_population_size < min_number_of_individuals:
                next_gen_individuals = np.empty((0,2))
                next_gen_individuals_fitnesses = np.empty(0)
            else:
                number_of_individuals_to_delete = population_size - next_gen_population_size
                worst_individuals_indices = np.argpartition(
                    next_gen_individuals_fitnesses, -number_of_individuals_to_delete
                )[:number_of_individuals_to_delete]

                next_gen_individuals = np.delete(next_gen_individuals, worst_individuals_indices, axis=0)
                next_gen_individuals_fitnesses = np.delete(
                    next_gen_individuals_fitnesses, worst_individuals_indices
                )
                population_size = next_gen_population_size
        
        previous_gen_population = next_gen_individuals
        previous_gen_individuals_fitnesses = next_gen_individuals_fitnesses
        generation_number = generation_number + 1

    return global_best_individual, global_best_individual_fitness
