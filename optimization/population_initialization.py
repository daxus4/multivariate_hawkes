import numpy as np
from scipy.stats.qmc import Halton

from optimization.hawkes_likelihood import get_spectral_radius


def generate_halton_sequence(n_points, n_dim, bounds):
    if len(bounds) != n_dim:
        raise ValueError("Number of bounds must match the number of dimensions")

    sampler = Halton(d=n_dim, scramble=True)
    halton_points = sampler.random(n=n_points)

    # Scale to given bounds
    scaled_points = np.zeros_like(halton_points)
    for i in range(n_dim):
        min_val, max_val = bounds[i]
        scaled_points[:, i] = min_val + (max_val - min_val) * halton_points[:, i]

    return scaled_points


def update_matrix(matrix, lower_spectral_radius=0.3):
    rows, cols = matrix.shape
    chosen_indices = set()  # Keep track of already modified indices

    spectral_radius = get_spectral_radius(matrix)
    if spectral_radius < lower_spectral_radius:
        num_elements = np.random.randint(1, rows)
        partial_increment = 1 / num_elements
        for _ in range(num_elements):
            i, j = np.random.randint(0, rows, size=2)
            matrix[i, j] += partial_increment

    while get_spectral_radius(matrix) >= 1:
        prob_matrix = 1 / (np.abs(matrix) + 1e-10)
        for x, y in chosen_indices:
            prob_matrix[x, y] = 0.01
        prob_matrix /= np.sum(prob_matrix)
        flat_index = np.random.choice(rows * cols, p=prob_matrix.ravel())
        i, j = divmod(flat_index, cols)

        matrix[i, j] = np.random.exponential(0.1)
        chosen_indices.add((i, j))

    matrix = np.clip(matrix, 0, 0.95)
    return matrix


def modify_random_elements(matrix):
    n, m = matrix.shape

    total_elements = n * m
    num_elements_to_modify = total_elements // 2

    indices = np.random.choice(total_elements, num_elements_to_modify, replace=False)

    flattened_matrix = matrix.flatten()
    flattened_matrix[indices] = np.random.uniform(0, 100, size=num_elements_to_modify)

    matrix = flattened_matrix.reshape(n, m)
    return matrix


def get_initial_sparsed_population(
    individuals_number: int,
    event_types_number: int,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    matrix_elements_number = event_types_number * event_types_number
    n_total_dim = event_types_number + 2 * matrix_elements_number

    bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(n_total_dim)]

    halton_data = generate_halton_sequence(individuals_number, n_total_dim, bounds)

    for i in range(halton_data.shape[0]):
        individual_rhos = halton_data[
            i, event_types_number : event_types_number + matrix_elements_number
        ]
        matrix = individual_rhos.reshape((event_types_number, event_types_number))
        matrix = update_matrix(matrix)
        halton_data[
            i, event_types_number : event_types_number + matrix_elements_number
        ] = matrix.flatten()

        individual_betas = halton_data[
            i, event_types_number + matrix_elements_number : n_total_dim
        ]
        matrix = individual_betas.reshape((event_types_number, event_types_number))
        matrix = modify_random_elements(matrix)
        halton_data[i, event_types_number + matrix_elements_number : n_total_dim] = (
            matrix.flatten()
        )

    return halton_data
