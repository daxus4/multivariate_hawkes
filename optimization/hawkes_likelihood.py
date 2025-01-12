import numpy as np
from numba import njit
from numba.typed import List


@njit
def negative_exponential_effect(
    alpha_mn: float, beta_mn: float, end_time: float, times_n: np.ndarray
) -> float:
    return (alpha_mn / beta_mn) * np.sum(1 - np.exp(-beta_mn * (end_time - times_n)))


@njit
def loglikelihood_negative_exponential_contribution(
    alphas_mn: np.ndarray,
    betas_mn: np.ndarray,
    end_time: float,
    events_times: np.ndarray,
) -> float:
    num_events = len(events_times)
    negative_exponential_effects = np.empty(num_events, dtype=np.float64)

    for n in range(num_events):
        negative_exponential_effects[n] = negative_exponential_effect(
            alphas_mn[n], betas_mn[n], end_time, events_times[n]
        )

    return np.sum(negative_exponential_effects)


@njit
def r_mn(
    beta_mn: float,
    current_t_m: float,
    previous_t_m: float,
    previous_r_mn: float,
    intermediate_t_n_values: np.ndarray,
) -> float:
    return np.exp(-beta_mn * (current_t_m - previous_t_m)) * previous_r_mn + np.sum(
        np.exp(-beta_mn * (current_t_m - intermediate_t_n_values))
    )


@njit
def counting_process_integral_subfunction(
    mu_m: float,
    alphas_mn: np.ndarray,
    rs_mn: np.ndarray,
) -> float:
    return np.log(mu_m + np.sum(alphas_mn * rs_mn))


@njit
def loglikelihood_m(
    m_index: int,
    mu_m: float,
    negative_exponential_contribution: float,
    alphas_mn: np.ndarray,
    betas_mn: np.ndarray,
    end_time: float,
    events_times: List[np.ndarray],
) -> float:
    num_events = len(events_times)

    dt_integral = -mu_m * end_time - negative_exponential_contribution

    recursive_part_effects = np.zeros(num_events, dtype=np.float64)
    end_indices_times_events = np.zeros(num_events, dtype=np.float64)

    for n in range(num_events):
        end_indices_times_events[n] = np.searchsorted(
            events_times[n], events_times[m_index][0], side="left"
        )

        recursive_part_effects[n] = r_mn(
            betas_mn[n],
            events_times[m_index][0],
            0,
            recursive_part_effects[n],
            events_times[n][: end_indices_times_events[n]],
        )

    counting_process_integral = counting_process_integral_subfunction(
        mu_m,
        alphas_mn,
        recursive_part_effects,
    )

    for k in range(1, len(events_times[m_index])):
        start_indices_times_events = end_indices_times_events.copy()

        for n in range(num_events):
            current_t_m = events_times[m_index][k]
            previous_t_m = events_times[m_index][k - 1]

            end_indices_times_events[n] = (
                np.searchsorted(
                    events_times[n][start_indices_times_events[n] :],
                    current_t_m,
                    side="left",
                )
                + start_indices_times_events[n]
            )

            recursive_part_effects[n] = r_mn(
                betas_mn[n],
                current_t_m,
                previous_t_m,
                recursive_part_effects[n],
                events_times[n][
                    start_indices_times_events[n] : end_indices_times_events[n]
                ],
            )

        counting_process_integral += counting_process_integral_subfunction(
            mu_m,
            alphas_mn,
            recursive_part_effects,
        )

    return dt_integral + counting_process_integral


@njit
def loglikelihood(
    mu: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    end_time: float,
    events_times: List[np.ndarray],
) -> float:
    num_events = len(events_times)
    loglikelihood_value = 0

    negative_exponential_contributions = np.empty(num_events, dtype=np.float64)

    for n in range(num_events):
        negative_exponential_contributions[n] = (
            loglikelihood_negative_exponential_contribution(
                alphas[n], betas[n], end_time, events_times
            )
        )

    for m in range(num_events):
        loglikelihood_value += loglikelihood_m(
            m,
            mu[m],
            negative_exponential_contributions[m],
            alphas[m],
            betas[m],
            end_time,
            events_times,
        )

    return loglikelihood_value


@njit
def l1_penalty(rhos: np.ndarray, regularization_param: float) -> float:
    return regularization_param * (np.sum(np.abs(rhos)))


@njit
def instability_penalty(rhos: np.ndarray, instability_param: float) -> float:
    kernel_spectral_norm = np.linalg.norm(rhos, ord=2)
    if kernel_spectral_norm > 1:
        return instability_param * kernel_spectral_norm
    else:
        return 0


@njit
def likelihood_fitness_to_minimize(
    mu: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    end_time: float,
    events_times: List[np.ndarray],
    regularization_param: float,
    instability_param: float,
) -> float:
    rhos = alphas / betas

    return (
        -loglikelihood(mu, alphas, betas, end_time, events_times)
        + l1_penalty(rhos, regularization_param)
        + instability_penalty(rhos, instability_param)
    )


@njit
def get_likelihood_fitness_from_individual(
    individual: np.ndarray,
    end_time: float,
    events_times: List[np.ndarray],
    regularization_param: float,
    instability_param: float,
) -> float:
    num_event_types = len(events_times)

    mu = individual[:num_event_types]

    alphas = individual[
        num_event_types : num_event_types + num_event_types * num_event_types
    ].reshape(num_event_types, num_event_types)

    betas = individual[num_event_types + num_event_types * num_event_types :].reshape(
        num_event_types, num_event_types
    )

    return likelihood_fitness_to_minimize(
        mu,
        alphas,
        betas,
        end_time,
        events_times,
        regularization_param,
        instability_param,
    )
