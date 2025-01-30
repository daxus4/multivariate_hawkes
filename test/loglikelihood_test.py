import unittest

import numpy as np
from numba.typed import List

from optimization.hawkes_likelihood import (
    loglikelihood_negative_exponential_contribution,
)


class TestStringMethods(unittest.TestCase):

    def setUp(self):
        self.alphas = np.array([[1, 0.5], [1.5, 2]])
        self.betas = np.array([[1.0, 2], [3.0, 4]])
        self.mus = np.array([0.5, 0.25])
        self.end_time = 7
        self.events_times = List(
            [np.array([0.5, 1.0, 5, 7]), np.array([1.0, 1.5, 2, 5, 6])]
        )

        self.test_r_mn_for_event_type_12_expected_values = [
            0,
            0,
            np.exp(-8) + np.exp(-7) + np.exp(-6),
            np.exp(-12) + np.exp(-11) + np.exp(-10) + np.exp(-4) + np.exp(-2),
        ]

        self.test_r_mn_for_event_type_11_expected_values = [
            0,
            np.exp(-0.5),
            np.exp(-4.5) + np.exp(-4),
            np.exp(-6.5) + np.exp(-6) + np.exp(-2),
        ]

    def test_negative_exponential_effect(self):
        from optimization.hawkes_likelihood import negative_exponential_effect

        attempt = negative_exponential_effect(
            self.alphas[0][0], self.betas[0][0], self.end_time, self.events_times[0]
        )
        self.assertAlmostEqual(attempt, 3 - np.exp(-6.5) - np.exp(-6) - np.exp(-2))

        attempt2 = negative_exponential_effect(
            self.alphas[1][1], self.betas[1][1], self.end_time, self.events_times[1]
        )
        self.assertAlmostEqual(
            attempt2,
            (5 - np.exp(-24) - np.exp(-22) - np.exp(-20) - np.exp(-8) - np.exp(-4)) / 2,
        )

        attempt_no_times = negative_exponential_effect(2, 4, 3, np.array([]))
        self.assertEqual(attempt_no_times, 0)

    def test_loglikelihood_negative_exponential_contribution(self):
        from optimization.hawkes_likelihood import (
            loglikelihood_negative_exponential_contribution,
        )

        attempt = loglikelihood_negative_exponential_contribution(
            self.alphas[0],
            self.betas[0],
            self.end_time,
            self.events_times,
        )
        self.assertAlmostEqual(
            attempt,
            3
            - np.exp(-6.5)
            - np.exp(-6)
            - np.exp(-2)
            + (5 - np.exp(-12) - np.exp(-11) - np.exp(-10) - np.exp(-4) - np.exp(-2))
            / 4,
        )

    def test_r_mn_for_event_type_12(self):
        from optimization.hawkes_likelihood import r_mn

        events_type_2 = self.events_times[1]

        r_i = r_mn(self.betas[0][1], self.events_times[0][0], 0.0, 0.0, np.array([]))
        self.assertEqual(r_i, 0, "R(1) should be 0")

        for i in range(1, len(self.events_times[0])):
            events_between = events_type_2[
                (events_type_2 >= self.events_times[0][i - 1])
                & (events_type_2 < self.events_times[0][i])
            ]
            r_i = r_mn(
                self.betas[0][1],
                self.events_times[0][i],
                self.events_times[0][i - 1],
                r_i,
                events_between,
            )
            self.assertAlmostEqual(
                r_i,
                self.test_r_mn_for_event_type_12_expected_values[i],
                msg=f"R({i}) is wrong",
            )

    def test_r_mn_for_event_type_11(self):
        from optimization.hawkes_likelihood import r_mn

        events_type_1 = self.events_times[0]

        r_i = r_mn(self.betas[0][0], self.events_times[0][0], 0.0, 0.0, np.array([]))
        self.assertEqual(r_i, 0, "R(1) should be 0")

        for i in range(1, len(self.events_times[0])):
            events_between = events_type_1[
                (events_type_1 >= self.events_times[0][i - 1])
                & (events_type_1 < self.events_times[0][i])
            ]
            r_i = r_mn(
                self.betas[0][0],
                self.events_times[0][i],
                self.events_times[0][i - 1],
                r_i,
                events_between,
            )
            self.assertAlmostEqual(
                r_i,
                self.test_r_mn_for_event_type_11_expected_values[i],
                msg=f"R({i}) is wrong",
            )

    """def test_print(self):
        from optimization.hawkes_likelihood import r_mn

        events_type_1 = self.events_times[1]

        r_i = r_mn(self.betas[0][1], self.events_times[0][0], 0.0, 0.0, np.array([]))
        self.assertEqual(r_i, 0, "R(1) should be 0")

        for i in range(1, len(self.events_times[0])):
            events_between = events_type_1[
                (events_type_1 >= self.events_times[0][i - 1])
                & (events_type_1 < self.events_times[0][i])
            ]
            r_i = r_mn(
                self.betas[0][1],
                self.events_times[0][i],
                self.events_times[0][i - 1],
                r_i,
                events_between,
            )
            print(f"{i}", r_i)"""

    def test_counting_process_integral_subfunction(self):
        from optimization.hawkes_likelihood import counting_process_integral_subfunction

        attempt = counting_process_integral_subfunction(
            self.mus[0],
            self.alphas[0],
            np.array(
                [
                    self.test_r_mn_for_event_type_11_expected_values[0],
                    self.test_r_mn_for_event_type_12_expected_values[0],
                ]
            ),
        )
        self.assertAlmostEqual(attempt, np.log(0.5), "k=1")

        attempt = counting_process_integral_subfunction(
            self.mus[0],
            self.alphas[0],
            np.array(
                [
                    self.test_r_mn_for_event_type_11_expected_values[1],
                    self.test_r_mn_for_event_type_12_expected_values[1],
                ]
            ),
        )
        self.assertAlmostEqual(attempt, np.log(0.5 + np.exp(-0.5)), "k=2")

        attempt = counting_process_integral_subfunction(
            self.mus[0],
            self.alphas[0],
            np.array(
                [
                    self.test_r_mn_for_event_type_11_expected_values[2],
                    self.test_r_mn_for_event_type_12_expected_values[2],
                ]
            ),
        )
        self.assertAlmostEqual(
            attempt,
            np.log(
                0.5
                + self.alphas[0][1] * (np.exp(-8) + np.exp(-7) + np.exp(-6))
                + np.exp(-4.5)
                + np.exp(-4)
            ),
            "k=3",
        )

    """def test_counting_process_integral_subfunction(self):
        from optimization.hawkes_likelihood import counting_process_integral_subfunction

        value = np.log(0.5)
        for i in range(1, len(self.test_r_mn_for_event_type_11_expected_values)):
            attempt = counting_process_integral_subfunction(
                self.mus[0],
                self.alphas[0],
                np.array(
                    [
                        self.test_r_mn_for_event_type_11_expected_values[i],
                        self.test_r_mn_for_event_type_12_expected_values[i],
                    ]
                ),
            )
            value += attempt
            print(f"{i}", value)"""

    def test_loglikelihood_1(self):
        from optimization.hawkes_likelihood import loglikelihood_m

        negative_exponential_contribution = (
            loglikelihood_negative_exponential_contribution(
                self.alphas[0],
                self.betas[0],
                self.end_time,
                self.events_times,
            )
        )

        attempt = loglikelihood_m(
            0,
            self.mus[0],
            negative_exponential_contribution,
            self.alphas[0],
            self.betas[0],
            self.end_time,
            self.events_times,
        )
        self.assertAlmostEqual(
            attempt,
            -self.mus[0] * self.end_time - 5.630449806186935,
        )


if __name__ == "__main__":
    unittest.main()
