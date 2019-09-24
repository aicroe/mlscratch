from unittest import TestCase
import numpy as np

from mlscratch.measurer import AssertionsMeasurer

class AssertionsMeasurerTest(TestCase):

    def test_measure_with_matrix_like_data_assert_all_match_expected(self):
        measurer = AssertionsMeasurer()

        result = measurer.measure(
            np.array([
                [1, 0, 3],
                [3, 2, 0],
                [0, 3, 1]
            ]),
            np.array([
                [1, 2, 3],
                [3, 2, 1],
                [2, 3, 1]
            ]),
        )

        self.assertEqual(result, 1)

    def test_measure_with_matrix_like_data_assert_none_match_expected(self):
        measurer = AssertionsMeasurer()

        result = measurer.measure(
            np.array([
                [1, 80, -3],
                [3, 22, 0],
                [0, 39, 1]
            ]),
            np.array([
                [100, 2, 3],
                [-3, -2, 1],
                [2, 3, 111]
            ]),
        )

        self.assertEqual(result, 0)

    def test_measure_with_matrix_like_data_assert_some_math_expected(self):
        measurer = AssertionsMeasurer()

        result = measurer.measure(
            np.array([
                [10, 15, 0],
                [11, 67, 20],
                [55, 0, 1],
            ]),
            np.array([
                [1, 2, 3],
                [70, 2, -1],
                [66, 10, 0],
            ])
        )

        self.assertEqual(result, 1/3)
