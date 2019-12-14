"""PairwiseMeasurer's test suit."""
from unittest import TestCase
import numpy as np

from mlscratch.measurer import PairwiseMeasurer

class PairwiseMeasurerTest(TestCase):

    def test_measure_all_match(self):
        measurer = PairwiseMeasurer()

        result = measurer.measure(
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
        )

        self.assertEqual(result, 1)

    def test_measure_none_match(self):
        measurer = PairwiseMeasurer()

        result = measurer.measure(
            np.array([10, 20, 30]),
            np.array([1, 2, 3]),
        )

        self.assertEqual(result, 0)

    def test_measure_none_match_if_shapes_differ(self):
        measurer = PairwiseMeasurer()

        result = measurer.measure(
            np.array([10, 20, 30]),
            np.array([1, 20, 30, 4]),
        )

        self.assertEqual(result, 0)

    def test_measure_some_match(self):
        measurer = PairwiseMeasurer()

        result = measurer.measure(
            np.array([[10], [20], [30]]),
            np.array([[1], [20], [3]]),
        )

        self.assertEqual(result, 1/3)
