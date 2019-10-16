"""Model and AssertionsMeasurer integration tests suit."""
from unittest import TestCase
from unittest.mock import MagicMock
import numpy as np

from mlscratch import Model
from mlscratch.measurer import AssertionsMeasurer
from mlscratch.trainer import SimpleTrainer
from ..test_helper import _TrainWatcherRecorder


class ModelAndAssertionsMeasurerIntegrationTest(TestCase):

    def test_measure_with_one_measurer_check_returned(self):
        arch = MagicMock()
        arch.check_cost.return_value = (
            None,
            np.array([
                [-1, 0, 3],
                [3, 22, 0],
                [1, 2, 0.5],
            ]),
        )
        model = Model(arch)

        (_, measures) = model.measure(
            np.array([]),
            np.array([
                [1, -2, 3],
                [3, 2, 1],
                [3, 22, 0],
            ]),
            [AssertionsMeasurer()],
        )

        self.assertEqual(measures, [2/3])

    def test_measure_with_several_measurers_check_returned(self):
        arch = MagicMock()
        arch.check_cost.return_value = (
            None,
            np.array([
                [-1, 0, 3],
                [3, 22, 0],
                [1, 2, 0.5],
                [-1, 0, 3],
            ]),
        )
        model = Model(arch)

        (_, measures) = model.measure(
            np.array([]),
            np.array([
                [1, -2, 3],
                [3, 2, 1],
                [3, 22, 0],
                [-1, 0, 3],
            ]),
            [
                AssertionsMeasurer(),
                AssertionsMeasurer(),
                AssertionsMeasurer(),
            ],
        )

        self.assertEqual(measures, [3/4] * 3)

    def test_train_with_simple_trainer_check_recorded_accuracies(self):
        arch = MagicMock()
        arch.update_params.return_value = (
            None,
            np.array([
                [1, 0, -3],
                [3, 22, 0],
                [1, 2, 0.5],
            ]),
        )
        arch.check_cost.return_value = (
            None,
            np.array([
                [10, 1, -9],
                [-0.01, -0.8, -0.5],
                [11, 20, 0.5],
            ]),
        )
        model = Model(arch)
        trainer = SimpleTrainer()
        recorder = _TrainWatcherRecorder()

        model.train(
            None,
            np.array([
                [6, 3, 0],
                [1, 0.5, 8],
                [10, -100, 9.5],
            ]),
            None,
            np.array([
                [11, 1, -9],
                [-0.02, -0.8, -0.5],
                [11, 21, 0.5],
            ]),
            trainer,
            AssertionsMeasurer(),
            recorder,
            epochs=7,
            validation_gap=1,
        )

        self.assertEqual(recorder.accuracies, [1/3] * 7)
        self.assertEqual(recorder.validation_accuracies, [1] * 7)
