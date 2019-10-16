"""Model and SgdEarlyStopTrainer integration tests suit."""
from unittest import TestCase
from unittest.mock import MagicMock
import numpy as np

from mlscratch import Model
from mlscratch.trainer import SgdEarlyStopTrainer
from mlscratch.measurer import ProbsMeasurer
from ..test_helper import _TrainWatcherRecorder, _PlaybackArch


class ModelAndSgdEarlyStopTrainerIntegrationTest(TestCase):

    def test_train_check_returned_and_call_count(self):
        arch = MagicMock()
        arch.update_params.return_value = (0.4, None)
        arch.check_cost.return_value = (0.66, None)
        model = Model(arch)
        trainer = SgdEarlyStopTrainer()
        measurer = MagicMock()
        measurer.measure.return_value = 60.5

        (epochs,
         costs,
         accuracies,
         validation_epochs,
         validation_costs,
         validation_accuracies,
        ) = model.train(
            np.array([1, 2, 3]),
            np.array([0, 1, 0]),
            None,
            None,
            trainer,
            measurer,
            None,
            epochs=8,
            validation_gap=1,
            minibatch_size=1,
            patience=2,
        )

        self.assertEqual(epochs, 3)
        self.assertEqual(costs, [0.4] * 3)
        self.assertEqual(accuracies, [60.5] * 3)
        self.assertEqual(validation_epochs, 3)
        self.assertEqual(validation_costs, [0.66] * 3)
        self.assertEqual(validation_accuracies, [60.5] * 3)

        self.assertEqual(arch.update_params.call_count, 9)
        self.assertEqual(arch.check_cost.call_count, 3)

    def test_train_assert_recorded_and_returned_are_identical(self):
        arch = MagicMock()
        arch.update_params.return_value = (0.1, None)
        arch.check_cost.return_value = (0.66, None)
        model = Model(arch)
        trainer = SgdEarlyStopTrainer()
        measurer = MagicMock()
        measurer.measure.return_value = 90.5
        train_watcher = _TrainWatcherRecorder()

        (epochs,
         costs,
         accuracies,
         validation_epochs,
         validation_costs,
         validation_accuracies,
        ) = model.train(
            np.array([1, 2, 3]),
            np.array([1, 0, 0]),
            None,
            None,
            trainer,
            measurer,
            train_watcher,
            epochs=9,
            validation_gap=2,
            minibatch_size=2,
            patience=1,
        )

        self.assertEqual(epochs, 3)
        self.assertEqual(train_watcher.epochs, list(range(epochs)))
        self.assertEqual(train_watcher.costs, costs)
        self.assertEqual(train_watcher.accuracies, accuracies)
        self.assertEqual(len(train_watcher.validation_epochs), validation_epochs)
        self.assertEqual(train_watcher.validation_costs, validation_costs)
        self.assertEqual(train_watcher.validation_accuracies, validation_accuracies)

        self.assertEqual(arch.update_params.call_count, 6)
        self.assertEqual(arch.check_cost.call_count, 2)

    def test_train_having_decreasing_cost_instance_check_it_does_not_stops_early(self):
        arch = _PlaybackArch(
            [0.05] * 9,
            [None] * 9,
            [0.1, 0.01, 0.001],
            [None] * 3,
        )
        model = Model(arch)
        measurer = MagicMock()
        measurer.measure.return_value = 90.5
        train_watcher = _TrainWatcherRecorder()

        model.train(
            np.array([1, 2, 3]),
            np.array([0, 1, 1]),
            None,
            None,
            SgdEarlyStopTrainer(),
            measurer,
            train_watcher,
            epochs=3,
            validation_gap=1,
            minibatch_size=1,
            patience=1,
            seed=0,
        )

        self.assertEqual(train_watcher.epochs, list(range(3)))
        self.assertEqual(train_watcher.costs, [0.05] * 3)
        self.assertEqual(train_watcher.accuracies, [90.5] * 3)
        self.assertEqual(train_watcher.validation_epochs, list(range(3)))
        self.assertEqual(train_watcher.validation_costs, [0.1, 0.01, 0.001])
        self.assertEqual(train_watcher.validation_accuracies, [90.5] * 3)

        self.assertEqual(arch.update_params_call_count, 9)
        self.assertEqual(arch.check_cost_call_count, 3)

    def test_train_with_measurer_check_accuracies(self):
        arch = _PlaybackArch(
            [0.05] * 9,
            [np.array([[0, 1], [1, 0], [0, 1]])] * 9,
            [0.1, 0.01, 0.001],
            [np.array([[1, 0], [1, 0]])] * 3,
        )
        model = Model(arch)
        train_watcher = _TrainWatcherRecorder()

        model.train(
            np.array([[1], [2], [3]]),
            np.array([[0, 1], [0, 1], [0, 1]]),
            np.array([[1], [2]]),
            np.array([[0, 1], [0, 1]]),
            SgdEarlyStopTrainer(),
            ProbsMeasurer(),
            train_watcher,
            epochs=3,
            validation_gap=1,
            minibatch_size=1,
            patience=1,
            seed=0,
        )

        self.assertEqual(train_watcher.epochs, list(range(3)))
        self.assertEqual(train_watcher.costs, [0.05] * 3)
        self.assertEqual(train_watcher.accuracies, [2/3] * 3)
        self.assertEqual(train_watcher.validation_epochs, list(range(3)))
        self.assertEqual(train_watcher.validation_costs, [0.1, 0.01, 0.001])
        self.assertEqual(train_watcher.validation_accuracies, [0] * 3)
