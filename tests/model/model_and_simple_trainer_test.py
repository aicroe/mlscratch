"""Model and SimpleTrainer integration tests suit."""
from unittest import TestCase
from unittest.mock import MagicMock
import numpy as np

from mlscratch import Model
from mlscratch.trainer import SimpleTrainer
from mlscratch.measurer import ProbsMeasurer
from ..test_helper import _TrainWatcherRecorder


class ModelAndSimpleTrainerIntegrationTest(TestCase):

    def test_train_check_returned(self):
        arch = MagicMock()
        arch.update_params.return_value = (0.4, None)
        model = Model(arch)
        trainer = SimpleTrainer()
        measurer = MagicMock()
        measurer.measure.return_value = 60.5

        (epochs,
         costs,
         accuracies,
         validation_epochs,
         validation_costs,
         validation_accuracies,
        ) = model.train(
            None,
            None,
            None,
            None,
            trainer,
            measurer,
            None,
            epochs=8,
        )

        self.assertEqual(epochs, 8)
        self.assertEqual(costs, [0.4] * 8)
        self.assertEqual(accuracies, [60.5] * 8)
        self.assertEqual(validation_epochs, 0)
        self.assertEqual(validation_costs, [])
        self.assertEqual(validation_accuracies, [])

    def test_train_with_validation_check_returned(self):
        arch = MagicMock()
        arch.update_params.return_value = (16.7, None)
        arch.check_cost.return_value = (24.6, None)
        model = Model(arch)
        trainer = SimpleTrainer()
        measurer = MagicMock()
        measurer.measure.return_value = 33.8

        (epochs,
         costs,
         accuracies,
         validation_epochs,
         validation_costs,
         validation_accuracies,
        ) = model.train(
            None,
            None,
            None,
            None,
            trainer,
            measurer,
            None,
            epochs=13,
            validation_gap=4,
        )

        self.assertEqual(epochs, 13)
        self.assertEqual(costs, [16.7] * 13)
        self.assertEqual(accuracies, [33.8] * 13)
        self.assertEqual(validation_epochs, 4)
        self.assertEqual(validation_costs, [24.6] * 4)
        self.assertEqual(validation_accuracies, [33.8] * 4)

    def test_train_check_recorded(self):
        arch = MagicMock()
        arch.update_params.return_value = (0.4, None)
        model = Model(arch)
        trainer = SimpleTrainer()
        measurer = MagicMock()
        measurer.measure.return_value = 60.5
        train_watcher = _TrainWatcherRecorder()

        model.train(
            None,
            None,
            None,
            None,
            trainer,
            measurer,
            train_watcher,
            epochs=8,
        )

        self.assertEqual(train_watcher.epochs, list(range(8)))
        self.assertEqual(train_watcher.costs, [0.4] * 8)
        self.assertEqual(train_watcher.accuracies, [60.5] * 8)
        self.assertEqual(train_watcher.validation_epochs, [])
        self.assertEqual(train_watcher.validation_costs, [])
        self.assertEqual(train_watcher.validation_accuracies, [])

    def test_train_with_validation_check_recorded(self):
        arch = MagicMock()
        arch.update_params.return_value = (0.1, None)
        arch.check_cost.return_value = (0.66, None)
        model = Model(arch)
        trainer = SimpleTrainer()
        measurer = MagicMock()
        measurer.measure.return_value = 90.5
        train_watcher = _TrainWatcherRecorder()

        model.train(
            None,
            None,
            None,
            None,
            trainer,
            measurer,
            train_watcher,
            epochs=9,
            validation_gap=3,
        )

        self.assertEqual(train_watcher.epochs, list(range(9)))
        self.assertEqual(train_watcher.costs, [0.1] * 9)
        self.assertEqual(train_watcher.accuracies, [90.5] * 9)
        self.assertEqual(train_watcher.validation_epochs, [0, 3, 6])
        self.assertEqual(train_watcher.validation_costs, [0.66] * 3)
        self.assertEqual(train_watcher.validation_accuracies, [90.5] * 3)

    def test_train_assert_recorded_and_returned_are_identical(self):
        arch = MagicMock()
        arch.update_params.return_value = (0.1, None)
        arch.check_cost.return_value = (0.66, None)
        model = Model(arch)
        trainer = SimpleTrainer()
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
            None,
            None,
            None,
            None,
            trainer,
            measurer,
            train_watcher,
            epochs=9,
            validation_gap=3,
        )

        self.assertEqual(train_watcher.epochs, list(range(epochs)))
        self.assertEqual(train_watcher.costs, costs)
        self.assertEqual(train_watcher.accuracies, accuracies)
        self.assertEqual(len(train_watcher.validation_epochs), validation_epochs)
        self.assertEqual(train_watcher.validation_costs, validation_costs)
        self.assertEqual(train_watcher.validation_accuracies, validation_accuracies)

    def test_train_with_measurer_check_accuracies(self):
        arch = MagicMock()
        arch.update_params.return_value = (
            None,
            np.array([
                [1, 0, -3],
                [3, 22, 0],
                [1, 2, 0.5],
            ]),
        )
        model = Model(arch)
        trainer = SimpleTrainer()

        _, _, accuracies, *_ = model.train(
            None,
            np.array([
                [6, 3, 0],
                [1, 0.5, 8],
                [10, -100, 9.5],
            ]),
            None,
            None,
            trainer,
            ProbsMeasurer(),
            None,
            epochs=7,
        )

        self.assertEqual(accuracies, [1/3] * 7)

    def test_train_with_validation_and_measurer_check_accuracies(self):
        arch = MagicMock()
        arch.update_params.return_value = (
            None,
            np.array([
                [-1, 8, 7],
                [-1, 1, 0],
            ]),
        )
        arch.check_cost.return_value = (
            None,
            np.array([
                [8, 0, 7],
                [-6, 1, 0],
            ]),
        )
        model = Model(arch)
        trainer = SimpleTrainer()

        _, _, accuracies, _, _, validation_accuracies = model.train(
            None,
            np.array([
                [62.9, 63, 0.0],
                [11, 0.5, 0.8],
            ]),
            None,
            np.array([
                [0, 1, 11.0],
                [-8, 0, 6.66],
            ]),
            trainer,
            ProbsMeasurer(),
            None,
            epochs=8,
            validation_gap=4,
        )

        self.assertEqual(accuracies, [1/2] * 8)
        self.assertEqual(validation_accuracies, [0] * 2)
