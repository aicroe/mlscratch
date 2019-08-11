"""SimpleTrainer test suit"""
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from mlbatch.trainer import SimpleTrainer
from ..test_helper import _TrainWatcherRecorder


class SimpleTrainerTest(TestCase):

    def test_train_assert_returned_epochs_is_zero(self):
        trainer = SimpleTrainer()
        trainable = MagicMock()

        epochs, validation_epochs = trainer.train(
            trainable,
            None,
            None,
            None,
            None,
            None,
        )

        self.assertEqual(epochs, 0)
        self.assertEqual(validation_epochs, 0)

    def test_train_assert_watcher_records_are_empty(self):
        trainer = SimpleTrainer()
        trainable = MagicMock()
        train_watcher = _TrainWatcherRecorder()

        trainer.train(
            trainable,
            None,
            None,
            None,
            None,
            train_watcher,
        )

        self.assertEqual(train_watcher.epochs, [])
        self.assertEqual(train_watcher.costs, [])
        self.assertEqual(train_watcher.accuracies, [])
        self.assertEqual(train_watcher.validation_epochs, [])
        self.assertEqual(train_watcher.validation_costs, [])
        self.assertEqual(train_watcher.validation_accuracies, [])

    def test_train_single_epoch_check_recorded(self):
        trainer = SimpleTrainer()
        trainable = MagicMock()
        trainable.update_params.return_value = (3.5, 0.5)
        train_watcher = _TrainWatcherRecorder()

        (epochs, validation_epochs) = trainer.train(
            trainable,
            None,
            None,
            None,
            None,
            train_watcher,
            epochs=1,
        )

        self.assertEqual(epochs, 1)
        self.assertEqual(validation_epochs, 0)
        self.assertEqual(train_watcher.epochs, [0])
        self.assertEqual(train_watcher.costs, [3.5])
        self.assertEqual(train_watcher.accuracies, [0.5])
        self.assertEqual(train_watcher.validation_epochs, [])
        self.assertEqual(train_watcher.validation_costs, [])
        self.assertEqual(train_watcher.validation_accuracies, [])

    def test_train_single_epoch_check_trainable(self):
        trainer = SimpleTrainer()
        trainable = MagicMock()
        trainable.update_params.return_value = (1.0, 0.5)

        trainer.train(
            trainable,
            None,
            None,
            None,
            None,
            None,
            epochs=1,
        )

        self.assertEqual(trainable.update_params.call_count, 1)
        self.assertEqual(trainable.evaluate_validation_set.call_count, 0)

    def test_train_multiple_epoch_check_recorded(self):
        trainer = SimpleTrainer()
        trainable = MagicMock()
        trainable.update_params.return_value = (5.0, 15.5)
        train_watcher = _TrainWatcherRecorder()

        epochs, validation_epochs = trainer.train(
            trainable,
            None,
            None,
            None,
            None,
            train_watcher,
            epochs=3,
        )

        self.assertEqual(epochs, 3)
        self.assertEqual(validation_epochs, 0)
        self.assertEqual(train_watcher.epochs, [0, 1, 2])
        self.assertEqual(train_watcher.costs, [5.0, 5.0, 5.0])
        self.assertEqual(train_watcher.accuracies, [15.5, 15.5, 15.5])
        self.assertEqual(train_watcher.validation_epochs, [])
        self.assertEqual(train_watcher.validation_costs, [])
        self.assertEqual(train_watcher.validation_accuracies, [])

    def test_train_multiple_epoch_check_trainable(self):
        trainer = SimpleTrainer()
        trainable = MagicMock()
        trainable.update_params.return_value = (0.09, 100)

        trainer.train(
            trainable,
            None,
            None,
            None,
            None,
            None,
            epochs=10,
        )

        self.assertEqual(trainable.update_params.call_count, 10)
        self.assertEqual(trainable.evaluate_validation_set.call_count, 0)

    def test_train_with_validation_check_recorded(self):
        trainer = SimpleTrainer()
        trainable = MagicMock()
        trainable.update_params.return_value = (0.66, 80.3)
        trainable.evaluate_validation_set.return_value = (1.25, 50.1)
        train_watcher = _TrainWatcherRecorder()

        (epochs, validation_epochs) = trainer.train(
            trainable,
            None,
            None,
            None,
            None,
            train_watcher,
            epochs=6,
            validation_gap=2,
        )

        self.assertEqual(epochs, 6)
        self.assertEqual(validation_epochs, 3)
        self.assertEqual(train_watcher.epochs, list(range(6)))
        self.assertEqual(train_watcher.costs, [0.66] * 6)
        self.assertEqual(train_watcher.accuracies, [80.3] * 6)
        self.assertEqual(train_watcher.validation_epochs, [0, 2, 4])
        self.assertEqual(train_watcher.validation_costs, [1.25] * 3)
        self.assertEqual(train_watcher.validation_accuracies, [50.1] * 3)

    def test_train_with_validation_on_odd_gap_check_recorded(self):
        trainer = SimpleTrainer()
        trainable = MagicMock()
        trainable.update_params.return_value = (0.66, 80.3)
        trainable.evaluate_validation_set.return_value = (1.25, 50.1)
        train_watcher = _TrainWatcherRecorder()

        (epochs, validation_epochs) = trainer.train(
            trainable,
            None,
            None,
            None,
            None,
            train_watcher,
            epochs=7,
            validation_gap=5,
        )

        self.assertEqual(epochs, 7)
        self.assertEqual(validation_epochs, 2)
        self.assertEqual(train_watcher.epochs, list(range(7)))
        self.assertEqual(train_watcher.costs, [0.66] * 7)
        self.assertEqual(train_watcher.accuracies, [80.3] * 7)
        self.assertEqual(train_watcher.validation_epochs, [0, 5])
        self.assertEqual(train_watcher.validation_costs, [1.25] * 2)
        self.assertEqual(train_watcher.validation_accuracies, [50.1] * 2)

    def test_train_with_validation_check_trainable(self):
        trainer = SimpleTrainer()
        trainable = MagicMock()
        trainable.update_params.return_value = (0.09, 100)
        trainable.evaluate_validation_set.return_value = (0.10, 60.0)

        trainer.train(
            trainable,
            None,
            None,
            None,
            None,
            None,
            epochs=11,
            validation_gap=5,
        )

        self.assertEqual(trainable.update_params.call_count, 11)
        self.assertEqual(trainable.evaluate_validation_set.call_count, 3)

    def test_train_assert_data_is_passed(self):
        trainer = SimpleTrainer()
        trainable = MagicMock()
        trainable.update_params.return_value = (0.66, 80.3)

        trainer.train(
            trainable,
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([0, 1]),
            None,
            None,
            None,
            epochs=3,
        )

        (train_dataset, train_labels), _ = trainable.update_params.call_args
        self.assertTrue(np.array_equal(
            train_dataset,
            np.array([[1, 2, 3], [4, 5, 6]]),
        ))
        self.assertTrue(np.array_equal(
            train_labels,
            np.array([0, 1]),
        ))

    def test_train_with_validation_assert_data_is_passed(self):
        trainer = SimpleTrainer()
        trainable = MagicMock()
        trainable.update_params.return_value = (0.66, 80.3)
        trainable.evaluate_validation_set.return_value = (1.25, 50.1)

        trainer.train(
            trainable,
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array([1.0, 0.0]),
            np.array([[2.0, 4.0, 6.0]]),
            np.array([0.5]),
            None,
            epochs=5,
            validation_gap=2,
        )

        (validation_dataset, validation_labels), _ = \
            trainable.evaluate_validation_set.call_args
        self.assertTrue(np.array_equal(
            validation_dataset,
            np.array([[2.0, 4.0, 6.0]]),
        ))
        self.assertTrue(np.array_equal(
            validation_labels,
            np.array([0.5]),
        ))
