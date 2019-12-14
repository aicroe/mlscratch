"""SgdEarlyStopTrainer's test suit."""
from unittest import TestCase
from unittest.mock import MagicMock
import numpy as np

from mlscratch.trainer import SgdEarlyStopTrainer
from ..test_helper import _TrainWatcherRecorder, _PlaybackTrainable


class SgdEarlyStopTrainerTest(TestCase):

    def test_train_check_results_are_empty_or_zero(self):
        trainer = SgdEarlyStopTrainer()
        trainable = MagicMock()
        train_watcher = _TrainWatcherRecorder()

        epochs, validation_epochs = trainer.train(
            trainable,
            np.array([1, 2, 3]),
            np.array([0, 1, 0]),
            None,
            None,
            train_watcher,
        )

        self.assertEqual(epochs, 0)
        self.assertEqual(validation_epochs, 0)

        self.assertEqual(train_watcher.epochs, [])
        self.assertEqual(train_watcher.costs, [])
        self.assertEqual(train_watcher.accuracies, [])
        self.assertEqual(train_watcher.validation_epochs, [])
        self.assertEqual(train_watcher.validation_costs, [])
        self.assertEqual(train_watcher.validation_accuracies, [])

    def test_train_single_epoch_check_recorded_and_trainable(self):
        trainer = SgdEarlyStopTrainer()
        trainable = MagicMock()
        trainable.update_params.return_value = (0.9, 55.0)
        train_watcher = _TrainWatcherRecorder()

        (epochs, validation_epochs) = trainer.train(
            trainable,
            np.array([1, 2, 3]),
            np.array([1, 0, 1]),
            None,
            None,
            train_watcher,
            epochs=1,
        )

        self.assertEqual(epochs, 1)
        self.assertEqual(validation_epochs, 0)
        self.assertEqual(train_watcher.epochs, [0])
        self.assertEqual(train_watcher.costs, [0.9])
        self.assertEqual(train_watcher.accuracies, [55.0])
        self.assertEqual(train_watcher.validation_epochs, [])
        self.assertEqual(train_watcher.validation_costs, [])
        self.assertEqual(train_watcher.validation_accuracies, [])

        self.assertEqual(trainable.update_params.call_count, 1)
        self.assertEqual(trainable.evaluate_validation_set.call_count, 0)

    def test_train_multiple_epoch_check_recorded_and_trainable(self):
        trainer = SgdEarlyStopTrainer()
        trainable = MagicMock()
        trainable.update_params.return_value = (5.0, 15.5)
        train_watcher = _TrainWatcherRecorder()

        epochs, validation_epochs = trainer.train(
            trainable,
            np.array([1, 2, 3]),
            np.array([1, 0, 1]),
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

        self.assertEqual(trainable.update_params.call_count, 3)
        self.assertEqual(trainable.evaluate_validation_set.call_count, 0)

    def test_train_with_validation_check_recorded(self):
        trainer = SgdEarlyStopTrainer()
        trainable = MagicMock()
        trainable.update_params.return_value = (0.66, 80.3)
        trainable.evaluate_validation_set.return_value = (1.25, 50.1)
        train_watcher = _TrainWatcherRecorder()

        (epochs, validation_epochs) = trainer.train(
            trainable,
            np.array([1, 2, 3]),
            np.array([1, 0, 1]),
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

    def test_train_with_multiple_minibatches_check_recorded_and_trainable(self):
        trainer = SgdEarlyStopTrainer()
        trainable = MagicMock()
        trainable.update_params.return_value = (1.2, 70.5)
        train_watcher = _TrainWatcherRecorder()

        (epochs, validation_epochs) = trainer.train(
            trainable,
            np.array([1, 2, 3]),
            np.array([0.1, 0.2, 0.3]),
            None,
            None,
            train_watcher,
            epochs=5,
            minibatch_size=1,
        )

        self.assertEqual(epochs, 5)
        self.assertEqual(validation_epochs, 0)
        self.assertEqual(train_watcher.epochs, list(range(5)))
        self.assertEqual(train_watcher.costs, [1.2] * 5)
        self.assertEqual(train_watcher.accuracies, [70.5] * 5)
        self.assertEqual(train_watcher.validation_epochs, [])
        self.assertEqual(train_watcher.validation_costs, [])
        self.assertEqual(train_watcher.validation_accuracies, [])

        self.assertEqual(trainable.update_params.call_count, 15)
        self.assertEqual(trainable.evaluate_validation_set.call_count, 0)

    def test_train_check_it_stops_early(self):
        trainer = SgdEarlyStopTrainer()
        trainable = MagicMock()
        trainable.update_params.return_value = (0.66, 80.3)
        trainable.evaluate_validation_set.return_value = (1.25, 50.1)
        train_watcher = _TrainWatcherRecorder()

        (epochs, validation_epochs) = trainer.train(
            trainable,
            np.array([10, 20, 30]),
            np.array([0, 0, 0]),
            None,
            None,
            train_watcher,
            epochs=15,
            validation_gap=2,
        )

        self.assertEqual(epochs, 11)
        self.assertEqual(validation_epochs, 6)
        self.assertEqual(train_watcher.epochs, list(range(11)))
        self.assertEqual(train_watcher.costs, [0.66] * 11)
        self.assertEqual(train_watcher.accuracies, [80.3] * 11)
        self.assertEqual(train_watcher.validation_epochs, [0, 2, 4, 6, 8, 10])
        self.assertEqual(train_watcher.validation_costs, [1.25] * 6)
        self.assertEqual(train_watcher.validation_accuracies, [50.1] * 6)

        self.assertEqual(trainable.update_params.call_count, 11)
        self.assertEqual(trainable.evaluate_validation_set.call_count, 6)

    def test_train_with_minibatches_check_it_stops_early(self):
        trainer = SgdEarlyStopTrainer()
        trainable = MagicMock()
        trainable.update_params.return_value = (0.66, 80.3)
        trainable.evaluate_validation_set.return_value = (1.25, 50.1)
        train_watcher = _TrainWatcherRecorder()

        (epochs, validation_epochs) = trainer.train(
            trainable,
            np.array([10, 20, 30]),
            np.array([0, 0, 0]),
            None,
            None,
            train_watcher,
            epochs=15,
            validation_gap=2,
            minibatch_size=2,
        )

        self.assertEqual(epochs, 11)
        self.assertEqual(validation_epochs, 6)
        self.assertEqual(train_watcher.epochs, list(range(11)))
        self.assertEqual(train_watcher.costs, [0.66] * 11)
        self.assertEqual(train_watcher.accuracies, [80.3] * 11)
        self.assertEqual(train_watcher.validation_epochs, [0, 2, 4, 6, 8, 10])
        self.assertEqual(train_watcher.validation_costs, [1.25] * 6)
        self.assertEqual(train_watcher.validation_accuracies, [50.1] * 6)

        self.assertEqual(trainable.update_params.call_count, 22)
        self.assertEqual(trainable.evaluate_validation_set.call_count, 6)

    def test_train_with_small_patience_check_it_stops_early(self):
        trainer = SgdEarlyStopTrainer()
        trainable = MagicMock()
        trainable.update_params.return_value = (0.33, 81)
        trainable.evaluate_validation_set.return_value = (5.4, 99.0)
        train_watcher = _TrainWatcherRecorder()

        (epochs, validation_epochs) = trainer.train(
            trainable,
            np.array([1, 2, 3]),
            np.array([1, 1, 1]),
            None,
            None,
            train_watcher,
            epochs=10,
            validation_gap=1,
            patience=2,
        )

        self.assertEqual(epochs, 3)
        self.assertEqual(validation_epochs, 3)
        self.assertEqual(train_watcher.epochs, list(range(3)))
        self.assertEqual(train_watcher.costs, [0.33] * 3)
        self.assertEqual(train_watcher.accuracies, [81] * 3)
        self.assertEqual(train_watcher.validation_epochs, list(range(3)))
        self.assertEqual(train_watcher.validation_costs, [5.4] * 3)
        self.assertEqual(train_watcher.validation_accuracies, [99.0] * 3)

        self.assertEqual(trainable.update_params.call_count, 3)
        self.assertEqual(trainable.evaluate_validation_set.call_count, 3)

    def test_train_having_decreasing_cost_check_it_does_not_stops_early(self):
        trainer = SgdEarlyStopTrainer()
        validation_costs = [10.0, 9.8, 11.0, 0.5, 0.4, 0.7, 0.8, 0.1, 0.01, 0.001]
        trainable = _PlaybackTrainable(
            [0.1] * 30,
            [90.0] * 30,
            validation_costs,
            [89.0] * 10,
        )
        train_watcher = _TrainWatcherRecorder()

        (epochs, validation_epochs) = trainer.train(
            trainable,
            np.array([10, 21, 32]),
            np.array([0, 0, 0]),
            None,
            None,
            train_watcher,
            epochs=10,
            validation_gap=1,
            minibatch_size=1,
            patience=3,
        )

        self.assertEqual(epochs, 10)
        self.assertEqual(validation_epochs, 10)
        self.assertEqual(train_watcher.epochs, list(range(10)))
        self.assertEqual(train_watcher.costs, [0.1] * 10)
        self.assertEqual(train_watcher.accuracies, [90.0] * 10)
        self.assertEqual(train_watcher.validation_epochs, list(range(10)))
        self.assertEqual(train_watcher.validation_costs, validation_costs)
        self.assertEqual(train_watcher.validation_accuracies, [89.0] * 10)

        self.assertEqual(trainable.update_params_call_count, 30)
        self.assertEqual(trainable.evaluate_validation_set_call_count, 10)

    def test_train_it_should_stop_as_soon_patience_is_surpassed(self):
        trainer = SgdEarlyStopTrainer()
        trainable = _PlaybackTrainable(
            [0.1] * 9,
            [90.0] * 9,
            [0.1, 0.2, 0.05],
            [89.0] * 3,
        )
        train_watcher = _TrainWatcherRecorder()

        (epochs, validation_epochs) = trainer.train(
            trainable,
            np.array([3, 2, 1]),
            np.array([1, 1, 0]),
            None,
            None,
            train_watcher,
            epochs=3,
            validation_gap=1,
            minibatch_size=1,
            patience=1,
        )

        self.assertEqual(epochs, 2)
        self.assertEqual(validation_epochs, 2)
        self.assertEqual(train_watcher.epochs, list(range(2)))
        self.assertEqual(train_watcher.costs, [0.1] * 2)
        self.assertEqual(train_watcher.accuracies, [90.0] * 2)
        self.assertEqual(train_watcher.validation_epochs, list(range(2)))
        self.assertEqual(train_watcher.validation_costs, [0.1, 0.2])
        self.assertEqual(train_watcher.validation_accuracies, [89.0] * 2)

        self.assertEqual(trainable.update_params_call_count, 6)
        self.assertEqual(trainable.evaluate_validation_set_call_count, 2)

    def test_train_assert_data_is_passed_and_shuffled(self):
        trainer = SgdEarlyStopTrainer()
        trainable = MagicMock()
        trainable.update_params.return_value = (0.66, 80.3)
        trainable.evaluate_validation_set.return_value = (1.25, 50.1)

        epochs, validation_epochs = trainer.train(
            trainable,
            np.array([1, 2, 3, 4, 5]),
            np.array([0, 1, 0, 1, 0]),
            np.array([1, 5]),
            np.array([0, 0]),
            None,
            epochs=6,
            validation_gap=2,
            minibatch_size=10,
            patience=2,
            seed=0,
        )

        self.assertEqual(epochs, 5)
        self.assertEqual(validation_epochs, 3)

        (train_dataset, train_labels), _ = trainable.update_params.call_args
        self.assertTrue(np.array_equal(
            train_dataset,
            np.array([5, 3, 4, 2, 1]),
        ))
        self.assertTrue(np.array_equal(
            train_labels,
            np.array([0, 0, 1, 1, 0]),
        ))

        (validation_dataset, validation_labels), _ = trainable.evaluate_validation_set.call_args
        self.assertTrue(np.array_equal(
            validation_dataset,
            np.array([1, 5]),
        ))
        self.assertTrue(np.array_equal(
            validation_labels,
            np.array([0, 0]),
        ))

    def test_default_patience_can_be_set_in_the_constructor(self):
        trainer = SgdEarlyStopTrainer(1)
        trainable = _PlaybackTrainable(
            [0.1] * 9,
            [90.0] * 9,
            [0.1, 0.2, 0.05],
            [89.0] * 3,
        )
        train_watcher = _TrainWatcherRecorder()

        (epochs, validation_epochs) = trainer.train(
            trainable,
            np.array([3, 2, 1]),
            np.array([1, 1, 0]),
            None,
            None,
            train_watcher,
            epochs=3,
            validation_gap=1,
            minibatch_size=1,
        )

        self.assertEqual(epochs, 2)
        self.assertEqual(validation_epochs, 2)
        self.assertEqual(train_watcher.epochs, list(range(2)))
        self.assertEqual(train_watcher.costs, [0.1] * 2)
        self.assertEqual(train_watcher.accuracies, [90.0] * 2)
        self.assertEqual(train_watcher.validation_epochs, list(range(2)))
        self.assertEqual(train_watcher.validation_costs, [0.1, 0.2])
        self.assertEqual(train_watcher.validation_accuracies, [89.0] * 2)
