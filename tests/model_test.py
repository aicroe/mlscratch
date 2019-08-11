""""Model test suit."""
from unittest import TestCase
from unittest.mock import MagicMock
import numpy as np

from mlbatch.model import Model
from mlbatch.trainer import SimpleTrainer, SgdTrainer
from .test_helper import _TrainWatcherRecorder


class ModelTest(TestCase):

    def test_evaluate_assert_it_returns_prob_array(self):
        arch = MagicMock()
        arch.evaluate.return_value = [0.1, 0.8, 0.1]
        model = Model(arch)

        result = model.evaluate(None)

        self.assertEqual(result, [0.1, 0.8, 0.1])

    def test_measure_assert_returned_cost(self):
        arch = MagicMock()
        arch.check_cost.return_value = (0.5, None)
        model = Model(arch)

        cost, measures = model.measure(None, None, [])

        self.assertEqual(cost, 0.5)
        self.assertEqual(measures, [])

    def test_measure_assert_returned_measures(self):
        arch = MagicMock()
        arch.check_cost.return_value = (None, None)
        model = Model(arch)
        measurer1 = MagicMock()
        measurer2 = MagicMock()
        measurer1.measure.return_value = 56.9
        measurer2.measure.return_value = 70.1

        (_, results) = model.measure(None, None, [measurer1, measurer2])

        self.assertEqual(results, [56.9, 70.1])

    def test_restore_assert_arch_gets_called(self):
        arch = MagicMock()
        model = Model(arch)

        model.restore()

        self.assertEqual(arch.restore.call_count, 1)

    def test_save_assert_arch_gets_called(self):
        arch = MagicMock()
        model = Model(arch)

        model.save()

        self.assertEqual(arch.save.call_count, 1)


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


class ModelAndSgdTrainerIntegrationTest(TestCase):

    def test_train_check_returned(self):
        arch = MagicMock()
        arch.update_params.return_value = (0.4, None)
        model = Model(arch)
        trainer = SgdTrainer()
        measurer = MagicMock()
        measurer.measure.return_value = 60.5

        (epochs,
         costs,
         accuracies,
         validation_epochs,
         validation_costs,
         validation_accuracies,
        ) = model.train(
            np.array([None]),
            np.array([None]),
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
        trainer = SgdTrainer()
        measurer = MagicMock()
        measurer.measure.return_value = 33.8

        (epochs,
         costs,
         accuracies,
         validation_epochs,
         validation_costs,
         validation_accuracies,
        ) = model.train(
            np.array([None]),
            np.array([None]),
            np.array([None]),
            np.array([None]),
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

    def test_train_with_minibatches_check_returned(self):
        arch = MagicMock()
        arch.update_params.return_value = (16.7, None)
        arch.check_cost.return_value = (24.6, None)
        model = Model(arch)
        trainer = SgdTrainer()
        measurer = MagicMock()
        measurer.measure.return_value = 33.8

        (epochs,
         costs,
         accuracies,
         validation_epochs,
         validation_costs,
         validation_accuracies,
        ) = model.train(
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3]),
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3]),
            np.array([None]),
            np.array([None]),
            trainer,
            measurer,
            None,
            epochs=13,
            validation_gap=4,
            minibatch_size=6,
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
        trainer = SgdTrainer()
        measurer = MagicMock()
        measurer.measure.return_value = 60.5
        train_watcher = _TrainWatcherRecorder()

        model.train(
            np.array([None]),
            np.array([None]),
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
        trainer = SgdTrainer()
        measurer = MagicMock()
        measurer.measure.return_value = 90.5
        train_watcher = _TrainWatcherRecorder()

        model.train(
            np.array([None]),
            np.array([None]),
            np.array([None]),
            np.array([None]),
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

    def test_train_with_minibatches_check_recorded(self):
        arch = MagicMock()
        arch.update_params.return_value = (0.1, None)
        arch.check_cost.return_value = (0.66, None)
        model = Model(arch)
        trainer = SgdTrainer()
        measurer = MagicMock()
        measurer.measure.return_value = 90.5
        train_watcher = _TrainWatcherRecorder()

        model.train(
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2]),
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2]),
            np.array([None]),
            np.array([None]),
            trainer,
            measurer,
            train_watcher,
            epochs=9,
            validation_gap=3,
            minibatch_size=3,
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
        trainer = SgdTrainer()
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
            np.array([None]),
            np.array([None]),
            np.array([None]),
            np.array([None]),
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

    def test_train_with_minibathes_assert_recorded_and_returned_are_identical(self):
        arch = MagicMock()
        arch.update_params.return_value = (0.1, None)
        arch.check_cost.return_value = (0.66, None)
        model = Model(arch)
        trainer = SgdTrainer()
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
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
            np.array([None]),
            np.array([None]),
            trainer,
            measurer,
            train_watcher,
            epochs=11,
            validation_gap=1,
            minibatch_size=4,
        )

        self.assertEqual(train_watcher.epochs, list(range(epochs)))
        self.assertEqual(train_watcher.costs, costs)
        self.assertEqual(train_watcher.accuracies, accuracies)
        self.assertEqual(len(train_watcher.validation_epochs), validation_epochs)
        self.assertEqual(train_watcher.validation_costs, validation_costs)
        self.assertEqual(train_watcher.validation_accuracies, validation_accuracies)
