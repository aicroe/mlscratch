""""Model test suit."""
from unittest import TestCase
from unittest.mock import MagicMock
import numpy as np

from mlscratch.model import Model
from mlscratch.trainer import SimpleTrainer, SgdTrainer
from mlscratch.measurer import AssertionsMeasurer
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

    def test_train_with_simple_trainer_check_accuracies(self):
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
            AssertionsMeasurer(),
            None,
            epochs=7,
        )

        self.assertEqual(accuracies, [1/3] * 7)

    def test_train_with_simple_trainer_and_validation_check_accuracies(self):
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

        result = model.train(
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
            AssertionsMeasurer(),
            None,
            epochs=8,
            validation_gap=4,
        )

        self.assertEqual(result[2], [1/2] * 8)
        self.assertEqual(result[5], [0] * 2)

    def test_train_with_sgd_trainer_check_accuracies(self):
        arch = MagicMock()
        arch.update_params.return_value = (
            0.0,
            np.array([
                [10, 7.8, -3.1],
                [2.4, 60, 0.6],
                [11, 9.2, 45.3],
            ]),
        )
        model = Model(arch)
        trainer = SgdTrainer()

        _, _, accuracies, *_ = model.train(
            np.array([None, None, None]),
            np.array([
                [11, 6.4, 0.77],
                [99.9, 99.3, 44.5],
                [10, 1.2, 19.5],
            ]),
            None,
            None,
            trainer,
            AssertionsMeasurer(),
            None,
            epochs=5,
            seed=0,
        )

        # Remember: Dataset and labels get shuffled, permutation indices:
        # epoch 0: [2, 1, 0]
        # epoch 1: [2, 0, 1]
        # epoch 2: [0, 2, 1]
        # epoch 3: [2, 0, 1]
        # epoch 4: [2, 1, 0]
        self.assertEqual(accuracies, [0, 0, 1/3, 0, 0])

    def test_train_with_sgd_trainer_and_validation_check_accuracies(self):
        arch = MagicMock()
        arch.update_params.return_value = (
            0.0,
            np.array([
                [11, 0.8, 33.1],
                [20, 21.4, 7.6],
                [0.2, 45.3, 91],
            ]),
        )
        arch.check_cost.return_value = (
            0.0,
            np.array([
                [0.2, 0.1, 0],
                [0.8, 0.9, 0.2],
                [0.0, 2.1, 0.1],
            ])
        )
        model = Model(arch)
        trainer = SgdTrainer()

        _, _, accuracies, _, _, validation_accuracies = model.train(
            np.array([None, None, None]),
            np.array([
                [16, 0.67, 9],
                [0.3, 0.7, 0.1],
                [0.3, -34, 1],
            ]),
            np.array([]),
            np.array([
                [56, 3, 0],
                [22, 111, 0.2],
                [0.4, 0.1, 0.6],
            ]),
            trainer,
            AssertionsMeasurer(),
            None,
            epochs=3,
            validation_gap=2,
            seed=0,
        )

        # Remember: Dataset and labels get shuffled, permutation indices:
        # epoch 0: [2, 1, 0]
        # epoch 1: [2, 0, 1]
        # epoch 2: [0, 2, 1]
        self.assertEqual(accuracies, [2/3, 1/3, 0])
        self.assertEqual(validation_accuracies, [2/3] * 2)

    def test_train_with_sgd_trainer_and_minibatches_check_accuracies(self):
        arch = MagicMock()
        arch.update_params.return_value = (
            0.0,
            np.array([
                [11, 0.8, 33.1],
                [20, 21.4, 7.6],
            ]),
        )
        arch.check_cost.return_value = (
            0.0,
            np.array([
                [0.2, 0.1, 0],
                [0.8, 0.9, 0.2],
                [0.0, 2.1, 0.1],
                [20, 21.4, 7.6],
            ])
        )
        model = Model(arch)
        trainer = SgdTrainer()

        _, _, accuracies, _, _, validation_accuracies = model.train(
            np.array([None, None, None, None]),
            np.array([
                [16, 0.67, 9],
                [0.3, 0.7, 0.1],
                [0.3, -34, 1],
                [20, 21.4, 7.6],
            ]),
            np.array([]),
            np.array([
                [20, 21.4, 7.6],
                [56, 3, 0],
                [22, 111, 0.2],
                [0.4, 0.1, 0.6],
            ]),
            trainer,
            AssertionsMeasurer(),
            None,
            epochs=4,
            validation_gap=1,
            minibatch_size=2,
            seed=0,
        )

        # Remember: Dataset and labels get shuffled, permutation indices:
        # epoch 0: [2, 3, 1, 0]
        # epoch 1: [0, 2, 1, 3]
        # epoch 2: [3, 0, 2, 1]
        # epoch 2: [1, 0, 2, 3]
        self.assertEqual(accuracies, [1/2, 1/4, 0.5, 0.5])
        self.assertEqual(validation_accuracies, [1/4] * 4)
