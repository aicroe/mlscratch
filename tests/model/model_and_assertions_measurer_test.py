"""Model and AssertionsMeasurer integration tests suit."""
from unittest import TestCase
from unittest.mock import MagicMock
import numpy as np

from mlscratch import Model
from mlscratch.measurer import AssertionsMeasurer
from mlscratch.trainer import SimpleTrainer, SgdTrainer


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
