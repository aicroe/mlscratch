""""Model test suit."""
from unittest import TestCase
from unittest.mock import MagicMock

from mlscratch import Model


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

    def test_train_assert_hooks_are_called(self):
        arch = MagicMock()
        model = Model(arch)
        trainer = MagicMock()
        trainer.train.return_value = (None, None)

        model.train(
            None,
            None,
            None,
            None,
            trainer,
            None,
            None,
            epochs=0,
        )

        self.assertEqual(arch.train_initialize.call_count, 1)
        self.assertEqual(arch.train_finalize.call_count, 1)
