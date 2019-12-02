"""Arch's test suit."""
from unittest import TestCase

from mlscratch.arch import Arch

class _MockArch(Arch):

    def update_params(self, dataset, labels):
        pass

    def evaluate(self, dataset):
        pass

    def check_cost(self, dataset, labels):
        pass


class ArchTest(TestCase):

    def test_train_initialize_is_optional_to_overwrite(self):
        arch = _MockArch()

        self.assertIsNone(arch.train_initialize())

    def test_train_finalize_is_optional_to_overwrite(self):
        arch = _MockArch()

        self.assertIsNone(arch.train_finalize())
