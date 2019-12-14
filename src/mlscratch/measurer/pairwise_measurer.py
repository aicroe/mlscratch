"""PairwiseMeasurer's module."""
import numpy as np

from mlscratch.tensor import Tensor
from .measurer import Measurer

class PairwiseMeasurer(Measurer[float]):
    """Compares the actual result against the expected element by element."""

    def measure(self, result: Tensor, expected: Tensor):
        return np.average(result == expected)
