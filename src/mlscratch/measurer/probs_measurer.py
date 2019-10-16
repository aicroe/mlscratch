"""ProbsMeasurer's module."""
import numpy as np
from mlscratch.tensor import Tensor
from .measurer import Measurer


class ProbsMeasurer(Measurer[float]):
    """Computes how many samples were evaluated correctly by
    getting the most probable label/index in the probability array."""

    def measure(
            self,
            result: Tensor,
            expected: Tensor) -> float:
        batch_size, *_ = result.shape
        result_max_indices = np.argmax(result, axis=-1)
        expected_max_indices = np.argmax(expected, axis=-1)
        asserts = np.sum(result_max_indices == expected_max_indices)
        return asserts / batch_size
