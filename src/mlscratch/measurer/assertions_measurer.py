import numpy as np
from mlscratch.tensor import Tensor
from .measurer import Measurer


class AssertionsMeasurer(Measurer[float]):

    def measure(
            self,
            result: Tensor,
            expected: Tensor) -> float:
        batch_size, *_ = result.shape
        result_max_indices = np.argmax(result, axis=-1)
        expected_max_indices = np.argmax(expected, axis=-1)
        asserts = np.sum(result_max_indices == expected_max_indices)
        return asserts / batch_size
