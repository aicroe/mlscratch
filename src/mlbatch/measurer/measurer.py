from typing import Generic, TypeVar
from abc import abstractmethod

from mlbatch.tensor import Tensor


T = TypeVar('T')

class Measurer(Generic[T]):
    """Abstracts an object that measures the accuracy of the
    real output against the expected one."""

    @abstractmethod
    def measure(
            self,
            result: Tensor,
            expected: Tensor) -> T:
        """Measures the accuracy of the real output against
        the expected one."""
