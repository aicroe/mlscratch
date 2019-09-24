"""Arch's module."""
from typing import Tuple
from abc import ABC, abstractmethod

from mlscratch.tensor import Tensor


class Arch(ABC):
    """Abstracts a machine learning instance."""

    @abstractmethod
    def update_params(
            self,
            dataset: Tensor,
            labels: Tensor) -> Tuple[float, Tensor]:
        """Updates/Optimizes its trainable parameters."""

    @abstractmethod
    def evaluate(self, dataset: Tensor) -> Tensor:
        """Runs this instance with the given dataset as
        input and returns the results."""

    @abstractmethod
    def check_cost(
            self,
            dataset: Tensor,
            labels: Tensor) -> Tuple[float, Tensor]:
        """Runs this instance with the given dataset as
        input and computes the cost on the provided labels."""
