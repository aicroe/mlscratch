"""Arch's module."""
from typing import Tuple
from abc import ABC, abstractmethod

from mlscratch.tensor import Tensor


class Arch(ABC):
    """Abstracts a machine learning instance."""

    def train_initialize(self) -> None:
        """Train hook. Called once before the training has started."""

    def train_finalize(self) -> None:
        """Train hook. Called once after the training has finalized."""

    @abstractmethod
    def update_params(
            self,
            dataset: Tensor,
            labels: Tensor) -> Tuple[float, Tensor]:
        """Updates/Optimizes its trainable parameters.
        Called while training to update this instance parameters."""

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
