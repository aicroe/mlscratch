from typing import Tuple
from abc import ABC, abstractmethod

from mlbatch.tensor import Tensor


class Trainable(ABC):
    """Abstracts a trainable model."""

    @abstractmethod
    def update_params(
            self,
            dataset: Tensor,
            labels: Tensor) -> Tuple[float, float]:
        """Updates instance's parameters.

        Returns a tuple: (cost, accuracy).
        """

    @abstractmethod
    def evaluate_validation_set(
            self,
            validation_dataset: Tensor,
            validation_labels: Tensor) -> Tuple[float, float]:
        """Runs current instance against a validation set,
        intended to work for testing and adjustment purposes.

        Returns a tuple: (cost, accuracy)"""
