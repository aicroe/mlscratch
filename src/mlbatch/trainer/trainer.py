from typing import Tuple, Optional
from abc import ABC, abstractmethod

from mlbatch.tensor import Tensor
from mlbatch.trainable import Trainable
from mlbatch.train_watcher import TrainWatcher


class Trainer(ABC):
    """Abstracts an object that can train a trainable instance."""

    @abstractmethod
    def train(
            self,
            trainable: Trainable,
            train_dataset: Tensor,
            train_labels: Tensor,
            validation_dataset: Optional[Tensor],
            validation_labels: Optional[Tensor],
            train_watcher: Optional[TrainWatcher],
            **options) -> Tuple[int, int]:
        """Trains a trainable instance, this should implement
        a optimization algorithm.

        Returns a tuple: (epochs, validation_epochs)
        """
