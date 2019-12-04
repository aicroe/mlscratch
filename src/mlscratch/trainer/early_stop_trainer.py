"""EarlyStopTrainer's module."""
from typing import Tuple, Optional
import math

from mlscratch.tensor import Tensor
from mlscratch.trainable import Trainable
from mlscratch.train_watcher import TrainWatcher
from .trainer import Trainer


class EarlyStopTrainer(Trainer):
    """Model optimizer, implements regular Gradient Descent algorithm
    plus Early Stop regularization."""

    def __init__(self, patience=5):
        self._patience = patience

    def train(
            self,
            trainable: Trainable,
            train_dataset: Tensor,
            train_labels: Tensor,
            validation_dataset: Optional[Tensor],
            validation_labels: Optional[Tensor],
            train_watcher: Optional[TrainWatcher],
            **options) -> Tuple[int, int]:
        epochs: int = options.get('epochs', 0)
        validation_gap: int = options.get('validation_gap', 0)
        patience: int = options.get('patience', self._patience)

        validation_epochs = 0
        iterations_failing = 0
        current_val_cost = math.inf

        for epoch in range(epochs):
            cost, accuracy = trainable.update_params(
                train_dataset,
                train_labels,
            )

            if train_watcher is not None:
                train_watcher.on_epoch(epoch, cost, accuracy)

            if validation_gap > 0 and epoch % validation_gap == 0:
                (validation_cost,
                 validation_accuracy,
                ) = trainable.evaluate_validation_set(
                    validation_dataset,
                    validation_labels,
                )
                validation_epochs += 1

                if train_watcher is not None:
                    train_watcher.on_validation_epoch(
                        epoch,
                        validation_cost,
                        validation_accuracy,
                    )

                if validation_cost < current_val_cost:
                    iterations_failing = 1
                else:
                    iterations_failing += 1
                current_val_cost = validation_cost

            if iterations_failing > patience:
                return epoch + 1, validation_epochs

        return epochs, validation_epochs
