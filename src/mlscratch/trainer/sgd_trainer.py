from typing import Tuple, Iterator, Optional

from mlscratch.tensor import Tensor
from mlscratch.trainable import Trainable
from mlscratch.chunk_generator import RandomMiniBatchGenerator
from mlscratch.train_watcher import TrainWatcher
from .trainer import Trainer


def _run_epoch_with_minibatches(
        trainable: Trainable,
        minibatches: Iterator[Tuple[Tensor, Tensor]],
        minibatches_size: int) -> Tuple[float, float]:
    cost = 0.
    accuracy = 0.
    for (mini_dataset, mini_labels) in minibatches:
        minibatch_cost, minibatch_accuracy = trainable.update_params(
            mini_dataset,
            mini_labels,
        )
        cost += minibatch_cost / minibatches_size
        accuracy += minibatch_accuracy / minibatches_size

    return cost, accuracy


class SgdTrainer(Trainer):
    """Stochastic Gradient Descent Trainer."""

    def train(
            self,
            trainable: Trainable,
            train_dataset: Tensor,
            train_labels: Tensor,
            validation_dataset: Optional[Tensor],
            validation_labels: Optional[Tensor],
            train_watcher: Optional[TrainWatcher],
            **options) -> Tuple[int, int]:
        batch_size, *_ = train_dataset.shape
        epochs: int = options.get('epochs', 0)
        validation_gap: int = options.get('validation_gap', 0)
        minibatch_size: int = options.get('minibatch_size', batch_size)
        minibatch_generator = RandomMiniBatchGenerator(options.get('seed', None))

        validation_epochs = 0

        for epoch in range(epochs):
            minibatches_size, minibatches = minibatch_generator.generate(
                train_dataset,
                train_labels,
                minibatch_size,
            )

            cost, accuracy = _run_epoch_with_minibatches(
                trainable,
                minibatches,
                minibatches_size,
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

        return epochs, validation_epochs
