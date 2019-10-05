"""Model's module."""
from typing import Tuple, List, Any, Optional

from .arch import Arch
from .tensor import Tensor
from .trainer import Trainer
from .measurer import Measurer
from .trainable import Trainable
from .train_watcher import TrainWatcher


class _TrainableArchAdapter(Trainable):

    def __init__(self, arch: Arch, measurer: Measurer[float]):
        self._arch = arch
        self._measurer = measurer

    def update_params(
            self,
            dataset: Tensor,
            labels: Tensor) -> Tuple[float, float]:
        cost, evaluations = self._arch.update_params(dataset, labels)
        accuracy = self._measurer.measure(evaluations, labels)
        return cost, accuracy

    def evaluate_validation_set(
            self,
            validation_dataset: Tensor,
            validation_labels: Tensor) -> Tuple[float, float]:
        cost, evaluations = self._arch.check_cost(
            validation_dataset,
            validation_labels,
        )
        accuracy = self._measurer.measure(evaluations, validation_labels)
        return cost, accuracy


class _TrainRecorder(TrainWatcher):

    def __init__(self, train_watcher: TrainWatcher):
        self.train_watcher = train_watcher
        self.costs = []
        self.accuracies = []
        self.validation_costs = []
        self.validation_accuracies = []

    def on_epoch(
            self,
            epoch: int,
            cost: float,
            accuracy: float):
        self.costs.append(cost)
        self.accuracies.append(accuracy)
        if self.train_watcher is not None:
            self.train_watcher.on_epoch(epoch, cost, accuracy)

    def on_validation_epoch(
            self,
            epoch: int,
            cost: float,
            accuracy: float):
        self.validation_costs.append(cost)
        self.validation_accuracies.append(accuracy)
        if self.train_watcher is not None:
            self.train_watcher.on_validation_epoch(
                epoch,
                cost,
                accuracy,
            )


class Model():
    """Abstracts an object that leverages different entities
    involved in the operations this class publishes.

    Can train, measure, run evaluations, save and restore
    a machine learning instance (encapsulated in the class Arch).
    """

    def __init__(self, arch: Arch):
        self._arch = arch

    def train(
            self,
            dataset: Tensor,
            labels: Tensor,
            validation_dataset: Optional[Tensor],
            validation_labels: Optional[Tensor],
            trainer: Trainer,
            measurer: Measurer[float],
            train_watcher: Optional[TrainWatcher],
            **options) -> Tuple[int, List[float], List[float],
                                int, List[float], List[float]]:
        """Trains this model's arch."""
        train_recorder = _TrainRecorder(train_watcher)
        self._arch.train_initialize()
        epochs, validation_epochs = trainer.train(
            _TrainableArchAdapter(self._arch, measurer),
            dataset,
            labels,
            validation_dataset,
            validation_labels,
            train_recorder,
            **options,
        )
        self._arch.train_finalize()
        return (epochs,
                train_recorder.costs,
                train_recorder.accuracies,
                validation_epochs,
                train_recorder.validation_costs,
                train_recorder.validation_accuracies)

    def evaluate(self, dataset: Tensor) -> Tensor:
        """Evaluates an input dataset."""
        return self._arch.evaluate(dataset)

    def measure(
            self,
            dataset: Tensor,
            labels: Tensor,
            measurers: List[Measurer[Any]]) -> Tuple[float, List[Any]]:
        """Measures the evaluations against expected labels."""
        cost, evaluations = self._arch.check_cost(dataset, labels)
        measures = [measurer.measure(evaluations, labels) for measurer in measurers]
        return (cost, measures)
