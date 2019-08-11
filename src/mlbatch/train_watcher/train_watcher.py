from abc import ABC, abstractmethod


class TrainWatcher(ABC):
    """Abstracts an observer object that would listen
    a train process step by step."""

    @abstractmethod
    def on_epoch(
            self,
            epoch: int,
            cost: float,
            accuracy: float):
        """Called when a train epoch is performed."""

    @abstractmethod
    def on_validation_epoch(
            self,
            epoch: int,
            cost: float,
            accuracy: float):
        """Called when a validation train epoch is performed."""
