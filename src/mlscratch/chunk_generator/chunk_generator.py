from typing import Iterator, Tuple
from abc import ABC, abstractmethod

from mlscratch.tensor import Tensor


class ChunkGenerator(ABC):
    """Chunk (or mini batch) generator base class."""

    @abstractmethod
    def generate(
            self,
            dataset: Tensor,
            labels: Tensor,
            chunk_size: int) -> Tuple[
                int, Iterator[Tuple[Tensor, Tensor]]]:
        """Generates small chunks of data from a bigger dataset."""
