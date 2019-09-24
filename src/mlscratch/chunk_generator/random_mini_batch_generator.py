"""RandomMiniBatchGenerator's module."""
from typing import Iterator, Tuple, List
import math
import numpy as np

from mlscratch.tensor import Tensor
from .chunk_generator import ChunkGenerator


def _get_split_ranges(
        chunk_size: int,
        num_minibatches: int) -> List[int]:
    return [index * chunk_size for index in range(1, num_minibatches)]


class RandomMiniBatchGenerator(ChunkGenerator):
    """Generates chunks of data in random order,
    though it can be controlled by passing a seed."""

    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

    def generate(
            self,
            dataset: Tensor,
            labels: Tensor,
            chunk_size: int) -> Tuple[
                int, Iterator[Tuple[Tensor, Tensor]]]:
        batch_size, *_ = dataset.shape
        permutation = np.random.permutation(batch_size)
        dataset_shuffled = dataset[permutation]
        labels_shuffled = labels[permutation]
        num_minibatches = math.ceil(batch_size / chunk_size)

        if num_minibatches == 0:
            return 0, zip([], [])

        split_ranges = _get_split_ranges(chunk_size, num_minibatches)
        return num_minibatches, zip(
            np.split(dataset_shuffled, split_ranges, axis=0),
            np.split(labels_shuffled, split_ranges, axis=0),
        )
