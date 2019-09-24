"""A Tensor is an abstraction of a data structure ML models can be built over.
It represents a multidimensional vector."""
from typing import Union
import numpy

Tensor = Union[numpy.ndarray]
