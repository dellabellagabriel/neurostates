# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

"""
This module defines the NeuroData class which contains all the required
data to perform a brain state analysis.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class RawMatrix:
    """
    Represents a raw matrix data with size subjects x regions x time

    Parameters
    ----------
    data: :py:class:`numpy.ndarray`
        A matrix of size subjects x regions x time
    """

    data: NDArray

    def __post_init__(self):
        if self.data.ndim != 3:
            raise ValueError(
                f"""The data matrix must be three-dimensional
                (subjects x regions x time).
                Expected 3 but got {self.data.ndim}."""
            )

    @property
    def subjects(self):
        return self.data.shape[0]

    @property
    def regions(self):
        return self.data.shape[1]

    @property
    def samples(self):
        return self.data.shape[2]


class NeuroImage(ABC):
    @classmethod
    @abstractmethod
    def create_from_matrix(
        cls, matrix: NDArray, *args, **kwargs
    ) -> "NeuroImage": ...

    @classmethod
    @abstractmethod
    def create_from_bids(cls, path: os.PathLike) -> "NeuroImage": ...

    @abstractmethod
    def time_to_samples(self, time: float) -> float: ...

    @abstractmethod
    def samples_to_time(self, samples: float) -> float: ...


class NeuroEEG(NeuroImage):
    def __init__(self, matrix: NDArray, sampling_rate):
        self.data = RawMatrix(matrix)
        self.sampling_rate = sampling_rate

    @classmethod
    def create_from_matrix(cls, matrix, sampling_rate):
        if sampling_rate is None:
            raise ValueError("sampling_rate cannot be None.")
        return NeuroEEG(matrix, sampling_rate)

    @classmethod
    def create_from_bids(cls, path):
        raise NotImplementedError()

    def time_to_samples(self, time):
        return time * self.sampling_rate

    def samples_to_time(self, samples):
        return samples / self.sampling_rate


class NeuroFMRI(NeuroImage):
    def __init__(self, matrix, tr):
        self.data = RawMatrix(matrix)
        self.tr = tr

    @classmethod
    def create_from_matrix(cls, matrix, tr):
        if tr is None:
            raise ValueError("tr cannot be None.")
        return NeuroFMRI(matrix, tr)

    @classmethod
    def create_from_bids(cls, path):
        raise NotImplementedError()

    def time_to_samples(self, time):
        return time / self.tr

    def samples_to_time(self, samples):
        return samples * self.tr
