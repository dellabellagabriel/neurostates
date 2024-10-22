# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

"""core defines the NeuroData class which contains all the required\
data to perform a brain state analysis."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class RawMatrix:
    """
    Represents a raw matrix data with size subjects x regions x time.

    Parameters
    ----------
    data: :py:class:`numpy.ndarray`
        A matrix of size subjects x regions x time
    """

    data: NDArray

    def __post_init__(self):
        """Validate that the array is three-dimensional."""
        if self.data.ndim != 3:
            raise ValueError(
                f"""The data matrix must be three-dimensional
                (subjects x regions x time).
                Expected 3 but got {self.data.ndim}."""
            )

    @property
    def subjects(self):
        """Number of subjects."""
        return self.data.shape[0]

    @property
    def regions(self):
        """Number of regions."""
        return self.data.shape[1]

    @property
    def samples(self):
        """Number of samples."""
        return self.data.shape[2]


class NeuroImage(ABC):
    """Represents the neuroimage data required for brain state analysis."""

    @classmethod
    @abstractmethod
    def create_from_matrix(
        cls, matrix: NDArray, *args, **kwargs
    ) -> "NeuroImage":
        """Create a NeuroEEG or NeuroFMRI object from a raw numpy matrix."""

    @abstractmethod
    def time_to_samples(self, time: float) -> float:
        """Convert time in seconds to samples."""

    @abstractmethod
    def samples_to_time(self, samples: float) -> float:
        """Convert samples to time in seconds."""


class NeuroEEG(NeuroImage):
    """
    Represents the necessary data for brain state analysis in EEG.

    Parameters
    ----------
    data: :py:class:`RawMatrix`
        The raw matrix representing the data
    sampling_rate: float
        The sampling rate in Hz
    """

    def __init__(self, matrix: NDArray, sampling_rate):
        self.data = RawMatrix(matrix)
        self.sampling_rate = sampling_rate

    @classmethod
    def create_from_matrix(cls, matrix, sampling_rate):
        """Create a NeuroEEG object from a raw numpy matrix.

        Parameters
        ----------
        matrix: :py:class:`numpy.array`
            Raw matrix of size subjects x regions x time
        sampling_rate: float
            Sampling rate of the EEG data

        Returns
        -------
        neuroimage: :py:class:`neurostates.core.NeuroEEG`
            An instance of the class that represents the EEG data.

        """
        if sampling_rate is None:
            raise ValueError("sampling_rate cannot be None.")
        return NeuroEEG(matrix, sampling_rate)

    def time_to_samples(self, time):
        """Convert time to samples.

        Parameters
        ----------
        time: float
            Time in seconds

        Returns
        -------
        samples: float
            Number of samples
        """
        return time * self.sampling_rate

    def samples_to_time(self, samples):
        """Convert samples to time.

        Parameters
        ----------
        samples: float
            Number of samples

        Returns
        -------
        time: float
            Time in seconds
        """
        return samples / self.sampling_rate


class NeuroFmri(NeuroImage):
    """
    Represents the necessary data for brain state analysis in fMRI.

    Parameters
    ----------
    data: :py:class:`RawMatrix`
        The raw matrix representing the data
    tr: float
        The repetition time (TR) in seconds
    """

    def __init__(self, matrix, tr):
        self.data = RawMatrix(matrix)
        self.tr = tr

    @classmethod
    def create_from_matrix(cls, matrix, tr):
        """Create a NeuroFmri object from a raw numpy matrix.

        Parameters
        ----------
        matrix: :py:class:`numpy.array`
            Raw matrix of size subjects x regions x time
        tr: float
            Repetition time of the fMRI data

        Returns
        -------
        neuroimage: :py:class:`neurostates.core.NeuroFmri`
            An instance of the class that represents the EEG data.

        """
        if tr is None:
            raise ValueError("tr cannot be None.")
        return NeuroFmri(matrix, tr)

    def time_to_samples(self, time):
        """Convert time to samples.

        Parameters
        ----------
        time: float
            Time in seconds

        Returns
        -------
        samples: float
            Number of samples
        """
        return time / self.tr

    def samples_to_time(self, samples):
        """Convert samples to time.

        Parameters
        ----------
        samples: float
            Number of samples

        Returns
        -------
        time: float
            Time in seconds
        """
        return samples * self.tr
