# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

"""
This module defines the NeuroData class which contains all the required
data to perform a brain state analysis.
"""


class NeuroData:
    """
    Represents the data necessary for the brain state analysis.

    Parameters
    ----------
    data: :py:class:`numpy.ndarray`
        A matrix of size subjects x regions x time
    modality: str
        It can take either "eeg" or "fmri" options.
    sampling_rate: float
        The sampling rate of the signal expressed in Hz.
    tr: float
        The TR of the signal
    """

    def __init__(self, data, modality, sampling_rate=None, tr=None):
        self.data = data
        self.modality = modality
        self.sampling_rate = sampling_rate
        self.tr = tr

    @property
    def subjects(self):
        return self.data.shape[0]

    @property
    def regions(self):
        return self.data.shape[1]

    @property
    def time(self):
        return self.data.shape[2]

    @classmethod
    def from_matrix_data(
        cls,
        data,
        modality,
        sampling_rate=None,
        tr=None,
    ):
        if data.ndim != 3:
            raise ValueError(
                f"""The data matrix must be three-dimensional
                (subjects x regions x time). Expected 3 but got {data.ndim}."""
            )

        modality = modality.lower()
        if modality != "eeg" and modality != "fmri":
            raise ValueError(
                f"""The modality must be either 'eeg' or 'fmri',
                but got '{modality}'."""
            )

        if sampling_rate is None and tr is None:
            raise ValueError(
                """A 'sampling_rate' or 'tr' parameter is required.
                None were given."""
            )

        return cls(
            data=data, modality=modality, sampling_rate=sampling_rate, tr=tr
        )
