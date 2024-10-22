# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

"""io deals with creating NeuroImage objects from raw objects."""

# flake8: noqa A005
from .core import NeuroEEG, NeuroFmri


def create_data_from_matrix(matrix, modality, sampling_rate=None, tr=None):
    """Create a NeuroEEG or NeuroFMRI object from a raw numpy matrix.

    Parameters
    ----------
    matrix: :py:class:`numpy.array`
        Raw matrix of size subjects x regions x time
    modality: str
        Type of neuroimage. It can be 'eeg' or 'fmri'
    sampling_rate: float
        Sampling rate of the EEG data
    tr: float
        Repetition time of the fMRI data

    Returns
    -------
    neuroimage: :py:class:`neurostates.core.NeuroEEG` | :py:class:`neurostates.core.NeuroFMRI`
        An instance of the class that represents the neuroimage.


    """
    if modality == "eeg":
        return NeuroEEG.create_from_matrix(matrix, sampling_rate)

    if modality == "fmri":
        return NeuroFmri.create_from_matrix(matrix, tr)

    raise ValueError(
        f"Unsupported modality {modality}. Options are 'eeg', 'fmri'."
    )
