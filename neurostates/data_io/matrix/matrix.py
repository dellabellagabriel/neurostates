# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

from neurostates.core.neurodata import NeuroEEG, NeuroFMRI


def create_data_from_matrix(matrix, modality, sampling_rate=None, tr=None):
    if modality == "eeg":
        return NeuroEEG.create_from_matrix(matrix, sampling_rate)

    if modality == "fmri":
        return NeuroFMRI.create_from_matrix(matrix, tr)

    raise ValueError(
        f"Unsupported modality {modality}. Options are 'eeg', 'fmri'."
    )
