# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

from neurostates.data_io.matrix.matrix import create_data_from_matrix

import numpy as np

import pytest


def test_create_fmri_data_from_matrix():
    n_subjects = 20
    n_regions = 90
    n_samples = 150
    matrix = np.random.rand(n_subjects, n_regions, n_samples)

    neurodata = create_data_from_matrix(matrix=matrix, modality="fmri", tr=2.0)

    assert neurodata.data.subjects == n_subjects
    assert neurodata.data.regions == n_regions
    assert neurodata.data.samples == n_samples


def test_create_eeg_data_from_matrix():
    n_subjects = 20
    n_regions = 128
    n_samples = 10240
    matrix = np.random.rand(n_subjects, n_regions, n_samples)

    neurodata = create_data_from_matrix(
        matrix=matrix, modality="eeg", sampling_rate=2.0
    )

    assert neurodata.data.subjects == n_subjects
    assert neurodata.data.regions == n_regions
    assert neurodata.data.samples == n_samples


def test_create_fmri_data_from_matrix_wrong_dimensions():
    n_regions = 90
    n_time = 150
    matrix = np.random.rand(n_regions, n_time)

    with pytest.raises(ValueError):
        create_data_from_matrix(matrix=matrix, modality="fmri", tr=2.0)


def test_create_eeg_data_from_matrix_wrong_dimensions():
    n_regions = 90
    n_time = 150
    matrix = np.random.rand(n_regions, n_time)

    with pytest.raises(ValueError):
        create_data_from_matrix(
            matrix=matrix, modality="eeg", sampling_rate=2.0
        )


def test_create_from_data_wrong_modality():
    n_subjects = 20
    n_regions = 90
    n_time = 150
    matrix = np.random.rand(n_subjects, n_regions, n_time)

    with pytest.raises(ValueError):
        create_data_from_matrix(matrix=matrix, modality="something", tr=2.0)


def test_create_fmri_data_from_matrix_missing_tr():
    n_subjects = 20
    n_regions = 90
    n_time = 150
    matrix = np.random.rand(n_subjects, n_regions, n_time)

    with pytest.raises(ValueError):
        create_data_from_matrix(matrix=matrix, modality="fmri")


def test_create_eeg_data_from_matrix_missing_sampling_rate():
    n_subjects = 20
    n_regions = 90
    n_time = 150
    matrix = np.random.rand(n_subjects, n_regions, n_time)

    with pytest.raises(ValueError):
        create_data_from_matrix(matrix=matrix, modality="eeg")
