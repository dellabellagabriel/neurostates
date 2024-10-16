from neurostates.data_io.matrix.matrix import create_data_from_matrix

import numpy as np

import pytest


def test_create_data_from_matrix():
    n_subjects = 20
    n_regions = 90
    n_time = 150
    matrix = np.random.rand(n_subjects, n_regions, n_time)

    neurodata = create_data_from_matrix(matrix=matrix, modality="fmri", tr=2.0)

    assert neurodata.subjects == n_subjects
    assert neurodata.regions == n_regions
    assert neurodata.time == n_time


def test_create_data_from_matrix_wrong_dimensions():
    n_regions = 90
    n_time = 150
    matrix = np.random.rand(n_regions, n_time)

    with pytest.raises(ValueError):
        create_data_from_matrix(matrix=matrix, modality="fmri", tr=2.0)


def test_create_from_data_wrong_modality():
    n_subjects = 20
    n_regions = 90
    n_time = 150
    matrix = np.random.rand(n_subjects, n_regions, n_time)

    with pytest.raises(ValueError):
        create_data_from_matrix(matrix=matrix, modality="something", tr=2.0)


def test_create_from_data_missing_sr_or_tr():
    n_subjects = 20
    n_regions = 90
    n_time = 150
    matrix = np.random.rand(n_subjects, n_regions, n_time)

    with pytest.raises(ValueError):
        create_data_from_matrix(matrix=matrix, modality="fmri")
