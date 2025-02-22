# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

from pathlib import Path

from neurostates.core.connectivity import DynamicConnectivity, connectivity
from neurostates.core.window import (
    SamplesWindower,
    SecondsWindower,
    window,
)

import numpy as np

import pytest

import scipy.io as sio
from scipy.signal.windows import hamming

from sklearn.pipeline import Pipeline


def test_connectivity_shape():
    np.random.seed(42)
    n_subjects = 20
    n_regions = 90
    n_samples = 150
    matrix = np.random.rand(n_subjects, n_regions, n_samples)

    size = 20
    step = 5
    sliding_window = window(matrix, size, step)

    connectivity_output = connectivity(sliding_window)

    assert connectivity_output.shape == (20, 27, 90, 90)


def test_connectivity_method():
    pass


def test_validate_data_array_string():
    np.random.seed(42)
    n_subjects = 20
    n_regions = 90
    n_samples = 15
    n_window = 10
    matrix = np.random.rand(n_subjects, n_regions, n_window, n_samples).astype(
        object
    )
    matrix[0, 0, 0] = "a"

    with pytest.raises(ValueError):
        connectivity(matrix)


def test_validate_data_array_nan():
    np.random.seed(42)
    n_subjects = 20
    n_regions = 90
    n_samples = 15
    n_window = 10
    matrix = np.random.rand(n_subjects, n_regions, n_window, n_samples)
    matrix[0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        connectivity(matrix)


def test_dynamic_connectivity_shape():
    np.random.seed(42)
    n_subjects = 20
    n_regions = 90
    n_samples = 150
    matrix = np.random.rand(n_subjects, n_regions, n_samples)

    length_in_samples = 20
    step_in_samples = 5
    sliding_windower = SamplesWindower(
        length=length_in_samples,
        step=step_in_samples,
        tapering_function=hamming,
    )
    sliging_window = sliding_windower.transform(matrix)

    dynamic_connectivity = DynamicConnectivity()
    connectivity_output = dynamic_connectivity.transform(sliging_window).shape

    assert connectivity_output == (20, 27, 90, 90)


def test_dynamic_connectivity_seconds_windower():
    path_to_tests = Path("tests/core")
    dataset_controls = sio.loadmat(
        path_to_tests / "dataset" / "controls_singleprec.mat"
    )["ts"]
    ground_truth_dynamic_connectivity = sio.loadmat(
        path_to_tests / "connectivity" / "dynamic_connectivity_controls.mat"
    )["dynamic_connectivity"]

    connectivity_pipeline = Pipeline(
        [
            ("windower", SecondsWindower(length=20, step=5, sample_rate=1)),
            ("connectivity", DynamicConnectivity(method=np.corrcoef)),
        ]
    )

    dynamic_connectivity = connectivity_pipeline.fit_transform(
        dataset_controls
    )
    # take the first 3 windows to reduce space in testing
    dynamic_connectivity = dynamic_connectivity[:, 0:3, :, :]

    np.testing.assert_allclose(
        ground_truth_dynamic_connectivity, dynamic_connectivity, atol=1e-5
    )


def test_dynamic_connectivity_samples_windower():
    path_to_tests = Path("tests/core")
    dataset_controls = sio.loadmat(
        path_to_tests / "dataset" / "controls_singleprec.mat"
    )["ts"]
    ground_truth_dynamic_connectivity = sio.loadmat(
        path_to_tests / "connectivity" / "dynamic_connectivity_controls.mat"
    )["dynamic_connectivity"]

    connectivity_pipeline = Pipeline(
        [
            ("windower", SamplesWindower(length=20, step=5)),
            ("connectivity", DynamicConnectivity(method=np.corrcoef)),
        ]
    )

    dynamic_connectivity = connectivity_pipeline.fit_transform(
        dataset_controls
    )
    # take the first 3 windows to reduce space in testing
    dynamic_connectivity = dynamic_connectivity[:, 0:3, :, :]

    np.testing.assert_allclose(
        ground_truth_dynamic_connectivity, dynamic_connectivity, atol=1e-5
    )
