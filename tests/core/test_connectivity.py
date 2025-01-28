from neurostates.core.connectivity import connectivity, DynamicConnectivity
from neurostates.core.window import window, SamplesWindower

import numpy as np


import pytest

from scipy.signal.windows import hamming

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


def test_dynamic_connectivity():
        np.random.seed(42)
        n_subjects = 20
        n_regions = 90
        n_samples = 150
        matrix = np.random.rand(n_subjects, n_regions, n_samples)

        length_in_samples = 20
        step_in_samples = 5
        sliding_windower = SamplesWindower(length=length_in_samples, step=step_in_samples, tapering_function=hamming)
        sliging_window = sliding_windower.transform(matrix)

        dynamic_connectivity = DynamicConnectivity()
        connectivity_output = dynamic_connectivity.transform(sliging_window).shape

        assert connectivity_output == (20, 27, 90, 90)