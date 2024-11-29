from neurostates.core.window import window

import numpy as np

import pytest

from scipy.signal.windows import hamming


def test_window_shape_tapering_none():
    np.random.seed(42)  # ver como cambiar esto
    n_subjects = 20
    n_regions = 90
    n_samples = 150
    data_array = np.random.rand(n_subjects, n_regions, n_samples)

    size = 20
    step = 5
    sliding_window = window(data_array, size, step)
    subjects, regions, windows, samples = sliding_window.shape

    n_windows_ref = int((n_samples - size) / step) + 1
    assert windows == n_windows_ref
    assert subjects == n_subjects
    assert regions == n_regions
    assert samples == size


def test_window_shape_tapering():
    np.random.seed(42)  # ver como cambiar esto
    n_subjects = 20
    n_regions = 90
    n_samples = 150
    data_array = np.random.rand(n_subjects, n_regions, n_samples)

    size = 20
    step = 5
    sliding_window = window(data_array, size, step, hamming)
    subjects, regions, windows, samples = sliding_window.shape

    n_windows_ref = int((n_samples - size) / step) + 1
    assert windows == n_windows_ref
    assert subjects == n_subjects
    assert regions == n_regions
    assert samples == size


def test_window_wrong_data():
    np.random.seed(42)
    n_subjects = 20
    n_regions = 90

    data_array = np.random.rand(n_subjects, n_regions)

    size = 20
    step = 5
    with pytest.raises(ValueError):
        window(data_array, size, step)


def test_validate_data_array_string():
    np.random.seed(42)  # ver como cambiar esto
    n_subjects = 20
    n_regions = 90
    n_samples = 150
    matrix = np.random.rand(n_subjects, n_regions, n_samples).astype(object)
    matrix[0, 0, 0] = "a"

    size = 20
    step = 5
    with pytest.raises(ValueError):
        window(matrix, size, step)


def test_validate_data_array_nan():
    np.random.seed(42)
    n_subjects = 20
    n_regions = 90
    n_samples = 150
    matrix = np.random.rand(n_subjects, n_regions, n_samples)
    matrix[0, 0, 0] = np.nan

    size = 20
    step = 5
    with pytest.raises(ValueError):
        window(matrix, size, step)
