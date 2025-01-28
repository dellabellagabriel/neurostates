from neurostates.core.window import window, SamplesWindower, SecondsWindower

import numpy as np

import pytest

from scipy.signal.windows import hamming


def test_window_shape_tapering_none():
    np.random.seed(42)  # ver como cambiar esto
    n_subjects = 20
    n_regions = 90
    n_samples = 150
    data_array = np.random.rand(n_subjects, n_regions, n_samples)

    length = 20
    step = 5
    sliding_window = window(data_array, length, step)
    subjects, regions, windows, samples = sliding_window.shape

    n_windows_ref = int((n_samples - length) / step) + 1
    assert windows == n_windows_ref
    assert subjects == n_subjects
    assert regions == n_regions
    assert samples == length


def test_window_shape_tapering():
    np.random.seed(42)  # ver como cambiar esto
    n_subjects = 20
    n_regions = 90
    n_samples = 150
    data_array = np.random.rand(n_subjects, n_regions, n_samples)

    length = 20
    step = 5
    sliding_window = window(data_array, length, step, hamming)
    subjects, regions, windows, samples = sliding_window.shape

    n_windows_ref = int((n_samples - length) / step) + 1
    assert windows == n_windows_ref
    assert subjects == n_subjects
    assert regions == n_regions
    assert samples == length


def test_window_wrong_data():
    np.random.seed(42)
    n_subjects = 20
    n_regions = 90

    data_array = np.random.rand(n_subjects, n_regions)

    length = 20
    step = 5
    with pytest.raises(ValueError):
        window(data_array, length, step)


def test_validate_data_array_string():
    np.random.seed(42)  # ver como cambiar esto
    n_subjects = 20
    n_regions = 90
    n_samples = 150
    matrix = np.random.rand(n_subjects, n_regions, n_samples).astype(object)
    matrix[0, 0, 0] = "a"

    length = 20
    step = 5
    with pytest.raises(ValueError):
        window(matrix, length, step)


def test_validate_data_array_nan():
    np.random.seed(42)
    n_subjects = 20
    n_regions = 90
    n_samples = 150
    matrix = np.random.rand(n_subjects, n_regions, n_samples)
    matrix[0, 0, 0] = np.nan

    length = 20
    step = 5
    with pytest.raises(ValueError):
        window(matrix, length, step)


# def test_seconds_windower():
#     np.random.seed(42)  # ver como cambiar esto
#     n_subjects = 20
#     n_regions = 90
#     n_samples = 150
#     data_array = np.random.rand(n_subjects, n_regions, n_samples)

#     length = 20
#     step = 5
#     seconds_windower = SecondsWindower(length=length, )
#     subjects, regions, windows, samples = sliding_window.shape

#     n_windows_ref = int((n_samples - length) / step) + 1
#     assert windows == n_windows_ref
#     assert subjects == n_subjects
#     assert regions == n_regions
#     assert samples == length