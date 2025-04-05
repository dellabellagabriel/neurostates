# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

from neurostates.core.window import (
    SamplesWindower,
    SamplesWindowerGroup,
    SecondsWindower,
    SecondsWindowerGroup,
    window,
)

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


def test_seconds_windower():
    np.random.seed(42)  # ver como cambiar esto
    n_subjects = 20
    n_regions = 90
    n_samples = 150
    sample_rate = 2
    data_array = np.random.rand(n_subjects, n_regions, n_samples)

    length_in_seconds = 10
    step_in_seconds = 2.5
    length_in_samples = int(length_in_seconds * sample_rate)
    step_in_samples = int(step_in_seconds * sample_rate)
    seconds_windower = SecondsWindower(
        length=length_in_seconds,
        step=step_in_seconds,
        tapering_function=hamming,
        sample_rate=sample_rate,
    )
    subjects, regions, windows, samples = seconds_windower.transform(
        data_array
    ).shape

    n_windows_ref = int((n_samples - length_in_samples) / step_in_samples) + 1
    assert windows == n_windows_ref
    assert subjects == n_subjects
    assert regions == n_regions
    assert samples == length_in_samples


def test_samples_windower():
    np.random.seed(42)  # ver como cambiar esto
    n_subjects = 20
    n_regions = 90
    n_samples = 150
    data_array = np.random.rand(n_subjects, n_regions, n_samples)

    length_in_samples = 20
    step_in_samples = 5
    samples_windower = SamplesWindower(
        length=length_in_samples,
        step=step_in_samples,
        tapering_function=hamming,
    )
    subjects, regions, windows, samples = samples_windower.transform(
        data_array
    ).shape

    n_windows_ref = int((n_samples - length_in_samples) / step_in_samples) + 1
    assert windows == n_windows_ref
    assert subjects == n_subjects
    assert regions == n_regions
    assert samples == length_in_samples


def test_samples_windower_group():
    np.random.seed(42)  # ver como cambiar esto
    n_subjects = 20
    n_regions = 90
    n_samples = 150
    dict_of_groups = {
        "group_1": np.random.rand(n_subjects, n_regions, n_samples),
        "group_2": np.random.rand(n_subjects, n_regions, n_samples),
    }

    length_in_samples = 20
    step_in_samples = 5
    samples_windower_group = SamplesWindowerGroup(
        length=length_in_samples,
        step=step_in_samples,
        tapering_function=hamming,
    )

    transformed = samples_windower_group.transform(dict_of_groups)

    for group, data in transformed.items():
        subjects, regions, windows, samples = data.shape
        n_windows_ref = (
            int((n_samples - length_in_samples) / step_in_samples) + 1
        )
        assert windows == n_windows_ref
        assert subjects == n_subjects
        assert regions == n_regions
        assert samples == length_in_samples


def test_seconds_windower_group():
    np.random.seed(42)  # ver como cambiar esto
    n_subjects = 20
    n_regions = 90
    n_samples = 150
    sample_rate = 2
    dict_of_groups = {
        "group_1": np.random.rand(n_subjects, n_regions, n_samples),
        "group_2": np.random.rand(n_subjects, n_regions, n_samples),
    }

    length_in_seconds = 10
    step_in_seconds = 2.5
    length_in_samples = int(length_in_seconds * sample_rate)
    step_in_samples = int(step_in_seconds * sample_rate)
    seconds_windower_group = SecondsWindowerGroup(
        length=length_in_seconds,
        step=step_in_seconds,
        tapering_function=hamming,
        sample_rate=sample_rate,
    )
    transformed = seconds_windower_group.transform(dict_of_groups)

    for group, data in transformed.items():
        subjects, regions, windows, samples = data.shape
        n_windows_ref = (
            int((n_samples - length_in_samples) / step_in_samples) + 1
        )
        assert windows == n_windows_ref
        assert subjects == n_subjects
        assert regions == n_regions
        assert samples == length_in_samples
