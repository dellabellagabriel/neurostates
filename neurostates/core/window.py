# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

"""window has functionalities related to sliding window operations."""

import numpy as np


from .utils import validate_data_array


def window(data_array_raw, length, step, tapering_function=None):
    """Represents a sliding window operation.

    Parameters
    ----------
    data_array: numpy array
        The neuroimage data.
        The shape should be subjects x regions x samples
    length: int
        The size of the window in samples.
    step: int
        The step size of the window in samples.
    tapering_function: callable
        The function that will be used to taper the window.
    """

    data_array = validate_data_array(data_array_raw, ndim=3)

    subjects, regions, samples = data_array.shape

    tapering_window = (
        np.ones(length)
        if tapering_function is None
        else tapering_function(length)
    )
    n_windows = int((samples - length) / step) + 1
    windowed_data = np.empty(
        (
            subjects,
            regions,
            n_windows,
            length,
        )
    )
    for i in range(n_windows):
        from_index = i * step
        to_index = from_index + length
        windowed_data[:, :, i, :] = (
            tapering_window * data_array[:, :, from_index:to_index]
        )

    return windowed_data
