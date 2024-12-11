# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

"""Functions useful for different common operations."""

import numpy as np


def validate_data_array(data_array, ndim):
    """Validates that the input is numeric and does not contain NaN."""
    data_array = data_array.astype(np.float32)
    if data_array.ndim != ndim:
        raise ValueError(
            f"Argument data_array must be a {ndim}-dimensional array."
        )
    if np.any(np.isnan(data_array)):
        raise ValueError("The input data cannot contain NaN.")

    return data_array


def validate_groups_dict(groups_dict):
    """Validates that the groups is a dictionary and the elements \
    of the dictionary are valid data arrays."""
    if not isinstance(groups_dict, dict):
        raise TypeError("groups_dict must be a dictionary")
    for group in groups_dict:
        validate_data_array(groups_dict[group], ndim=4)
