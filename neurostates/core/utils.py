# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

"""Functions useful for different common operations."""

import numpy as np


def validate_data_array(data_array):
    """Validates that the input is numeric and does not contain NaN."""
    data_array = data_array.astype(np.float32)
    if np.any(np.isnan(data_array)):
        raise ValueError("The input data cannot contain NaN.")

    return data_array