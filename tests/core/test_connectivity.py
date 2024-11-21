from neurostates.core.connectivity import connectivity
from neurostates.core.window import window

import numpy as np


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