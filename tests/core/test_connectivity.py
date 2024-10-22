from neurostates.core.connectivity import Connectivity
from neurostates.core.io import create_data_from_matrix
from neurostates.core.window import SlidingWindow

import numpy as np

np.random.seed(42)


def test_connectivity_using_pearson():
    n_subjects = 20
    n_regions = 90
    n_samples = 150
    matrix = np.random.rand(n_subjects, n_regions, n_samples)
    neurodata = create_data_from_matrix(matrix=matrix, modality="fmri", tr=2.0)

    size = 20
    step = 5
    sliding_window = SlidingWindow(size, step)
    sliding_window.apply_to(neurodata)

    connectivity = Connectivity(method=np.corrcoef)
    connectivity.apply_to(sliding_window)

    assert connectivity.connectivity_data_.shape == (27, 20, 90, 90)
