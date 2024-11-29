
from neurostates.core.window import window
from neurostates.core.connectivity import connectivity
from neurostates.core.clustering import clustering

import numpy as np


def test_clustering():
    np.random.seed(42)
    n_subjects = 20
    n_regions = 90
    n_samples = 150
    matrix = np.random.rand(n_subjects, n_regions, n_samples)

    size = 20
    step = 5
    sliding_window = window(matrix, size, step)

    connectivity_output = connectivity(sliding_window)
    
    groups_dict = {
        "group_a": connectivity_output,
        "group_b": connectivity_output
    }
    clustering(groups_dict, n_clusters=3)