from neurostates.core.clustering import Concatenator, clustering
from neurostates.core.connectivity import DynamicConnectivity, connectivity
from neurostates.core.window import SamplesWindower, window

import numpy as np

import pytest

from scipy.signal.windows import hamming


def test_concatenator():
    np.random.seed(42)
    n_subjects = 20
    n_regions = 90
    n_samples = 150
    matrix = np.random.rand(n_subjects, n_regions, n_samples)
    
    length_in_samples = 20
    step_in_samples = 5
    
    sliding_windower = SamplesWindower(
        length=length_in_samples,
        step=step_in_samples,
        tapering_function=hamming,
    )
    
    sliding_window = sliding_windower.transform(matrix)
    dynamic_connectivity = DynamicConnectivity()
    connectivity_output = dynamic_connectivity.transform(sliding_window)
    
    groups_dict = {
        "group_a": connectivity_output,
        "group_b": connectivity_output
    }
    
    concatenated = Concatenator()
    concatenated_shape = concatenated.transform(groups_dict).shape
    assert concatenated_shape == (1080,8100)

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
        "group_b": connectivity_output,
    }
    centroids = clustering(groups_dict, n_clusters=3)
    assert centroids.shape == (3, n_regions, n_regions)


def test_clustering_no_dict():

    groups_dict = np.array([1, 3, 4, 5])
    with pytest.raises(TypeError):
        clustering(groups_dict, n_clusters=3)


def test_clustering_wrong_values():
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
        "group_a": np.array(["a", "b", "c"]),
        "group_b": connectivity_output,
    }
    with pytest.raises(ValueError):
        clustering(groups_dict, n_clusters=3)
