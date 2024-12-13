from neurostates.core.classification import classification

import numpy as np

import pytest


def test_classification():
    np.random.seed(42)
    n_subjects = 20
    n_regions = 90
    n_centroids = 3
    n_windows = 30

    connectivity_output = np.random.rand(
        n_subjects, n_windows, n_regions, n_regions
    )
    centroids = np.random.rand(n_centroids, n_regions, n_regions)

    groups_dict = {
        "group_a": connectivity_output,
        "group_b": connectivity_output,
    }

    classif = classification(groups_dict, centroids)

    assert len(classif) == 2
    assert classif[0]["group_a"].shape == (n_subjects, n_windows)
    assert classif[0]["group_b"].shape == (n_subjects, n_windows)
    assert classif[1]["group_a"].shape == (n_subjects, n_centroids)
    assert classif[1]["group_b"].shape == (n_subjects, n_centroids)


def test_classification_no_dict():
    np.random.seed(42)
    n_regions = 90
    n_centroids = 3

    centroids = np.random.rand(n_centroids, n_regions, n_regions)

    groups_dict = np.array([1, 2, 3])

    with pytest.raises(TypeError):
        classification(groups_dict, centroids)


def test_classification_wrong_values():
    np.random.seed(42)
    n_subjects = 20
    n_regions = 90
    n_centroids = 3
    n_windows = 30

    connectivity_output = np.random.rand(
        n_subjects, n_windows, n_regions, n_regions
    )
    centroids = np.random.rand(n_centroids, n_regions, n_regions)

    groups_dict = {
        "group_a": np.array(["a", "b", "c"]),
        "group_b": connectivity_output,
    }

    with pytest.raises(ValueError):
        classification(groups_dict, centroids)
