import pickle
from pathlib import Path

from neurostates.core.classification import Frequencies, classification
from neurostates.core.clustering import Concatenator
from neurostates.core.connectivity import DynamicConnectivity
from neurostates.core.window import SamplesWindower

import numpy as np

import pytest

import scipy.io as sio

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline


def test_classification_shape():
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


def test_classification():
    path_to_tests = Path("tests/core")
    dataset_controls = sio.loadmat(
        path_to_tests / "dataset" / "controls_singleprec.mat"
    )["ts"]
    dataset_patients = sio.loadmat(
        path_to_tests / "dataset" / "patients_singleprec.mat"
    )["ts"]

    with open(path_to_tests / "classification" / "freqs.pkl", "rb") as f:
        ground_truth_freqs = pickle.load(f)

    connectivity_pipeline = Pipeline(
        [
            ("windower", SamplesWindower(length=20, step=5)),
            ("connectivity", DynamicConnectivity(method=np.corrcoef)),
        ]
    )

    dynamic_connectivity_controls = connectivity_pipeline.fit_transform(
        dataset_controls
    )
    dynamic_connectivity_patients = connectivity_pipeline.fit_transform(
        dataset_patients
    )

    diccionario_de_grupos = {
        "controls": dynamic_connectivity_controls,
        "patients": dynamic_connectivity_patients,
    }

    clustering_pipeline = Pipeline(
        [
            ("preclustering", Concatenator()),
            ("clustering", KMeans(n_clusters=3, random_state=42)),
        ]
    )

    kmeans = clustering_pipeline.fit(diccionario_de_grupos)
    centroids = kmeans["clustering"].cluster_centers_

    frequencies = Frequencies(centroids=centroids)
    freqs = frequencies.fit_transform(diccionario_de_grupos)

    assert freqs.keys() == ground_truth_freqs.keys()
    np.testing.assert_allclose(
        list(freqs.values()), list(ground_truth_freqs.values()), atol=1e-5
    )
