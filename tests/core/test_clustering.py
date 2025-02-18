# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

from pathlib import Path

from neurostates.core.clustering import Concatenator
from neurostates.core.connectivity import DynamicConnectivity
from neurostates.core.window import SamplesWindower

import numpy as np

import pytest

import scipy.io as sio
from scipy.signal.windows import hamming

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline


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
        "group_b": connectivity_output,
    }

    concatenated = Concatenator()
    concatenated_shape = concatenated.transform(groups_dict).shape
    assert concatenated_shape == (1080, 8100)


def test_concatenator_no_dict():
    groups_dict = np.array([1, 3, 4, 5])
    with pytest.raises(TypeError):
        concatenated = Concatenator()
        concatenated.transform(groups_dict)


def test_concatenator_wrong_values():
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
        "group_a": np.array(["a", "b", "c"]),
        "group_b": connectivity_output,
    }
    with pytest.raises(ValueError):
        concatenated = Concatenator()
        concatenated.transform(groups_dict)


def test_clustering():
    path_to_tests = Path("tests/core")
    dataset_controls = sio.loadmat(
        path_to_tests / "dataset" / "controls_singleprec.mat"
    )["ts"]
    dataset_patients = sio.loadmat(
        path_to_tests / "dataset" / "patients_singleprec.mat"
    )["ts"]

    ground_truth_centroid_1 = sio.loadmat(
        path_to_tests / "clustering" / "centroid1.mat"
    )["centroid1"]
    ground_truth_centroid_2 = sio.loadmat(
        path_to_tests / "clustering" / "centroid2.mat"
    )["centroid2"]
    ground_truth_centroid_3 = sio.loadmat(
        path_to_tests / "clustering" / "centroid3.mat"
    )["centroid3"]

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
    centroids = kmeans["clustering"].cluster_centers_.reshape(3, 90, 90)

    np.testing.assert_allclose(
        ground_truth_centroid_1, centroids[0, :, :], atol=1e-5
    )
    np.testing.assert_allclose(
        ground_truth_centroid_2, centroids[1, :, :], atol=1e-5
    )
    np.testing.assert_allclose(
        ground_truth_centroid_3, centroids[2, :, :], atol=1e-5
    )
