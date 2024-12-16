# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

"""classification uses the output from clustering and connectivity to\
classify the connectivity matrices by comparing them to the centroids."""

import numpy as np

from scipy.spatial.distance import cdist

from .utils import validate_data_array, validate_groups_dict


# TODO: agregar criterio para ordenamiento de centroides
def classification(groups_dict, centroids, metric="euclidean"):
    """Takes the centroids previously calculated with clustering()\
    and a dictionary of groups/conditions containing the dynamic\
    connectivity matrices and returns the labels corresponding\
    to the nearest centroids.

    For each matrix it calculates the distance to each centroid.\
    A label is assigned based on the centroid that minimizes this distance.

    Parameters
    ----------
        groups_dict (dict[str, ndarray]): A dictionary of conditions.
        Each condition must have a numpy array with the dynamic connectivity
        matrices from connectivity().

        centroids (ndarray): An array of n_clusters x regions x regions.
        **clustering_kwargs: Keyword arguments for clustering.

        metric (str): Metric to use as a distance between matrices and\
        centroids. Default: "euclidean"

    Returns
    -------
        Tuple(group_labels(ndarray), group_freqs(ndarray)): A tuple that
        contains two arrays, group_labels which is the label resulting
        from the classification based on centroids, and group_freqs which
        calculates the frequency of each label.
    """
    validate_groups_dict(groups_dict)
    centroids = validate_data_array(centroids, 3)

    n_centroids, regions, regions = centroids.shape
    centroids_flatten = centroids.reshape(n_centroids, regions * regions)

    group_labels = {}
    group_freqs = {}
    for group in groups_dict:
        data_group = groups_dict[group]
        subjects, windows, regions, regions = data_group.shape
        data_group_flatten = data_group.reshape(
            subjects, windows, regions * regions
        )
        subjects_labels = np.empty((subjects, windows))
        subjects_freqs = np.empty((subjects, n_centroids))
        for subject in range(subjects):
            distances = cdist(
                data_group_flatten[subject], centroids_flatten, metric=metric
            )
            labels = np.argmin(distances, axis=1)

            # We count the frequency of each centroid and sort them
            # in ascending order.
            # This guarantees that all frequencies for all subjects
            # are in the same order.
            centroids_idx, frequencies = np.unique(labels, return_counts=True)
            sort_idx = np.argsort(centroids_idx)
            subjects_freqs[subject] = frequencies[sort_idx]
            subjects_labels[subject, :] = labels

        group_labels[group] = subjects_labels
        group_freqs[group] = subjects_freqs

    return group_labels, group_freqs
