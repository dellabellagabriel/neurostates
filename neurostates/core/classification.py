# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

"""classification uses the output from clustering and connectivity to \
classify the connectivity matrices by comparing them to the centroids."""

import numpy as np

from scipy.spatial.distance import cdist

from sklearn.base import BaseEstimator, TransformerMixin

from .utils import (
    compute_frequencies,
    validate_data_array,
    validate_groups_dict,
)


class Frequencies(BaseEstimator, TransformerMixin):
    """Perform the calculation of frequencies based on centroids from\
    clustering and a dictionary of dynamic connectivity matrices.

    Parameters
    ----------
    centroids : ndarray
        An array of centroids from the clustering step of size\
        n_clusters x n_rois x n_rois.
    metric : str or callable, default='euclidean'
        The distance metric to use. It can be a str or callable\
        per scipy's cdit() documentation.
    """

    def __init__(self, centroids, metric="euclidean"):
        n_clusters, n_rois2 = centroids.shape
        self.centroids = centroids.reshape(
            n_clusters, int(np.sqrt(n_rois2)), int(np.sqrt(n_rois2))
        )
        self.metric = metric

    def fit(self, X, y=None):  # noqa: N803
        """
        Required by the scikit-learn\
        interface.\
        No parameters are fit in this transformer.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data to fit the transformer.
        y : array-like, shape (n_samples,), default=None
            Target labels (not used in this case).

        Returns
        -------
        self : object
            The fitted transformer (no changes in this case).
        """
        return self

    def transform(self, X):  # noqa: N803
        """
        Transforms a dictionary of dynamic connectivity matrices to a\
        dictionary of frequencies.\
        These frequencies represent how often the matrices are classified\
        as each centroid.

        Parameters
        ----------
        X : dict[str, ndarray]
            A dictionary of dynamic connectivity matrices per group

        Returns
        -------
            freqs: dict[str, ndarray]
            A dictionary of frequencies for the centroids for each group
        """
        labels, freqs = classification(X, self.centroids, self.metric)
        self.labels_ = labels
        return freqs


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
            subjects_freqs[subject] = compute_frequencies(labels, n_centroids)
            subjects_labels[subject, :] = labels

        group_labels[group] = subjects_labels
        group_freqs[group] = subjects_freqs

    return group_labels, group_freqs
