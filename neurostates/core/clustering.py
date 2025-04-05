# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

"""clustering performs a clustering algortithm over the data processed by \
connectivity."""

import numpy as np

from scipy.stats import entropy

from sklearn.base import BaseEstimator, TransformerMixin

from .utils import validate_groups_dict


class Concatenator(BaseEstimator, TransformerMixin):
    """Concatenate the dynamic connectivity matrices by subjects\
    and windows for the clustering step."""

    def __init__(self):
        pass

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
        Transforms a numpy array of size\
        n_subjects x n_windows x n_rois x n_rois to\
        n_subjects*n_windows x n_rois*n_rois.

        Parameters
        ----------
        X : ndarray
            A dynamic connectivity matrix of size\
            n_subjects x n_windows x n_rois x n_rois

        Returns
        -------
        data_concatenate : ndarray
            A concatenated matrix of size\
            n_subjects*n_windows x n_rois*n_rois
        """
        validate_groups_dict(X)

        data_concatenate = []
        for group in X:
            subjects, windows, rois, _ = X[group].shape
            data = (
                X[group]
                .reshape(subjects, windows, -1)
                .reshape(subjects * windows, -1)
            )
            data_concatenate.append(data)

        data_concatenate = np.concatenate(data_concatenate, axis=0)

        return data_concatenate


def sort_centroids_by_entropy(cluster_centers_):
    """Sort centroids by decreasing entropy.

    Parameters
    ----------
        `cluster_centers_` (np.ndarray): A k x N matrix
        with k the number of centroids and N the features.

    Returns
    -------
        tuple(int, ndarray): the order of the centroids
        and their entropies.
    """
    num_centroids, _ = cluster_centers_.shape
    entropies = np.zeros(num_centroids)
    for centroid in range(num_centroids):
        cluster_data = cluster_centers_[centroid, :]

        num_bins = int(np.sqrt(len(cluster_data)))

        hist, _ = np.histogram(cluster_data, bins=num_bins, density=True)
        hist = hist / hist.sum()

        shannon_entropy = entropy(hist, base=2)
        entropies[centroid] = shannon_entropy

    indices = np.argsort(entropies)[::-1]
    return indices, entropies[indices]
