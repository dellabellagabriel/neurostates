# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

"""clustering performs a clustering algortithm over the data processed by \
connectivity."""

import numpy as np

from sklearn.cluster import KMeans

from sklearn.base import BaseEstimator, TransformerMixin

from .utils import validate_groups_dict

class Concatenator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X):
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

        data_concatenate = np.array(data_concatenate)
        data_concatenate = data_concatenate.reshape(-1, rois**2)

        return data_concatenate

def clustering(groups_dict, n_clusters, **clustering_kwargs):
    """Perform k-means clustering on dynamic connectivity matrices.

    This function uses the groups found in `groups_dict` as the\
    groups/conditions of the experiment and returns an\
    n_clusters x rois x rois matrix.

    Parameters
    ----------
        groups_dict (dict[str, ndarray]): A dictionary of conditions.
        Each condition must have a numpy array with the dynamic connectivity
        matrices from connectivity().

        n_clusters (int): The number of clusters.
        **clustering_kwargs: Keyword arguments for clustering.

    Returns
    -------
        centroids: ndarray
        An array of size n_centroids x regions x regions that results from\
        the clustering algorithm.
    """
    validate_groups_dict(groups_dict)

    data_concatenate = []
    for group in groups_dict:
        subjects, windows, rois, _ = groups_dict[group].shape
        data = (
            groups_dict[group]
            .reshape(subjects, windows, -1)
            .reshape(subjects * windows, -1)
        )
        data_concatenate.append(data)

    data_concatenate = np.array(data_concatenate)
    data_concatenate = data_concatenate.reshape(-1, rois**2)
    kmeans = KMeans(n_clusters=n_clusters, **clustering_kwargs)
    kmeans.fit(data_concatenate)

    return kmeans.cluster_centers_.reshape(-1, rois, rois)
