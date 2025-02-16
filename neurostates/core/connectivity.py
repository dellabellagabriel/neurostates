# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

"""connectivity has functionalities related to functional connectivity."""

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from .utils import validate_data_array


class DynamicConnectivity(BaseEstimator, TransformerMixin):
    """Calculate the dynamic connectivity windows based on the\
    windowed neuroimage timeseries.

    Parameters
    ----------
    method : callable
        The function to use when calculating the correlation\
        between two timeseries.
    """

    def __init__(self, method=None):
        self.method = method

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
        Transforms a windowed timeseries into dynamic connectivity\
        matrices.

        Parameters
        ----------
        X : ndarray
            A numpy array of size n_subjects x n_windows x n_rois x n_samples

        Returns
        -------
        connectivity: ndarray
            A numpy array of size n_subjects x n_windows x n_rois x n_rois
        """
        return connectivity(X, self.method)


def connectivity(windowed_data_raw, method=None):
    """Represents the functional connectivity operation.\
    This usually comes from the output of window function.

    Parameters
    ----------
    windowed_data: numpy array
        The output of window function.
        The shape should be subjets x regions x window x samples.

    method: callable
        The function that will be used to compute the connectivity between
        regions. Default is None, which means the method will be the
        Pearson correlation (np.corrcoef).

    Returns
    -------
    connectivity_data: ndarray
    An array of size subjects x windows x regions x regions that holds the\
    connectivity values.
    """
    windowed_data = validate_data_array(windowed_data_raw, ndim=4)

    subjects, regions, windows, _ = windowed_data.shape
    method = np.corrcoef if method is None else method

    connectivity_data = np.empty(
        (
            subjects,
            windows,
            regions,
            regions,
        )
    )

    for subject in range(subjects):
        for window in range(windows):
            connectivity_data[subject, window, :, :] = method(
                windowed_data[subject, :, window, :]
            )

    return connectivity_data
