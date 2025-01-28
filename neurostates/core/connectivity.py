# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

"""connectivity has functionalities related to functional connectivity."""

import numpy as np


from .utils import validate_data_array
from sklearn.base import BaseEstimator, TransformerMixin

class DynamicConnectivity(BaseEstimator, TransformerMixin):
    def __init__(self, method=None):
        self.method = method
    
    def transform(self, X):
        return connectivity(X,self.method)

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
