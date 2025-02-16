# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

"""clustering performs a clustering algortithm over the data processed by \
connectivity."""

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from .utils import validate_groups_dict


class Concatenator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
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
