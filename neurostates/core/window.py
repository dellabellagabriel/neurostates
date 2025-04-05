# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

"""window has functionalities related to sliding window operations."""

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from .utils import validate_data_array


class SecondsWindower(BaseEstimator, TransformerMixin):
    """Perform the sliding window operation from the data's timeseries\
    using seconds as the unit.

    Parameters
    ----------
    length : float
        The length of the window in seconds.
    step : float
        The step of the window in seconds.
    sample_rate : float
        The sample rate of the timeseries in Hz.
    tapering_function: callable, default=None
        The tapering function to use for each window.
    """

    def __init__(self, length, step, sample_rate, tapering_function=None):
        self.length = length
        self.step = step
        self.tapering_function = tapering_function
        self.sample_rate = sample_rate

    def fit(self, X):  # noqa: N803
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

    def transform(self, X, y=None):  # noqa: N803
        """
        Transforms a numpy array of size n_subjects x n_rois x n_samples\
        representing the timeseries to a sliding window timeseries of\
        size n_subjects x n_windows x n_rois x n_samples.

        Parameters
        ----------
        X : ndarray
            A numpy array representing the timeseries

        Returns
        -------
            window: ndarray
            A sliding window timeseries
        """
        return window(
            X,
            length=int(self.length * self.sample_rate),
            step=int(self.step * self.sample_rate),
            tapering_function=self.tapering_function,
        )


class SamplesWindower(BaseEstimator, TransformerMixin):
    """Perform the sliding window operation from the data's timeseries\
    using samples as the unit.

    Parameters
    ----------
    length : float
        The length of the window in samples.
    step : float
        The step of the window in samples.
    tapering_function: callable, default=None
        The tapering function to use for each window.
    """

    def __init__(self, length, step, tapering_function=None):
        self.length = length
        self.step = step
        self.tapering_function = tapering_function

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
        Transforms a numpy array of size n_subjects x n_rois x n_samples\
        representing the timeseries to a sliding window timeseries of\
        size n_subjects x n_windows x n_rois x n_samples.

        Parameters
        ----------
        X : ndarray
            A numpy array representing the timeseries

        Returns
        -------
            window: ndarray
            A dictionary of sliding window timeseries
        """
        return window(
            X,
            length=self.length,
            step=self.step,
            tapering_function=self.tapering_function,
        )


class SecondsWindowerGroup(BaseEstimator, TransformerMixin):
    """Perform the sliding window operation from the data's timeseries\
    using seconds as the unit on a group basis.

    Parameters
    ----------
    length : float
        The length of the window in seconds.
    step : float
        The step of the window in seconds.
    sample_rate : float
        The sample rate of the timeseries in Hz.
    tapering_function: callable, default=None
        The tapering function to use for each window.
    """

    def __init__(self, length, step, sample_rate, tapering_function=None):
        self.length = length
        self.step = step
        self.tapering_function = tapering_function
        self.sample_rate = sample_rate

    def fit(self, dict_of_groups):  # noqa: N803
        """
        Required by the scikit-learn\
        interface.\
        No parameters are fit in this transformer.

        Parameters
        ----------
        dict_of_groups : dictionary of groups. Each key (group)
        must be an ndarray of size n_subjects x n_rois x n_samples
        y : array-like, shape (n_samples,), default=None
            Target labels (not used in this case).

        Returns
        -------
        self : object
            The fitted transformer (no changes in this case).
        """
        return self

    def transform(self, dict_of_groups, y=None):  # noqa: N803
        """
        Transforms a numpy array of size n_subjects x n_rois x n_samples\
        representing the timeseries to a sliding window timeseries of\
        size n_subjects x n_windows x n_rois x n_samples on a group basis.

        Parameters
        ----------
        dict_of_groups : ndarray
            A dictionary of groups. Each key (group)
            must be a numpy array representing the timeseries

        Returns
        -------
            window: ndarray
            A dictionary sliding window timeseries
        """
        dict_of_sliding_window = {}
        for group in dict_of_groups.keys():
            dict_of_sliding_window[group] = window(
                dict_of_groups[group],
                length=int(self.length * self.sample_rate),
                step=int(self.step * self.sample_rate),
                tapering_function=self.tapering_function,
            )

        self.dict_of_groups_ = dict_of_sliding_window
        return dict_of_sliding_window


class SamplesWindowerGroup(BaseEstimator, TransformerMixin):
    """Perform the sliding window operation from the data's timeseries\
    using samples as the unit.

    Parameters
    ----------
    length : float
        The length of the window in samples.
    step : float
        The step of the window in samples.
    tapering_function: callable, default=None
        The tapering function to use for each window.
    """

    def __init__(self, length, step, tapering_function=None):
        self.length = length
        self.step = step
        self.tapering_function = tapering_function

    def fit(self, dict_of_groups, y=None):  # noqa: N803
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

    def transform(self, dict_of_groups):  # noqa: N803
        """
        Transforms a numpy array of size n_subjects x n_rois x n_samples\
        representing the timeseries to a sliding window timeseries of\
        size n_subjects x n_windows x n_rois x n_samples.

        Parameters
        ----------
        X : ndarray
            A numpy array representing the timeseries

        Returns
        -------
            window: ndarray
            A dictionary of sliding window timeseries
        """
        dict_of_sliding_window = {}
        for group in dict_of_groups.keys():
            dict_of_sliding_window[group] = window(
                dict_of_groups[group],
                length=self.length,
                step=self.step,
                tapering_function=self.tapering_function,
            )

        self.dict_of_groups_ = dict_of_sliding_window
        return dict_of_sliding_window


def window(data_array_raw, length, step, tapering_function=None):
    """Represents a sliding window operation.

    Parameters
    ----------
    data_array: numpy array
        The neuroimage data.
        The shape should be subjects x regions x samples
    length: int
        The size of the window in samples.
    step: int
        The step size of the window in samples.
    tapering_function: callable
        The function that will be used to taper the window.

    Returns
    -------
    windowed_data: ndarray
    An array of size subjects x regions x windows x samples that\
    holds the data after the sliding window procedure.
    """
    data_array = validate_data_array(data_array_raw, ndim=3)

    subjects, regions, samples = data_array.shape

    tapering_window = (
        np.ones(length)
        if tapering_function is None
        else tapering_function(length)
    )
    n_windows = int((samples - length) / step) + 1
    windowed_data = np.empty(
        (
            subjects,
            regions,
            n_windows,
            length,
        )
    )
    for i in range(n_windows):
        from_index = i * step
        to_index = from_index + length
        windowed_data[:, :, i, :] = (
            tapering_window * data_array[:, :, from_index:to_index]
        )

    return windowed_data
