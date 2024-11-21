# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

"""window has functionalities related to sliding window operations."""

import numpy as np

def window(data_array, length, step, tapering_function=None):
    """ Represents a sliding window operation. 
        
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
    """
    if data_array.ndim != 3:
        raise ValueError("Argument data_array must be a 3-dimensional array.")
    
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
        windowed_data[:,:,i,:] = (
            tapering_window
            * data_array[:, :, from_index:to_index]
        )

    return windowed_data 

class SlidingWindow:
    """Represents a sliding window operation.

    It is applied to objects of type NeuroImage via the `apply_to()` method.

    Parameters
    ----------
    size: int
        The size of the window in samples.
    step: int
        The step size of the window in samples.
    tapering_function: callable
        The function that will be used to taper the window.
    """

    def __init__(self, size, step, tapering_function=None):
        self.size = size
        self.step = step
        self.tapering_function = tapering_function

    def apply_to(self, neurodata):
        """Apply the sliding window operation to a NeuroImage object."""
        tapering_window = (
            np.ones(self.size)
            if self.tapering_function is None
            else self.tapering_function(self.size)
        )
        n_windows = int((neurodata.data.samples - self.size) / self.step) + 1
        windowed_data = np.empty(
            (
                n_windows,
                neurodata.data.subjects,
                neurodata.data.regions,
                self.size,
            )
        )
        for i in range(n_windows):
            from_index = i * self.step
            to_index = from_index + self.size
            windowed_data[i] = (
                tapering_window
                * neurodata.data.data[:, :, from_index:to_index]
            )

        self.windowed_data_ = windowed_data

    @property
    def windows(self):
        """Number of windows."""
        return self.windowed_data_.shape[0]

    @property
    def subjects(self):
        """Number of subjects."""
        return self.windowed_data_.shape[1]

    @property
    def regions(self):
        """Number of regions."""
        return self.windowed_data_.shape[2]

    @property
    def samples(self):
        """Number of samples."""
        return self.windowed_data_.shape[3]
