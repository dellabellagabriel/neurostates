# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

"""connectivity has functionalities related to functional connectivity."""

import numpy as np
from .utils import validate_data_array

def connectivity(windowed_data_raw, method=None):
    """
    Represents the functional connectivity operation.
    This usually comes from the output of window function.

    Parameters
    ----------
    windowed_data: numpy array
        The output of window function. 
        The shape should be subjets x regions x window x samples

    method: callable
        The function that will be used to compute the connectivity between
        regions. Default is None, which means the method will be the 
        Pearson correlation (np.corrcoef)
    """
    
    windowed_data = validate_data_array(windowed_data_raw)
    
    subjects, regions, windows, samples = windowed_data.shape
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
    

class Connectivity:
    """
    Represents the functional connectivity operation. It is applied to objects\
    of type SlidingWindow.

    Parameters
    ----------
    method: callable
        The function that will be used to compute the connectivity between
        regions
    """

    def __init__(self, method):
        self.method = method

    def apply_to(self, slidingwindow):
        """Apply the connectivity operation to a SlidingWindow object."""
        connectivity_data = np.empty(
            (
                slidingwindow.windows,
                slidingwindow.subjects,
                slidingwindow.regions,
                slidingwindow.regions,
            )
        )
        for window in range(slidingwindow.windows):
            for subject in range(slidingwindow.subjects):
                connectivity_data[window, subject, :, :] = self.method(
                    slidingwindow.windowed_data_[window, subject, :, :]
                )

        self.connectivity_data_ = connectivity_data
