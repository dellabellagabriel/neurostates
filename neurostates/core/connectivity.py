# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

"""connectivity has functionalities related to functional connectivity."""

import numpy as np


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
