import numpy as np


class Connectivity:
    """
    Represents the functional connectivity operation. It is applied to objects
    of type SlidingWindow

    Parameters
    ----------
    method: callable
        The function that will be used to compute the connectivity between
        regions
    """

    def __init__(self, method):
        self.method = method

    def apply_to(self, window_obj):
        """
        Applies the connectivity operation to a SlidingWindow object
        """
        connectivity_data = np.empty(
            (
                window_obj.windows,
                window_obj.subjects,
                window_obj.regions,
                window_obj.regions,
            )
        )
        for window in range(window_obj.windows):
            for subject in range(window_obj.subjects):
                connectivity_data[window, subject, :, :] = self.method(
                    window_obj.windowed_data_[window, subject, :, :]
                )

        self.connectivity_data_ = connectivity_data
