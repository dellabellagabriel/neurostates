import numpy as np


class SlidingWindow:
    """
    Represents an sliding window operation. It is applied to objects of type
    NeuroImage via the `apply_to()` method.

    Parameters
    ----------
    size: int
        The size of the window in samples
    step: int
        The step size of the window in samples
    tapering_function: callable
        The function that will be used to taper the window
    """

    def __init__(self, size, step, tapering_function=None):
        self.size = size
        self.step = step
        self.tapering_function = tapering_function

    def apply_to(self, neurodata):
        """
        Applies the sliding window operation to a NeuroImage object
        """

        if self.tapering_function is not None:
            tapering_window = self.tapering_function(self.size)
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
            if self.tapering_window is None:
                windowed_data[i] = neurodata.data.data[
                    :, :, from_index:to_index
                ]
            else:
                windowed_data[i] = (
                    tapering_window
                    * neurodata.data.data[:, :, from_index:to_index]
                )

        self.windowed_data_ = windowed_data

    @property
    def windows(self):
        return self.windowed_data_.shape[0]

    @property
    def subjects(self):
        return self.windowed_data_.shape[1]

    @property
    def regions(self):
        return self.windowed_data_.shape[2]

    @property
    def samples(self):
        return self.windowed_data_.shape[3]
