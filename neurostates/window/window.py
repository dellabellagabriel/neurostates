import numpy as np


class Window:
    def __init__(self, size, step, tapering_function=None):
        self.size = size
        self.step = step
        self.tapering_function = tapering_function

    def apply_to(self, neurodata):
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
