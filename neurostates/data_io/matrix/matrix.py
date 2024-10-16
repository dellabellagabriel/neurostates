from neurostates.core.neurodata import NeuroData


def create_data_from_matrix(matrix, modality, sampling_rate=None, tr=None):
    return NeuroData.from_matrix_data(
        data=matrix, modality=modality, sampling_rate=sampling_rate, tr=tr
    )


def create_data_from_bids(bids_path):
    raise NotImplementedError()
