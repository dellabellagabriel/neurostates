# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2024, Della Bella, Gabriel; Rodriguez, Natalia
# All rights reserved.

"""clustering performs a clustering algortithm over the data processed by
connectivity"""

from .utils import validate_groups_dict

from sklearn.cluster import KMeans


def clustering(groups_dict, n_clusters, **clustering_kwargs):
    """_summary_

    Args:
        groups_dict (_type_): _description_
        n_clusters (_type_): _description_
    """
    
    validate_groups_dict(groups_dict)
    
    # TODO: concatenar todos los grupos
    # data_concatenated = ....
    for group in groups_dict:
        subjects, windows, rois, _ = groups_dict[group].shape
        data = groups_dict[group].reshape(subjects, windows, -1).reshape(subjects*windows, -1)
        
    
    kmeans = KMeans(n_clusters=n_clusters, **clustering_kwargs)
    # kmeans.fit(data_concatenated)
    
    return kmeans.cluster_centers_
    
