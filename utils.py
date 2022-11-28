import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()
CONDITION_KEY, CELL_TYPE_KEY = os.getenv('CONDITION_KEY'), os.getenv('CELL_TYPE_KEY')
CONTROL_KEY, STIMULATED_KEY, PREDICTED_KEY = os.getenv('CONTROL_KEY'), os.getenv('STIMULATED_KEY'), os.getenv(
    'PREDICTED_KEY')


def get_sample(adata, sample_size=1000):
    barcodes = adata.obs.index.values
    return adata[np.random.choice(barcodes, sample_size), :]


def remove_stimulated_for_celltype(adata, celltype):
    return adata[~((adata.obs[CELL_TYPE_KEY] == celltype) &
                   (adata.obs[CONDITION_KEY] == "stimulated"))].copy()


def extractor(adata, cell_type):
    """
    Returns a list of `data` files while filtering for a specific `cell_type`.
    """
    cell_with_both_condition = adata[adata.obs[CELL_TYPE_KEY] == cell_type]
    condition_1 = adata[
        (adata.obs[CELL_TYPE_KEY] == cell_type) & (adata.obs[CONDITION_KEY] == CONTROL_KEY)
        ]
    condition_2 = adata[
        (adata.obs[CELL_TYPE_KEY] == cell_type) & (adata.obs[CONDITION_KEY] == STIMULATED_KEY)
        ]
    training = adata[
        ~(
                (adata.obs[CELL_TYPE_KEY] == cell_type)
                & (adata.obs[CONDITION_KEY] == STIMULATED_KEY)
        )
    ]
    return [training, condition_1, condition_2, cell_with_both_condition]


def balancer(adata):
    """
    Makes cell type population equal.
    """
    class_names = np.unique(adata.obs[CELL_TYPE_KEY])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = adata[adata.obs[CELL_TYPE_KEY] == cls].shape[0]
    max_number = np.max(list(class_pop.values()))
    index_all = []
    for cls in class_names:
        class_index = np.array(adata.obs[CELL_TYPE_KEY] == cls)
        index_cls = np.nonzero(class_index)[0]
        index_cls_r = index_cls[np.random.choice(len(index_cls), max_number)]
        index_all.append(index_cls_r)
    balanced_data = adata[np.concatenate(index_all)].copy()
    return balanced_data
