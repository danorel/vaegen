import os
import numpy as np

from dotenv import load_dotenv

load_dotenv()
CONDITION_KEY, CELL_TYPE_KEY = os.getenv(
    'CONDITION_KEY'), os.getenv('CELL_TYPE_KEY')
CONTROL_KEY, STIMULATED_KEY, PREDICTED_KEY = os.getenv('CONTROL_KEY'), os.getenv('STIMULATED_KEY'), os.getenv(
    'PREDICTED_KEY')


def create_model_dir(name):
    model_directory = os.path.join("models", name)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    pass


def get_sample(adata, cell_type=None, sample_size=1000):
    adata_target = adata.copy()
    if cell_type:
        adata_target = adata_target[adata.obs[CELL_TYPE_KEY] == cell_type]
    barcodes = adata_target.obs.index.values
    indices = np.random.choice(barcodes, sample_size)
    return adata_target[indices, :]


def remove_cell_type(adata, cell_type, verbose=False):
    if verbose:
        print("-" * 10 + "Before" + "-" * 10)
        print(adata)
    adata_no_cell_type = adata[~(adata.obs[CELL_TYPE_KEY] == cell_type)].copy()
    if verbose:
        print("-" * 10 + "After" + "-" * 10)
        print(adata_no_cell_type)
    return adata_no_cell_type


def remove_stimulated_for_cell_type(adata, cell_type, verbose=False):
    if verbose:
        print("-" * 10 + "Before" + "-" * 10)
        print(adata)
    adata_no_stimulated_cell_type = adata[~((adata.obs[CELL_TYPE_KEY] == cell_type) &
                                            (adata.obs[CONDITION_KEY] == STIMULATED_KEY))].copy()
    if verbose:
        print("-" * 10 + "After" + "-" * 10)
        print(adata_no_stimulated_cell_type)
    return adata_no_stimulated_cell_type


def remove_stimulated_for_cell_types(adata, cell_types, sample_size=None, verbose=False):
    if verbose:
        print("-" * 10 + "Before" + "-" * 10)
        print(adata[adata.obs[CONDITION_KEY] == STIMULATED_KEY])
    adata_modified = adata[~((adata.obs[CELL_TYPE_KEY].isin(cell_types)) &
                             (adata.obs[CONDITION_KEY] == STIMULATED_KEY))].copy()
    if verbose:
        print("-" * 10 + "After" + "-" * 10)
        print(
            adata_modified[adata_modified.obs[CONDITION_KEY] == STIMULATED_KEY])
    return adata_modified


def extractor(adata, cell_type):
    """
    Returns a list of `data` files while filtering for a specific `cell_type`.
    """
    cell_with_both_condition = adata[adata.obs[CELL_TYPE_KEY] == cell_type]
    condition_1 = adata[
        (adata.obs[CELL_TYPE_KEY] == cell_type) & (
            adata.obs[CONDITION_KEY] == CONTROL_KEY)
    ]
    condition_2 = adata[
        (adata.obs[CELL_TYPE_KEY] == cell_type) & (
            adata.obs[CONDITION_KEY] == STIMULATED_KEY)
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
