import os
import math
import numpy as np

from dotenv import load_dotenv

load_dotenv()
CONDITION_KEY, CELL_TYPE_KEY, CONTROL_KEY, STIMULATED_KEY = os.getenv('CONDITION_KEY'), \
                                                            os.getenv('CELL_TYPE_KEY'), \
                                                            os.getenv("CONTROL_KEY"), \
                                                            os.getenv("STIMULATED_KEY")


def get_downsampling_proportion(adata,
                                threshold=1000,
                                verbose=False):
    np.random.seed(43)
    if verbose:
        print(f'Dataset size before downsampling: {adata.shape[0]} cells')
    # Downsampling by minimum threshold
    index_all = []
    cell_types = adata.obs[CELL_TYPE_KEY].drop_duplicates().to_list()
    if verbose:
        print('\n')
        print('-' * 10, "Select/Remove distribution", '-' * 10)
    for condition in [CONTROL_KEY, STIMULATED_KEY]:
        if verbose:
            print('-' * 10, '\n', condition)
        adata_in_condition = adata[adata.obs[CONDITION_KEY] == condition]
        for cell_type in cell_types:
            if verbose:
                print(cell_type, end=' ')
            index = adata_in_condition[adata_in_condition.obs[CELL_TYPE_KEY] == cell_type].obs.index.values
            if verbose:
                print(len(index), end=' | ')
            index_stay, index_hold_out = index[:threshold], index[threshold:]
            if verbose:
                print(len(index_stay), len(index_hold_out))
            index_all.append(index_stay)
    if verbose:
        print('-' * 10, "Percentage distribution", '-' * 10)
    # Display control percentage
    indices_control = index_all[:len(cell_types)]
    indices_control_total_size = len(np.concatenate(indices_control))
    if verbose:
        print('-' * 10, '\n', CONTROL_KEY)
        for i, index_control in enumerate(indices_control):
            print(f"{cell_types[i]}: {(len(index_control) / indices_control_total_size) * 100:.2f}%")
    # Display stimulated percentage
    indices_stimulated = index_all[len(cell_types):]
    indices_stimulated_total_size = len(np.concatenate(indices_stimulated))
    if verbose:
        print('-' * 10, '\n', STIMULATED_KEY)
        for i, index_stimulated in enumerate(indices_stimulated):
            print(f"{cell_types[i]}: {(len(index_stimulated) / indices_stimulated_total_size) * 100:.2f}%")
    index_all = np.concatenate(index_all)
    adata_sample = adata[index_all]
    if verbose: 
        print(f'\nDataset size after downsampling: {adata_sample.shape[0]} cells')
    return adata_sample


def get_downsampling_minimum(adata,
                             exclude_cell_type=None,
                             verbose=False):
    threshold, min_cell_type = (+math.inf, None)
    for condition in [STIMULATED_KEY, CONTROL_KEY]:
        adata_in_condition = adata[adata.obs[CONDITION_KEY] == condition]
        for cell_type in adata.obs[CELL_TYPE_KEY].drop_duplicates().to_list():
            if cell_type != exclude_cell_type:
                index_size = len(
                    adata_in_condition[adata_in_condition.obs[CELL_TYPE_KEY] == cell_type].obs.index.values)
                if index_size != 0 and threshold > index_size:
                    threshold = index_size
                    min_cell_type = cell_type
    if verbose:
        print(f"Minimum cell type {min_cell_type} with size: {threshold}")
    if verbose:
        print(f"Excluding {exclude_cell_type} cell type from dataset for downsampling...")
    return get_downsampling_proportion(adata,
                                       threshold,
                                       verbose)
