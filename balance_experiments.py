import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import anndata

from load_data import get_adata
from balance import balance_classes_up_inside_segments, balance_classes_down_inside_segments, balance_classes_inside_segments
from sc_condition_prediction import create_and_train_vae_model, evaluate_r2_custom, N_INPUT, N_LAYERS, N_HIDDEN, N_LATENT


import warnings
warnings.filterwarnings('ignore')

# Initialize constants
load_dotenv()
CONDITION_KEY, CELL_TYPE_KEY = os.getenv(
    'CONDITION_KEY'), os.getenv('CELL_TYPE_KEY')

model_params = dict(n_input=N_INPUT,
                    n_layers=N_LAYERS,
                    n_hidden=N_HIDDEN,
                    n_latent=N_LATENT)


def sample_index(adata, n_sample):
    index = adata.obs.index.values
    np.random.shuffle(index)
    index_sample = index[:n_sample]
    return index_sample


def train_valid_test_split(adata, verbose=False):
    n_sample_per_cell = 200
    adata_val = None
    index_val = None

    # 1. Test set
    adata_test = adata[adata.obs['cell_type'] == 'CD4T']

    # 2. Validation set
    for cell_type in ['CD14+Mono', 'B', 'FCGR3A+Mono']:
        adata_cell = adata[adata.obs['cell_type'] == cell_type]
        for condition in ('control', 'stimulated'):
            adata_cell_condition = adata_cell[adata_cell.obs['condition'] == condition]
            index_celltype_condition = sample_index(
                adata_cell_condition, n_sample_per_cell)
            if index_val is None:
                index_val = index_celltype_condition
            else:
                index_val = np.concatenate(
                    [index_val, index_celltype_condition])
    adata_val = adata[index_val]

    # 3. Train
    index_valid_exclude = adata_val[adata_val.obs['condition']
                                    == 'stimulated'].obs.index
    index_test_exclude = adata_test[adata_test.obs['condition']
                                    == 'stimulated'].obs.index
    index_train = [i for i in adata.obs.index if (
        i not in index_valid_exclude) and (i not in index_test_exclude)]
    adata_train = adata[index_train]
    if verbose:
        print('Train valid test split:')
        print(f'\tAdata original: {adata.shape[0]}')
        print(f'\tTrain         : {adata_train.shape[0]}')
        print(f'\tValidation    : {adata_val.shape[0]}')
        print(f'\tTest          : {adata_test.shape[0]}')
    return adata_train, adata_val, adata_test


# --- Experiment #1 ----
# ----------------------


def get_balanced_dataset_name(n):
    if isinstance(n, float):
        if n == -1.0:
            return 'balance downsampling'
        elif n == 1.0:
            return 'balance upsampling'
        elif n == 0.0:
            return 'original dataset'
    elif isinstance(n, int):
        return f'balance to {n} instances'


def get_balanced_datasets(adata, balance_schemas):
    adatas = {}
    for n in balance_schemas:
        if isinstance(n, float):
            if n == -1.0:
                adatas[get_balanced_dataset_name(
                    n)] = balance_classes_down_inside_segments(adata)
            elif n == 1.0:
                adatas[get_balanced_dataset_name(
                    n)] = balance_classes_up_inside_segments(adata)
            elif n == 0.0:
                adatas[get_balanced_dataset_name(n)] = adata
        elif isinstance(n, int):
            adatas[get_balanced_dataset_name(
                n)] = balance_classes_inside_segments(adata, n=n)
    return adatas


def experiment_upsampling(adata,
                          cv_count=5,
                          save_to_filename='upsampling_effect_CV.csv',
                          balance_schemas=[0., -1., 600, 800, 1200, 1600, +1.],
                          n_epochs=15):
    print('STARTED')
    print(f'CV: {cv_count} folded')
    print(f'balance_schemas: {balance_schemas}')
    print('Save to: ', save_to_filename)
    print('Epochs for training: ', n_epochs)
    balance_schemas_names = [
        get_balanced_dataset_name(n) for n in balance_schemas]

    # Create table for storing the scores
    index = pd.MultiIndex.from_product(
        [balance_schemas_names, ['validation', 'test']])
    df_results = pd.DataFrame(data=np.zeros(
        (len(index), cv_count)), index=index, columns=range(cv_count))

    for i_cv in range(cv_count):
        print(f'CV Fold: {i_cv}/{cv_count-1}')
        train_adata, valid_adata, test_adata = train_valid_test_split(adata)
        valid_ctrl, valid_stim = valid_adata[valid_adata.obs['condition'] ==
                                             'control'], valid_adata[valid_adata.obs['condition'] == 'stimulated']
        test_ctrl, test_stim = test_adata[test_adata.obs['condition'] ==
                                          'control'],   test_adata[test_adata.obs['condition'] == 'stimulated']

        adatas = get_balanced_datasets(train_adata, balance_schemas)

        for i, (dataset_name, adata_train_i) in enumerate(adatas.items()):
            print(f'\t{dataset_name} ... ', end='')
            params_filename = os.path.join(
                "result", "1 Effect of upsampling", f"{i}_temp_autoencoder.pt")
            create_and_train_vae_model(adata_train_i,
                                       epochs=n_epochs,
                                       save_params_to_filename=params_filename,
                                       verbose=False,
                                       **model_params)
            r2_valid, _ = evaluate_r2_custom(
                adata_train_i, valid_ctrl, valid_stim, params_filename, **model_params)
            r2_test, _ = evaluate_r2_custom(
                adata_train_i,  test_ctrl,  test_stim, params_filename, **model_params)
            df_results.loc[(dataset_name, 'validation'), i_cv] = r2_valid
            df_results.loc[(dataset_name,       'test'), i_cv] = r2_test
            df_results.to_csv(os.path.join(
                'result',  "1 Effect of upsampling", save_to_filename))
            print('✔️')
        print()
    print('FINISHED')


# --- Experiment #4 ----
# ----------------------

def get_adatas_diversity_experiment(adata, data_schemas):
    # Leave only types 'CD4T', 'FCGR3A+Mono', 'B'. CD4T will be used for evaluation, it's necessary to include it into model.
    adata = adata[adata.obs['cell_type'].isin(['CD4T', 'FCGR3A+Mono', 'B'])]
    adatas = []
    for data_schema in data_schemas:
        adata_new = None
        for cell_type, n in data_schema.items():
            for condition in ['control', 'stimulated']:
                index = adata[(adata.obs['cell_type'] == cell_type) & (
                    adata.obs['condition'] == condition)].obs.index.values
                try:
                    new_index = np.random.choice(index, n)
                    if adata_new is None:
                        adata_new = adata[new_index]
                    else:
                        adata_new = anndata.AnnData.concatenate(
                            adata_new, adata[new_index])
                except ValueError as e:
                    print(data_schema, cell_type, condition)
        adatas.append(adata_new)
    return adatas


def experiment_diversity(adata,
                         cv_count=5,
                         save_to_filename='diversity.csv',
                         data_schemas=None,
                         n_epochs=15):
    print('STARTED: Experiment \"Diversity impact on model training\"')
    print(f'CV: {cv_count} folded')
    print('Save to: ', save_to_filename)
    print('Epochs for training: ', n_epochs)

    if data_schemas is None:
        data_schemas = [
            {'FCGR3A+Mono': 800},
            {'FCGR3A+Mono': 600, 'B': 200},
            {'FCGR3A+Mono': 400, 'B': 400},
            {'FCGR3A+Mono': 200, 'B': 600},
            {'B': 800}
        ]
    data_schemas_names = [
        ', '.join([f'{v} {k}' for k, v in d.items()]) for d in data_schemas]

    # Create table for storing the scores
    index = pd.MultiIndex.from_product(
        [data_schemas_names, ['validation', 'test']])
    df_results = pd.DataFrame(data=np.zeros(
        (len(index), cv_count)), index=index, columns=range(cv_count))

    for i_cv in range(cv_count):
        print(f'CV Fold: {i_cv}/{cv_count-1}')
        train_adata, valid_adata, test_adata = train_valid_test_split(adata)
        valid_ctrl, valid_stim = valid_adata[valid_adata.obs['condition'] ==
                                             'control'], valid_adata[valid_adata.obs['condition'] == 'stimulated']
        test_ctrl, test_stim = test_adata[test_adata.obs['condition'] ==
                                          'control'],   test_adata[test_adata.obs['condition'] == 'stimulated']
        adatas = get_adatas_diversity_experiment(train_adata, data_schemas)

        for i, adata_train_i in enumerate(adatas):
            print(f'\tData schema : {data_schemas_names[i]} ... ', end='')
            params_filename = os.path.join(
                "models", "imbalance_test", f"temp_autoencoder.pt")

            create_and_train_vae_model(adata_train_i,
                                       epochs=15,
                                       save_params_to_filename=params_filename,
                                       verbose=False,
                                       **model_params)
            r2_valid, _ = evaluate_r2_custom(adata_train_i.concatenate(
                valid_ctrl), valid_ctrl, valid_stim, params_filename, **model_params)
            r2_test, _ = evaluate_r2_custom(adata_train_i.concatenate(
                test_ctrl),  test_ctrl,  test_stim, params_filename, **model_params)
            df_results.loc[(data_schemas_names[i],
                            'validation'), i_cv] = r2_valid
            df_results.loc[(data_schemas_names[i],
                            'test'), i_cv] = r2_test
            df_results.to_csv(os.path.join(
                'result', '2 Diversity', save_to_filename))
            print('✔️')
        print()
    print('FINISHED')


if __name__ == '__main__':
    # TODO: execute script, uncomment the experiment that you want to run:

    adata = get_adata(dataset="kang", train=True)
    np.random.seed(43)

    # EXPERIMENT 1: Upsampling (and other balancing) effect on training
    # experiment_upsampling(adata=adata,
    #                       cv_count=10,
    #                       save_to_filename='[cv 1].csv',
    #                       balance_schemas=[0., -1., 600, 800, 1000, 1200, 1600, 2000, +1.],
    #                       n_epochs=15)

    # Experiment 4: Diversity
    data_schemas = [
        {'FCGR3A+Mono': 800},
        {'FCGR3A+Mono': 600, 'B': 200},
        {'FCGR3A+Mono': 400, 'B': 400},
        {'FCGR3A+Mono': 200, 'B': 600},
        {'B': 800}
    ]
    experiment_diversity(adata=adata,
                         cv_count=10,
                         save_to_filename='[cv 1].csv',
                         data_schemas=data_schemas,
                         n_epochs=15)
