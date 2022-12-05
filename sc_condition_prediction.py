import os

import numpy as np
from dotenv import load_dotenv
import scanpy as sc
import torch
from torch.distributions import kl_divergence, Normal

from VAE import VAE, DEVICE
from load_data import get_adata, get_dataset_torch, get_dataloader_torch
from utils import extractor, balancer, remove_stimulated_for_celltype
from plotting import reg_mean_plot


# Loading AnnData dataset with single-cell data 
train_adata = get_adata(dataset="kang", train=True)
train_new = remove_stimulated_for_celltype(train_adata, celltype="CD4T")

# Initializing constants
load_dotenv()
CONDITION_KEY, CELL_TYPE_KEY = os.getenv('CONDITION_KEY'), os.getenv('CELL_TYPE_KEY')
CONTROL_KEY, STIMULATED_KEY, PREDICTED_KEY = os.getenv('CONTROL_KEY'), os.getenv('STIMULATED_KEY'), os.getenv('PREDICTED_KEY')

N_INPUT = n_input = train_new.shape[1] # number of features, dimensionality of an input space
N_HIDDEN = 100  # size of a hidden layer
N_LAYERS = 3    # number of hidden layers in fully-connected NN 
N_LATENT = 10   # dimensionality of latent space
BATCH_SIZE = 32



def train(autoencoder, dataloader, epochs=20, verbose=False):
    """
        Trains Variational Autoencoder. 
    """
    kl_weight = autoencoder.kl_weight
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        if verbose: print(f"Epoch {epoch} is running...", end='')
        for x, y in dataloader:
            # initialization
            x = x.to(DEVICE)
            opt.zero_grad()
            # forward
            qz_m, qz_v, z = autoencoder.encoder(x)
            x_hat = autoencoder.decoder(z)
            # loss & backward
            kl_div = kl_divergence(
                Normal(qz_m, torch.sqrt(qz_v)),
                Normal(0, 1),
            ).sum(dim=1)
            reconstruction_loss = ((x - x_hat)**2).sum()
            loss = (0.5*reconstruction_loss + 0.5*(kl_div * kl_weight)).mean()
            loss.backward()
            # optimization step
            opt.step()
        if verbose: print('loss: {:.3f}'.format(loss))
    return autoencoder




def create_and_train_vae_model(adata, epochs=20, verbose=False, save_params_to_filename='autoencoder.pt'):
    np.random.seed(43)
    # DATASET & DATALOADER
    dataset = get_dataset_torch(adata)
    dataloader = get_dataloader_torch(dataset, batch_size=BATCH_SIZE)
    # CREATE model
    autoencoder = VAE(n_input=N_INPUT, 
                      n_layers=N_LAYERS, 
                      n_hidden=N_HIDDEN, 
                      n_latent=N_LATENT)
    # TRAIN
    autoencoder = train(autoencoder, 
                        dataloader=dataloader, 
                        epochs=epochs,
                        verbose=verbose)
    # SAVE parameters
    torch.save(autoencoder.state_dict(), save_params_to_filename)


def load_vae_model(filename='autoencoder.pt'):
    """
        Returns pretrained VAE model
    """
    autoencoder = VAE(n_input=N_INPUT, 
                      n_layers=N_LAYERS, 
                      n_hidden=N_HIDDEN, 
                      n_latent=N_LATENT)
    autoencoder.load_state_dict(torch.load(filename))
    return autoencoder
    
    

def get_latent_representation(autoencoder, adata, as_numpy=False):
    _, _, latent_X = autoencoder.encoder(torch.tensor(adata.to_df().values))
    if as_numpy:
        return latent_X.detach().numpy()
    else:
        return sc.AnnData(X=latent_X.detach().numpy(), obs=adata.obs.copy())




def predict(autoencoder, 
            adata, 
            celltype_to_predict):
    ctrl_x = adata[adata.obs[CONDITION_KEY] == CONTROL_KEY, :]
    stim_x = adata[adata.obs[CONDITION_KEY] == STIMULATED_KEY, :]
    
    # balance control and stimulated dataset
    ctrl_x = balancer(adata=ctrl_x)
    stim_x = balancer(adata=stim_x)
    
    # Get control adata (we predict stimulated for it)
    ctrl_pred = extractor(adata, celltype_to_predict)[1]

    # Equalize the sized od control and stimulated dataset
    eq = min(ctrl_x.X.shape[0], stim_x.X.shape[0])
    cd_ind = np.random.choice(range(ctrl_x.shape[0]), size=eq, replace=False)
    stim_ind = np.random.choice(range(stim_x.shape[0]), size=eq, replace=False)
    ctrl_adata = ctrl_x[cd_ind, :]
    stim_adata = stim_x[stim_ind, :]

    # compute mean of control/stimulated in latent space
    latent_ctrl = np.mean(get_latent_representation(autoencoder, ctrl_adata, as_numpy=True), axis=0)
    latent_stim = np.mean(get_latent_representation(autoencoder, stim_adata, as_numpy=True), axis=0)

    delta = latent_stim - latent_ctrl
    
    # get latent representation of anndata we want to predict stimulated for
    latent_cd = get_latent_representation(autoencoder, ctrl_pred, as_numpy=True)

    stim_pred = delta + latent_cd
    
    # decode predicted stimulated
    predicted_cells = (
        autoencoder.decoder(torch.Tensor(stim_pred)).cpu().detach().numpy()
    )

    predicted_adata = sc.AnnData(
        X=predicted_cells,
        obs=ctrl_pred.obs.copy(),
        var=ctrl_pred.var.copy(),
        obsm=ctrl_pred.obsm.copy(),
    )

    predicted_adata.obs[CONDITION_KEY] = PREDICTED_KEY
    return predicted_adata, delta



def evaluate_r2(params_filename):
    autoencoder = load_vae_model(params_filename)

    # Predict stimulated cells from control and visualize
    ctrl_adata = train_adata[((train_adata.obs[CELL_TYPE_KEY] == 'CD4T') & (train_adata.obs[CONDITION_KEY] == CONTROL_KEY))]
    stim_adata = train_adata[((train_adata.obs[CELL_TYPE_KEY] == 'CD4T') & (train_adata.obs[CONDITION_KEY] == STIMULATED_KEY))]

    predicted_adata, delta = predict(autoencoder, train_new, celltype_to_predict='CD4T')
    
    eval_adata = ctrl_adata.concatenate(stim_adata, predicted_adata)

    CD4T = train_adata[train_adata.obs[CELL_TYPE_KEY] == "CD4T"]
    sc.tl.rank_genes_groups(CD4T, groupby=CONDITION_KEY, method="wilcoxon")
    diff_genes = CD4T.uns["rank_genes_groups"]["names"][STIMULATED_KEY]

    r2_value, r2_value_diff_genes = reg_mean_plot(
        eval_adata,
        axis_keys={"x": PREDICTED_KEY, "y": STIMULATED_KEY},
        gene_list=diff_genes[:10],
        top_100_genes=diff_genes,
        labels={"x": "predicted", "y": "ground truth"},
        path_to_save=os.path.join("figures", "archive", "reg_mean_top100_genes.pdf"),
        show=False,
        legend=False
    )

    return r2_value, r2_value_diff_genes


def evaluate(show_plots=True):
    autoencoder = load_vae_model()
    
    # Plot lattent representation of cell data.
    _, _, latent_X = autoencoder.encoder(torch.tensor(train_new.to_df().values))

    latent_adata = sc.AnnData(X=latent_X.detach().numpy(), obs=train_new.obs.copy())
    sc.pp.neighbors(latent_adata)
    sc.tl.umap(latent_adata)
    sc.pl.umap(latent_adata, color=[CONDITION_KEY, CELL_TYPE_KEY], wspace=0.4, frameon=False,
            save='_latent_space.pdf', show=show_plots)

    # Predict stimulated cells from control and visualize
    ctrl_adata = train_adata[((train_adata.obs[CELL_TYPE_KEY] == 'CD4T') & (train_adata.obs[CONDITION_KEY] == CONTROL_KEY))]
    stim_adata = train_adata[((train_adata.obs[CELL_TYPE_KEY] == 'CD4T') & (train_adata.obs[CONDITION_KEY] == STIMULATED_KEY))]

    predicted_adata, delta = predict(autoencoder, train_new, celltype_to_predict='CD4T')
    
    eval_adata = ctrl_adata.concatenate(stim_adata, predicted_adata)

    sc.tl.pca(eval_adata)   
    sc.pl.pca(eval_adata, color=CONDITION_KEY, frameon=False, 
              save='_pred_eval.pdf', show=show_plots)

    # Mean correlation plot``
    CD4T = train_adata[train_adata.obs[CELL_TYPE_KEY] =="CD4T"]
    sc.tl.rank_genes_groups(CD4T, groupby=CONDITION_KEY, method="wilcoxon")
    diff_genes = CD4T.uns["rank_genes_groups"]["names"][STIMULATED_KEY]
    print(diff_genes)

    r2_value = reg_mean_plot(
        eval_adata,
        axis_keys={"x": PREDICTED_KEY, "y": STIMULATED_KEY},
        gene_list=diff_genes[:10],
        labels={"x": "predicted", "y": "ground truth"},
        path_to_save=os.path.join("figures", "reg_mean.pdf"),
        show=show_plots,
        legend=False
    )

    r2_value_diff_genes = reg_mean_plot(
        eval_adata,
        axis_keys={"x": PREDICTED_KEY, "y": STIMULATED_KEY},
        gene_list=diff_genes[:10],
        top_100_genes= diff_genes,
        labels={"x": "predicted", "y": "ground truth"},
        path_to_save=os.path.join("figures", "reg_mean_top100_genes.pdf"),
        show=show_plots,
        legend=False
    )

    # Violin plot  or a specific gene
    sc.pl.violin(eval_adata, 
                 keys="ISG15", groupby=CONDITION_KEY, 
                 save='_violin_ISG15.pdf',
                 show=show_plots)

    return r2_value, r2_value_diff_genes


if __name__ == "__main__":
    create_and_train_vae_model(train_new, epochs=20)
    evaluate(show_plots=False)
