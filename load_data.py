import os

import scanpy
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor





def save_kang():
    train = scanpy.read("./tests/data/train_kang.h5ad",
                backup_url='https://drive.google.com/uc?id=1r87vhoLLq6PXAYdmyyd89zG90eJOFYLk')
    train.write(os.path.join("data", "train_kang.h5ad"))



class scDataset(Dataset):
    """
        Constructs custom Dataset from single-cell anndata array to be used for training VAE model.
    """
    def __init__(self, ann_array, transform=None):
        self.X = ann_array.to_df().values
        self.y = ann_array.obs['condition'].apply(lambda x: 0 if x == 'control' else 1).values
        self.transform = transform
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx, :], self.y[idx]


def get_adata(dataset="kang", train=True, verbose=False):
    adata = None
    train_or_test = "train" if train else "test"
    if dataset == "kang":
        adata = scanpy.read_h5ad(os.path.join("data", f"{train_or_test}_kang.h5ad"))
        if verbose:
            print(adata)
    return adata


def get_dataset_torch(adata):
    dataset = scDataset(adata,      # get_sample(train_new) # You can train on a sample (to decrease duration of training)
                        transform=ToTensor())
    return dataset


def get_dataloader_torch(dataset, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)




if __name__ == "__main__":
    # save_kang()
    pass