
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torch.distributions
from torch.distributions import Normal


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class FCLayers(nn.Module):
    """
        A fully-connected neural network model. 
        It is a base component for the Encoder and Decoder parts of Variational Autoencoder. 
        FCLayers doesn't include an output layer to the latent space (and to the original feature space as well).
    """
    def __init__(self, n_input, n_layers, n_hidden, dropout_rate):
        super(FCLayers, self).__init__()
        modules = []
        hidden_dims = [n_hidden]*n_layers
        for in_size, out_size in zip([n_input]+hidden_dims, hidden_dims):
            modules.append(nn.Linear(in_size, out_size, bias=True))
            modules.append(nn.BatchNorm1d(out_size, momentum=0.01, eps=0.001))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=dropout_rate))
        self.fc = nn.Sequential(*modules)

    def forward(self, *inputs):
        input_cat = torch.cat(inputs, dim=-1)
        return self.fc(input_cat)


class Encoder(nn.Module):
    def __init__(self, n_input, n_layers, n_hidden, n_latent, dropout_rate=0.1, distribution='normal'):
        super(Encoder, self).__init__()
        self.fc = FCLayers(n_input, n_layers, n_hidden, dropout_rate)
        self.mean_encoder = nn.Linear(n_hidden, n_latent)
        self.var_encoder = nn.Linear(n_hidden, n_latent)
        self.var_activation = torch.exp
        
    def forward(self, x, *cat_list):
        q = self.fc(x)
        qz_m = self.mean_encoder(q)
        qz_v = self.var_activation(self.var_encoder(q))  # we often apply an activation function exp() on variation to ensure positivity (more: https://avandekleut.github.io/vae/)
        latent = Normal(qz_m, torch.sqrt(qz_v)).rsample() # reparametrized sample, allows differentiation (see more: https://stackoverflow.com/questions/60533150/what-is-the-difference-between-sample-and-rsample)
        return qz_m, qz_v, latent
    
    
class Decoder(nn.Module):
    def __init__(self, n_latent, n_layers, n_hidden, n_output, dropout_rate=0.2):
        super(Decoder, self).__init__()
        self.fc = FCLayers(n_latent, n_layers, n_hidden, dropout_rate)
        self.linear_out = nn.Linear(n_hidden, n_output) # the last layer - to map results of FC neural network to original space, decode
    
    def forward(self, x, *cat_list):
        p = self.linear_out(self.fc(x, *cat_list))
        return p
    

class VAE(nn.Module):
    def __init__(self, n_input, n_layers=2, n_hidden=100, n_latent=10, kl_weight=0.00005):
        """
        Parameters
            n_input: dimensionality of an input space
            n_hidden: size of a hidden layer
            n_latent: dimensionality of latent space
            n_layers: number of hidden layers in fully-connected NN 
        """
        super(VAE, self).__init__()
        self.encoder = Encoder(n_input, n_layers, n_hidden, n_latent)
        self.decoder = Decoder(n_latent, n_layers, n_hidden, n_output=n_input)
        self.kl_weight = kl_weight
        
    def forward(self, x, *cat_list):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

