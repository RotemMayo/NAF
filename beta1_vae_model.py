import numpy as np
import pandas as pd
from torch import nn
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import gc as gc
from tqdm import tqdm
import sys


def loss(output):
        """
        KLD of the output with a centered spherical gaussian distribution of the latent space dimension
        output: (mean, logvar) should be the latent space output
        """
        mu, logvar = output
        KLD_element = 1 - logvar.exp() + logvar - mu**2
        KLD = - 0.5 * torch.sum(KLD_element, 1)
        return KLD


class VAE(nn.Module):

    def __init__(self, layer_dims, dropout=0.):
        super(VAE, self).__init__()
        self.layer_dims = layer_dims
        self.dropout = dropout
        self.encoder = nn.Sequential(nn.Linear(layer_dims[0], layer_dims[1]),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(layer_dims[1], layer_dims[2]),
                                     nn.ReLU(),
                                     nn.Dropout(dropout))
        self.latent_dim = layer_dims[-1]
        self.mu = nn.Sequential(nn.Linear(layer_dims[-2], layer_dims[-1]),
                                nn.Tanh())
        self.logvar = nn.Sequential(nn.Linear(layer_dims[-2], layer_dims[-1]),
                                    nn.Tanh())
        self.criterion = loss

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.encode(x)

    def density(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return torch.sum(- z**2, 1)

    def get_losses(self, x):
        output = self.forward(x)
        return self.criterion(output)
    
    def train_it(self, dataloader, optimizer, epochs=10):
        losses = []
        for epoch in tqdm(range(epochs)):
            for x in dataloader:
                # Forward
                batch_loss = torch.mean(self.get_losses(x))
                # Back-prop
                if not torch.isinf(batch_loss) and not torch.isnan(batch_loss):
                    optimizer.zero_grad()
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 5) #clip gradients
                    optimizer.step()
                    losses.append(batch_loss.item())
                    gc.collect()
        return losses


class NumpyDataset(torch.utils.data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """

    def __init__(self, data, normalize=True):
        """
        param data: a numpy array
        """
        self.length = data.shape[0]
        self.original_data = np.nan_to_num(data).astype(np.float32)
        if normalize:
            n_features = self.original_data.shape[1]
            self.mean = self.original_data.mean(axis=0).reshape(1, n_features)
            self.std = self.original_data.std(axis=0).reshape(1, n_features)
            self.data = torch.Tensor((self.original_data - self.mean) / self.std)
        else:
            self.data = torch.Tensor(self.original_data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index]


def train_and_save_model(ds, cp_path, dropout=0., batch_size=128, epochs=100):
    layer_dims = [14, 12, 6, 3]
    lr = 1.e-5
    model = VAE(layer_dims, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = NumpyDataset(ds)
    dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses = model.train_it(dataloader, optimizer, epochs)
    cp_dict = {
        'epochs': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'layer_dims': layer_dims,
        'dropout': dropout,
        'lr': lr
    }
    torch.save(cp_dict, cp_path)


def get_n_signal_dataset(n_sig, df):
    np.random.seed(42)
    signal = df['label'] == 1
    print(df.index[signal])
    indices = np.random.choice(df.index[signal], n_sig, replace=False)
    mask = np.full(df.shape[0], False)
    mask[df.index[~signal]] = True
    mask[indices] = True
    return df[mask].iloc[:, :-3].to_numpy(dtype=np.float32)


def main():
    df = pd.read_csv('external_maf/datasets/data/lhc/lhc_features_and_bins.csv')
    n_sig = int(sys.argv[1])
    ds = get_n_signal_dataset(n_sig, df)
    cp_path = 'models/vae_lhc_141263_nsig{}.pt'.format(n_sig)
    train_and_save_model(ds, cp_path, dropout=0.05)


if __name__ == "__main__":
    main()
